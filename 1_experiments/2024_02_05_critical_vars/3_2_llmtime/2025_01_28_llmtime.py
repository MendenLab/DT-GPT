import __init__
from pipeline.EvaluationManager import EvaluationManager
from pipeline.Experiment import Experiment
import wandb
import pandas as pd
import numpy as np
from pipeline.Splitters import LoTSplitNDays
from BaselineHelpers import forward_fill_median_backup
import json
from pipeline.NormalizationFilterManager import Double3_sigma_Filtering_And_Standardization
from pipeline.MetricManager import MetricManager
from plotnine import *
import matplotlib.pyplot
matplotlib.pyplot.set_loglevel("error")
from PlottingHelpers import PlotHelper
import logging
import LLMTimeHelpers


#: setup correctly
WANDB_DEBUG_MODE = False


LLAMA_7B = "meta-llama/Llama-2-7b-hf"
LLAMA_70B = "meta-llama/Llama-2-70b-hf" 

ADDRESS_7B = "http://0.0.0.0:8001/v1/"
ADDRESS_70B = "http://0.0.0.0:9006/v1/"


def main(model):

    MIN_NR_DAYS_FORECAST = 91   # We want to forecast up to the first visit after this value, or until the start of the next therapy (which ever comes first) - using 91 since it is the closest multiple of 7 to 90 days - often used for meds
    MAX_LOOKBACK_POSITIONS = 91  # We want to look back up to 91 days
    NR_DAYS_FORECAST = MIN_NR_DAYS_FORECAST

    eval_manager = EvaluationManager("2024_02_05_critical_vars")
    experiment = Experiment("llmtime")

    if WANDB_DEBUG_MODE:
        experiment.setup_wandb_debug_mode()
    else:
        experiment.setup_wandb("LLMTime - Full Validation & Training - " + str(model), "LLMTime", project="UC2 - NSCLC- Critical Vars")

    
    # Get paths patientids to datasets
    training_set = "TRAIN"
    validation_set = "VALIDATION"
    test_set = "TEST"
    

    training_full_paths, training_full_patientids = eval_manager.get_paths_to_events_in_split(training_set)
    validation_full_paths, validation_full_patientids = eval_manager.get_paths_to_events_in_split(validation_set)
    test_full_paths, test_full_patientids = eval_manager.get_paths_to_events_in_split(test_set)


    # Load data
    validation_full_constants, validation_full_events = eval_manager.load_list_of_patient_dfs_and_constants(validation_full_patientids)
    test_full_constants, test_full_events = eval_manager.load_list_of_patient_dfs_and_constants(test_full_patientids)

    # Setup splitter object
    splitter = LoTSplitNDays()
    
    # Setup also validation and test
    validation_full_events, validation_full_meta = splitter.setup_split_indices(validation_full_events, eval_manager, 
                                                                                nr_days_to_forecast=NR_DAYS_FORECAST, 
                                                                                therapies_to_ignore=("Clinical Study Drug",))

    test_full_events, test_full_meta = splitter.setup_split_indices(test_full_events, eval_manager, 
                                                                    nr_days_to_forecast=NR_DAYS_FORECAST, 
                                                                    therapies_to_ignore=("Clinical Study Drug",))
    


    path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_05_critical_vars/1_data/dataset_statistics.json"
    with open(path_to_statistics_file) as f:
        statistics_dic = json.load(f)
    
    double_3_sigma_and_standardize = Double3_sigma_Filtering_And_Standardization(path_to_statistics_file)
    metric_manager = MetricManager(path_to_statistics_file)



    def evaluate_and_record(eval_set_events, eval_set_name, eval_meta_data):

        #: Get predictions
        eval_targets, eval_prediction = LLMTimeHelpers.get_output_for_llm(eval_set_events, eval_manager,
                                                                          dataset_stats=statistics_dic,
                                                                          freq="W",  # Weeks for NSCLC 
                                                                          model_to_use=model,
                                                                          max_lookback_positions=MAX_LOOKBACK_POSITIONS // 7,
                                                                          prediction_url=ADDRESS_7B if model == LLAMA_7B else ADDRESS_70B)
        
        # Do filtering without standardizing
        eval_targets_filtered, eval_prediction_filtered = double_3_sigma_and_standardize.normalize_and_filter(eval_targets, eval_prediction)
        
        #: set grouping by therapy
        eval_targets_filtered_with_meta_data = experiment.join_meta_data_to_targets(eval_targets_filtered, eval_meta_data)
        grouping_list = eval_targets_filtered_with_meta_data["line_name"].tolist()

        # Calculate performance metrics
        eval_performance = metric_manager.calculate_metrics(eval_targets_filtered, eval_prediction_filtered, group_by=grouping_list)

        # Save tables locally and record in wandb
        experiment.save_df_targets_predictions_locally_and_statistics_to_wandb(eval_set_name, eval_targets_filtered, eval_prediction_filtered, meta_data_df=eval_targets_filtered_with_meta_data)

        # Save performance to wandb
        experiment.save_to_wandb_final_performances(eval_performance, eval_set_name)

        return eval_targets_filtered, eval_prediction_filtered, eval_targets_filtered_with_meta_data


    validation_targets, validation_prediction, validation_meta_data = evaluate_and_record(validation_full_events, validation_set, validation_full_meta)
    test_targets, test_prediction, test_meta_data = evaluate_and_record(test_full_events, test_set, test_full_meta)


    ############################################################ Finish run ############################################################
    wandb.run.finish()




# Call main runner
if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()

    # Add model override
    parser.add_argument("--model", type=str, default=LLAMA_7B, help="Model to use for evaluation")

    # Parse arguments
    args = parser.parse_args()
    model = args.model

    main(model)



