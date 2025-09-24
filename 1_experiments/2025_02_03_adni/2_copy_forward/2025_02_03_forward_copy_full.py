import __init__
from pipeline.EvaluationManager import EvaluationManager
from pipeline.Experiment import Experiment
import wandb
import pandas as pd
import numpy as np
from pipeline.Splitters import After1VisitSplitter
from BaselineHelpers import forward_fill_median_backup
import json
from pipeline.NormalizationFilterManager import Only_Standardization
from pipeline.MetricManager import MetricManager
from plotnine import *
import matplotlib.pyplot
matplotlib.pyplot.set_loglevel("error")
from PlottingHelpers import PlotHelper


WANDB_DEBUG = False


def main():

    MIN_NR_DAYS_FORECAST = 24 * 30
    NR_DAYS_FORECAST = MIN_NR_DAYS_FORECAST

    eval_manager = EvaluationManager("2025_02_03_adni")
    experiment = Experiment("copy_forward_adni")

    # Uncomment for debug
    if WANDB_DEBUG:
        experiment.setup_wandb_debug_mode()
    else:
        experiment.setup_wandb("Copy Forward - Full Validation & Training", "Copy Forward - Full", project="UC - ADNI")


    # Get paths patientids to datasets - no validation set in ADNI
    training_set = "TRAIN"
    test_set = "TEST"

    inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
    target_cols_raw = target_cols.copy()
    
    # Get patientids
    training_full_paths, training_full_patientids = eval_manager.get_paths_to_events_in_split(training_set)
    test_full_paths, test_full_patientids = eval_manager.get_paths_to_events_in_split(test_set)

    # Load data
    test_full_constants, test_full_events = eval_manager.load_list_of_patient_dfs_and_constants(test_full_patientids)

    # Setup splitter object
    splitter = After1VisitSplitter()
    
    # Setup also validation and test
    test_full_events, test_full_meta = splitter.setup_split_indices(test_full_events, eval_manager)
    


    # Load statistics
    path_to_statistics_file = experiment.base_path + "2_experiments/2025_02_03_adni/1_data/1_final_data/dataset_statistics.json"
    with open(path_to_statistics_file) as f:
        statistics_dic = json.load(f)

    
    # Setup actual copy forward model
    def model_function(constants_row, true_events_input, true_future_events_input, target_dataframe, eval_manager):

        skip_cols = ["patientid", "date", "patient_sample_index"]

        #: make target_df drop empty rows
        target_dataframe_no_empty_rows = target_dataframe.dropna(axis=0, how='all', subset=target_dataframe.columns.difference(["patientid", "patient_sample_index", "date"]))

        empty_target_dataframe = eval_manager.make_empty_df(target_dataframe_no_empty_rows)

        predicted_df = forward_fill_median_backup(true_events_input, empty_target_dataframe, skip_cols, statistics_dic)

        return predicted_df

    # Setup eval
    only_standardize = Only_Standardization(path_to_statistics_file)
    metric_manager = MetricManager(path_to_statistics_file)


    def evaluate_and_record(eval_set_events, eval_set_name, eval_meta_data):

        # Get predictions
        eval_targets, eval_prediction = experiment.get_output_for_split_generic_model(eval_set_events, eval_manager, preprocessing_and_model_and_postprocessing_function=model_function)

        # Apply numeric
        eval_prediction[target_cols_raw] = eval_prediction[target_cols_raw].apply(pd.to_numeric, errors='raise')
        eval_targets[target_cols_raw] = eval_targets[target_cols_raw].apply(pd.to_numeric, errors='raise')
        
        # Do filtering without standardizing
        eval_targets_filtered, eval_prediction_filtered = only_standardize.normalize_and_filter(eval_targets, eval_prediction)
        
        #: set grouping by therapy
        eval_targets_filtered_with_meta_data = experiment.join_meta_data_to_targets(eval_targets_filtered, eval_meta_data)

        # Calculate performance metrics
        eval_performance = metric_manager.calculate_metrics(eval_targets_filtered, eval_prediction_filtered, group_by=None)

        # Save tables locally and record in wandb
        experiment.save_df_targets_predictions_locally_and_statistics_to_wandb(eval_set_name, eval_targets_filtered, eval_prediction_filtered, meta_data_df=eval_targets_filtered_with_meta_data)

        # Save performance to wandb
        experiment.save_to_wandb_final_performances(eval_performance, eval_set_name)

        return eval_targets_filtered, eval_prediction_filtered, eval_targets_filtered_with_meta_data

    
    test_targets, test_prediction, test_meta_data = evaluate_and_record(test_full_events, test_set, test_full_meta)
    


    ############################################################ Finish run ############################################################
    wandb.run.finish()





# Call main runner
if __name__ == "__main__":
    main()



