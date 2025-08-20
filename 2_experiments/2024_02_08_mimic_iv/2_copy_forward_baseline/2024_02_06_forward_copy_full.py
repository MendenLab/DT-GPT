import __init__
from EvaluationManager import EvaluationManager
from Experiment import Experiment
import wandb
import pandas as pd
import numpy as np
from Splitters import After24HSplitter
from BaselineHelpers import forward_fill_median_backup
import json
from NormalizationFilterManager import Only_Standardization
from MetricManager import MetricManager
from plotnine import *
import matplotlib.pyplot
matplotlib.pyplot.set_loglevel("error")
from PlottingHelpers import PlotHelper


def main():

    MIN_NR_DAYS_FORECAST = 91   # We want to forecast up to the first visit after this value, or until the start of the next therapy (which ever comes first) - using 91 since it is the closest multiple of 7 to 90 days - often used for meds
    NR_DAYS_FORECAST = MIN_NR_DAYS_FORECAST

    eval_manager = EvaluationManager("2024_03_15_mimic_iv")
    experiment = Experiment("copy_forward_mimic_iv")

    # Uncomment for debug
    # experiment.setup_wandb_debug_mode()

    experiment.setup_wandb("Copy Forward - Full Validation & Training", "Copy Forward - Full", project="UC - MIMIC-IV")

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
    splitter = After24HSplitter()
    
    # Setup also validation and test
    validation_full_events, validation_full_meta = splitter.setup_split_indices(validation_full_events, eval_manager)

    test_full_events, test_full_meta = splitter.setup_split_indices(test_full_events, eval_manager)
    



    path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/dataset_statistics.json"
    with open(path_to_statistics_file) as f:
        statistics_dic = json.load(f)

    

    def model_function(constants_row, true_events_input, true_future_events_input, target_dataframe, eval_manager):

        skip_cols = ["patientid", "date", "patient_sample_index"]

        #: make target_df drop empty rows
        target_dataframe_no_empty_rows = target_dataframe.dropna(axis=0, how='all', subset=target_dataframe.columns.difference(["patientid", "patient_sample_index", "date"]))

        empty_target_dataframe = eval_manager.make_empty_df(target_dataframe_no_empty_rows)

        predicted_df = forward_fill_median_backup(true_events_input, empty_target_dataframe, skip_cols, statistics_dic)

        return predicted_df


    only_standardize = Only_Standardization(path_to_statistics_file)
    metric_manager = MetricManager(path_to_statistics_file)


    def evaluate_and_record(eval_set_events, eval_set_name, eval_meta_data):

        # Get predictions
        eval_targets, eval_prediction = experiment.get_output_for_split_generic_model(eval_set_events, eval_manager, preprocessing_and_model_and_postprocessing_function=model_function)
        
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

    
    validation_targets, validation_prediction, validation_meta_data = evaluate_and_record(validation_full_events, validation_set, validation_full_meta)
    test_targets, test_prediction, test_meta_data = evaluate_and_record(test_full_events, test_set, test_full_meta)
    

    ########################################################### PLOT #################################################################

    denormalized_test_targets, denormalized_test_prediction, denormalized_meta = only_standardize.denormalize(test_targets, test_prediction, test_meta_data)

    plotter = PlotHelper(dataset_statistics_json_path=experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/dataset_statistics.json", 
                    column_descriptive_mapping_path=experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/column_descriptive_name_mapping.csv")
    
    #: go over all variables
    target_cols = eval_manager.get_column_usage()[2]

    for target_col in target_cols:

        y_range_lower_bound = statistics_dic[target_col]["min"]
        y_range_upper_bound = statistics_dic[target_col]["max"]

        r = plotter.plot_mimic_trajectories(predicted_df=denormalized_test_prediction, target_df=denormalized_test_targets, meta_data=denormalized_meta, column_to_visualize=target_col, 
                                            ylims=(y_range_lower_bound, y_range_upper_bound),
                                            num_patients_to_show = 10, trajectory_alpha = 1.0, trajectory_size= 1.0)
        
        experiment.save_plotnine_image_to_wandb(r, "mimic_plot_" + str(target_col), dpi=200)


    ############################################################ Finish run ############################################################
    wandb.run.finish()




# Call main runner
if __name__ == "__main__":
    main()



