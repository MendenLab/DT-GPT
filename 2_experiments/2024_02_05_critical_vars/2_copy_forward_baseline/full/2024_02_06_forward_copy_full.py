import __init__
from EvaluationManager import EvaluationManager
from Experiment import Experiment
import wandb
import pandas as pd
import numpy as np
from Splitters import LoTSplitNDays
from BaselineHelpers import forward_fill_median_backup
import json
from NormalizationFilterManager import Double3_sigma_Filtering_And_Standardization
from MetricManager import MetricManager
from plotnine import *
import matplotlib.pyplot
matplotlib.pyplot.set_loglevel("error")
from PlottingHelpers import PlotHelper


def main():

    MIN_NR_DAYS_FORECAST = 91   # We want to forecast up to the first visit after this value, or until the start of the next therapy (which ever comes first) - using 91 since it is the closest multiple of 7 to 90 days - often used for meds
    NR_DAYS_FORECAST = MIN_NR_DAYS_FORECAST

    eval_manager = EvaluationManager("2024_02_05_critical_vars")
    experiment = Experiment("copy_forward_nsclc")


    # Uncomment for debug
    #experiment.setup_wandb_debug_mode()

    experiment.setup_wandb("Copy Forward - Full Validation & Training", "Copy Forward - Full", project="UC2 - NSCLC- Critical Vars")

    # Get paths patientids to datasets
    training_set = "TRAIN"
    validation_set = "VALIDATION"
    test_set = "TEST"
    

    training_full_paths, training_full_patientids = eval_manager.get_paths_to_events_in_split(training_set)
    validation_full_paths, validation_full_patientids = eval_manager.get_paths_to_events_in_split(validation_set)
    test_full_paths, test_full_patientids = eval_manager.get_paths_to_events_in_split(test_set)

    # Load data
    training_full_constants, training_full_events = eval_manager.load_list_of_patient_dfs_and_constants(training_full_patientids)
    validation_full_constants, validation_full_events = eval_manager.load_list_of_patient_dfs_and_constants(validation_full_patientids)
    test_full_constants, test_full_events = eval_manager.load_list_of_patient_dfs_and_constants(test_full_patientids)

    # Setup splitter object
    splitter = LoTSplitNDays()

    training_full_events_split, training_full_meta_data = splitter.setup_split_indices(training_full_events, eval_manager, 
                                                                                        nr_days_to_forecast=NR_DAYS_FORECAST,
                                                                                        therapies_to_ignore=("Clinical Study Drug",))
    
    # Setup also validation and test
    validation_full_events, validation_full_meta = splitter.setup_split_indices(validation_full_events, eval_manager, 
                                                                                nr_days_to_forecast=NR_DAYS_FORECAST, 
                                                                                therapies_to_ignore=("Clinical Study Drug",))

    test_full_events, test_full_meta = splitter.setup_split_indices(test_full_events, eval_manager, 
                                                                    nr_days_to_forecast=NR_DAYS_FORECAST, 
                                                                    therapies_to_ignore=("Clinical Study Drug",))
    


    
    ############################################################ Plot some of the training meta data ############################################################

    # Plot the total nr visits to predict
    explore_distribution = [x["nr_visits_to_predict"] for x in training_full_meta_data]
    explore_distribution = pd.DataFrame({'values': explore_distribution})

    p = ggplot(explore_distribution, aes('values'))
    p = p + geom_histogram(binwidth=1, fill="green", color="white")
    p = p + xlab("Nr of visits to predict")
    p = p + scale_x_continuous(breaks=range(0, 15, 1), limits=(0, 15))
    p = p + theme(figure_size=(4, 3))
    experiment.save_plotnine_image_to_wandb(p, "nr_visits_to_predict")

    # Plot distribution of nr Days Between Last Visit and Split Date" - values over 90 days can be since we want the first value AFTER 90 days
    explore_distribution = [x["last_visit_delta_days"] for x in training_full_meta_data]
    explore_distribution = pd.DataFrame({'values': explore_distribution})
    p = ggplot(explore_distribution, aes('values'))
    p = p + geom_histogram(binwidth=1, fill="green")
    p = p + xlab("Nr Days Between Last Visit and Split Date")
    p = p + theme(figure_size=(4, 3))
    experiment.save_plotnine_image_to_wandb(p, "nr_days_between_last_visit_and_split_date")

    # Plot difference between actual split date and the lot starting date (varince due to missing date vars)
    explore_distribution = [(x["split_date"] - x["lot_start_date"]).days for x in training_full_meta_data]
    explore_distribution = pd.DataFrame({'values': explore_distribution})
    p = ggplot(explore_distribution, aes('values'))
    p = p + geom_histogram(binwidth=1, fill="green", color="white")
    p = p + xlab("Days difference between actual split date and LoT start date")
    p = p + theme(figure_size=(6, 3))
    experiment.save_plotnine_image_to_wandb(p, "nr_days_between_lot_date_and_split_date")

    # plot distributions of "vist_days_distribution_after_split_date"
    explore_distribution = [x["vist_days_distribution_after_split_date"] for x in training_full_meta_data]
    explore_distribution = [item for sublist in explore_distribution for item in sublist]
    explore_distribution = pd.DataFrame({'values': explore_distribution})
    p = ggplot(explore_distribution, aes('values'))
    p = p + geom_histogram(binwidth=1, fill="green", color="white")
    p = p + scale_x_continuous(breaks=range(0, 98, 7), limits=(0, 98))
    p = p + xlab("Nr Days from Last Input Days")
    p = p + theme(figure_size=(6, 4))
    experiment.save_plotnine_image_to_wandb(p, "distribution_of_prediction_days")



    path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_05_critical_vars/1_data/dataset_statistics.json"
    with open(path_to_statistics_file) as f:
        statistics_dic = json.load(f)

    

    def model_function(constants_row, true_events_input, true_future_events_input, target_dataframe, eval_manager):

        skip_cols = ["patientid", "date", "patient_sample_index"]

        empty_target_dataframe = eval_manager.make_empty_df(target_dataframe)

        predicted_df = forward_fill_median_backup(true_events_input, empty_target_dataframe, skip_cols, statistics_dic)

        return predicted_df


    double_3_sigma_and_standardize = Double3_sigma_Filtering_And_Standardization(path_to_statistics_file)
    metric_manager = MetricManager(path_to_statistics_file)


    def evaluate_and_record(eval_set_events, eval_set_name, eval_meta_data):

        # Get predictions
        eval_targets, eval_prediction = experiment.get_output_for_split_generic_model(eval_set_events, eval_manager, preprocessing_and_model_and_postprocessing_function=model_function)
        
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


    ############################################################ Plot outputs ############################################################
    
    # Set up plot helper
    plotter = PlotHelper(dataset_statistics_json_path=experiment.base_path + "2_experiments/2024_02_05_critical_vars/1_data/dataset_statistics.json", 
                        column_descriptive_mapping_path=experiment.base_path + "2_experiments/2024_02_05_critical_vars/1_data/column_descriptive_name_mapping.csv")


    # Scatter plot across all variables
    p = plotter.facet_scatter_plot_all_numeric_values_by_column(test_prediction, test_targets)
    p = p + theme(figure_size=(4, 2)) + ylim(0, 25) + xlim(0, 25)
    experiment.save_plotnine_image_to_wandb(p, "plot_scatter_target_vs_prediction_all")

    # Scatter plot, stratified by most common meta data
    p = plotter.facet_scatter_plot_column_across_meta_data(predicted_df=test_prediction, 
                                                        target_df=test_targets, 
                                                        meta_data=test_meta_data, 
                                                        column_to_visualize="lab_26499_4", 
                                                        meta_data_column_with_groups="line_name", 
                                                        top_k_most_common_groups=10)
    p = p + theme(figure_size=(16, 8)) + ylim(0, 25) + xlim(0, 25)
    experiment.save_plotnine_image_to_wandb(p, "plot_scatter_target_vs_prediction_by_therapy", dpi=100)


    # Scatter plot, stratified by most common meta data
    p = plotter.facet_plot_trajectories_across_meta_data(predicted_df=test_prediction, 
                                                        target_df=test_targets, 
                                                        meta_data=test_meta_data, 
                                                        column_to_visualize="lab_26499_4", 
                                                        meta_data_column_with_groups="line_name", 
                                                        top_k_most_common_groups=10,
                                                        meta_data_column_with_starting_date="lot_start_date",
                                                        trajectory_alpha=0.2,   # Set to 0.2 from 0.7 - need to see if this is too low
                                                        trajectory_size=0.5,
                                                        xlims=(0, 91),
                                                        ylims=(0, 17))

    p = p + theme(figure_size=(10, 16))
    experiment.save_plotnine_image_to_wandb(p, "plot_trajectory_by_therapy", dpi=300)


    ############################################################ Finish run ############################################################
    wandb.run.finish()




# Call main runner
if __name__ == "__main__":
    main()



