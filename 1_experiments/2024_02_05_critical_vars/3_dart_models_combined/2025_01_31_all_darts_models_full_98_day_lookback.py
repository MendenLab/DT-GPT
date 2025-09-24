import __init__
from pipeline.EvaluationManager import EvaluationManager
from pipeline.Experiment import Experiment
import wandb
import pandas as pd
import numpy as np
from pipeline.Splitters import LoTSplitNDays
import logging
import os
import torch
from plotnine import *
import matplotlib.pyplot
matplotlib.pyplot.set_loglevel("error")
import DartsHelpers
import math
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
import json
from pipeline.NormalizationFilterManager import Double3_sigma_Filtering_And_Standardization
from pipeline.MetricManager import MetricManager
from PlottingHelpers import PlotHelper
from darts.models import TiDEModel, TCNModel, TransformerModel, RNNModel
from darts.models import LightGBMModel
from darts.models import LinearRegressionModel


# Longer lookback since this is required by the models


WANDB_DEBUG_MODE = False



def setup_model_TCN(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up TCN model with following params: " + str(params))

     # Setting up model
    model_name = "model"
    input_chunk_length = int(max_lookback_window / 7)
    forecast_horizon = int(nr_days_forecast / 7)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=params["early_stopping_patients"],
        verbose=True,
        mode='min'
    )


    # Setup up actual model
    model = TCNModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        batch_size=64,
        n_epochs=params["max_nr_epochs"],
        add_encoders=None,
        likelihood=None, 
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
        pl_trainer_kwargs={
            'precision': '32-true',
            "callbacks": [early_stopping_callback if "early_stopping_patients" in params else None],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Train model
    model.fit(training_full_ts["target_ts"], 
            past_covariates=training_full_ts["past_covariate_ts"],
            val_series=validation_full_ts["target_ts"], 
            val_past_covariates=validation_full_ts["past_covariate_ts"], 
            verbose=True)

    return model





def setup_model_transformer(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up Transformer model with following params: " + str(params))

     # Setting up model
    model_name = "model"
    input_chunk_length = int(max_lookback_window / 7)
    forecast_horizon = int(nr_days_forecast / 7)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=params["early_stopping_patients"],
        verbose=True,
        mode='min'
    )


    # Setup up actual model
    model = TransformerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        batch_size=64,
        n_epochs=params["max_nr_epochs"],
        add_encoders=None,
        likelihood=None, 
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
        pl_trainer_kwargs={
            'precision': '32-true',
            "callbacks": [early_stopping_callback if "early_stopping_patients" in params else None],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Train model
    model.fit(training_full_ts["target_ts"], 
            past_covariates=training_full_ts["past_covariate_ts"],
            val_series=validation_full_ts["target_ts"], 
            val_past_covariates=validation_full_ts["past_covariate_ts"], 
            verbose=True)

    return model






def setup_model_lstm(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up LSTM model with following params: " + str(params))

     # Setting up model
    model_name = "model"
    input_chunk_length = int(max_lookback_window / 7)
    forecast_horizon = int(nr_days_forecast / 7)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=params["early_stopping_patients"],
        verbose=True,
        mode='min'
    )


    # Setup up actual model
    model = RNNModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        model="LSTM",
        batch_size=64,
        n_epochs=params["max_nr_epochs"],
        add_encoders=None,
        likelihood=None, 
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
        pl_trainer_kwargs={
            'precision': '32-true',
            "callbacks": [early_stopping_callback if "early_stopping_patients" in params else None],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"], 
            val_series=validation_full_ts["target_ts"], 
            val_future_covariates=validation_full_ts["future_covariates_ts"],
            verbose=True)

    return model








def setup_model_rnn(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up RNN model with following params: " + str(params))

     # Setting up model
    model_name = "model"
    input_chunk_length = int(max_lookback_window / 7)
    forecast_horizon = int(nr_days_forecast / 7)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=params["early_stopping_patients"],
        verbose=True,
        mode='min'
    )


    # Setup up actual model
    model = RNNModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        model="RNN",
        batch_size=64,
        n_epochs=params["max_nr_epochs"],
        add_encoders=None,
        likelihood=None, 
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
        pl_trainer_kwargs={
            'precision': '32-true',
            "callbacks": [early_stopping_callback if "early_stopping_patients" in params else None],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"], 
            val_series=validation_full_ts["target_ts"], 
            val_future_covariates=validation_full_ts["future_covariates_ts"],
            verbose=True)

    return model









def main():


    ############################################################ Setup Experiment ############################################################


    NR_DAYS_FORECAST = 91   # We want to forecast up to the first visit after this value, or until the start of the next therapy (which ever comes first) - using 91 since it is the closest multiple of 7 to 90 days - often used for meds
    MAX_LOOKBACK_WINDOW = 98    # Max 35 days before LoT to use as input
    CONSTANT_ROW_COLUMNS_TO_USE = ["histology", "groupstage", "smokingstatus", "isadvanced", "sesindex2015_2019", "birthyear", "gender", "race", "ethnicity"]  # These are the columns that will be used as static covariates from the constant row
    MAX_NR_EPOCHS = 100


    eval_manager = EvaluationManager("2024_02_05_critical_vars")

    experiment = Experiment("setup")

    # Uncomment for debug mode of WandB
    if WANDB_DEBUG_MODE:
        experiment.setup_wandb_debug_mode()
    else:
        experiment.setup_wandb("All Darts Models Setup - Lookback: " + str(MAX_LOOKBACK_WINDOW), "Setup", project="UC2 - NSCLC- Critical Vars")

    #: Add hyperparameters to wandb
    wandb.config.nr_days_forecast = NR_DAYS_FORECAST
    wandb.config.max_lookback_window = MAX_LOOKBACK_WINDOW
    wandb.config.constant_row_columns_to_use = CONSTANT_ROW_COLUMNS_TO_USE
    wandb.config.max_nr_epochs = MAX_NR_EPOCHS



    ############################################################ Load & Split Data ############################################################

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
    
    path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_05_critical_vars/1_data/dataset_statistics.json"
    with open(path_to_statistics_file) as f:
        statistics_dic = json.load(f)
    
    
    ############################################################ Convert to Darts format ############################################################
        
    training_full_ts = DartsHelpers.convert_to_darts_dataset(training_full_events_split, training_full_meta_data,
                                                             statistics_dic=statistics_dic,
                                                                eval_manager=eval_manager,
                                                                forecast_horizon=NR_DAYS_FORECAST,
                                                                max_look_back_window=MAX_LOOKBACK_WINDOW, 
                                                                n_jobs=1,
                                                                constant_row_columns=CONSTANT_ROW_COLUMNS_TO_USE)

    
    validation_full_ts = DartsHelpers.convert_to_darts_dataset(validation_full_events, validation_full_meta, 
                                                               statistics_dic=statistics_dic,
                                                                eval_manager=eval_manager,
                                                                max_look_back_window=MAX_LOOKBACK_WINDOW, 
                                                                forecast_horizon=NR_DAYS_FORECAST,
                                                                save_target_dfs=True,
                                                                n_jobs=1,
                                                                constant_row_columns=CONSTANT_ROW_COLUMNS_TO_USE,
                                                                pipeline_targets=training_full_ts["pipeline_targets"], 
                                                                pipeline_past_covariates=training_full_ts["pipeline_past_covariates"], 
                                                                pipeline_static_covariates=training_full_ts["pipeline_static_covariates"], 
                                                                past_covariate_encoder=training_full_ts["past_covariate_encoder"])

    test_full_ts = DartsHelpers.convert_to_darts_dataset(test_full_events, test_full_meta, 
                                                         statistics_dic=statistics_dic,
                                                            max_look_back_window=MAX_LOOKBACK_WINDOW, 
                                                            eval_manager=eval_manager,
                                                            forecast_horizon=NR_DAYS_FORECAST,
                                                            save_target_dfs=True,
                                                            n_jobs=1,
                                                            constant_row_columns=CONSTANT_ROW_COLUMNS_TO_USE,
                                                            pipeline_targets=training_full_ts["pipeline_targets"], 
                                                            pipeline_past_covariates=training_full_ts["pipeline_past_covariates"], 
                                                            pipeline_static_covariates=training_full_ts["pipeline_static_covariates"], 
                                                            past_covariate_encoder=training_full_ts["past_covariate_encoder"])

    #: log missing data stats to wandb
    DartsHelpers.log_to_wandb_missing_data_statistics(training_full_ts, training_set)
    DartsHelpers.log_to_wandb_missing_data_statistics(validation_full_ts, validation_set)
    DartsHelpers.log_to_wandb_missing_data_statistics(test_full_ts, test_set)

    #: print for target ts missingness stats
    logging.info("Missingness ratio in target: " + str(training_full_ts["missing_data_statistics"]["target_ts_missing_ratio"]))
    logging.info("Nr of TS in training dataset: " + str(len(training_full_ts["target_ts"])))
    logging.info("Nr of TS in validation dataset: " + str(len(validation_full_ts["target_ts"])))
    logging.info("Nr of TS in test dataset: " + str(len(test_full_ts["target_ts"])))

    # Kill off old experiment and finish wandb
    wandb.run.finish()
    del experiment


    ############################################################ Model setup and training ############################################################


    def run_and_eval_model(model_setup_and_fit_func, model_wandb_name, model_wandb_group, curr_params):

        ############################################################ Experiment Setup ############################################################
        experiment = Experiment("baselines")

        # Uncomment for debug mode of WandB
        if WANDB_DEBUG_MODE:
            experiment.setup_wandb_debug_mode()
        else:
            model_wandb_group = model_wandb_group
            experiment.setup_wandb(model_wandb_name, model_wandb_group, project="UC2 - NSCLC- Critical Vars")
        
        input_chunk_length = int(MAX_LOOKBACK_WINDOW / 7)
        forecast_horizon = int(NR_DAYS_FORECAST / 7)


        ############################################################ Model setup and fitting ############################################################

        model = model_setup_and_fit_func(curr_params, MAX_LOOKBACK_WINDOW, NR_DAYS_FORECAST, experiment, training_full_ts, validation_full_ts)
        
        #: add hyperparameters to wandb
        wandb.config.update({"model_config": str(model)}, allow_val_change=True)

        
        ############################################################ Evaluate model ############################################################


        path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_05_critical_vars/1_data/dataset_statistics.json"
        with open(path_to_statistics_file) as f:
            statistics_dic = json.load(f)

            
        double_3_sigma_and_standardize = Double3_sigma_Filtering_And_Standardization(path_to_statistics_file)
        metric_manager = MetricManager(path_to_statistics_file)


        def evaluate_and_record(eval_dataset_dictionary, eval_set_name, eval_meta_data, forecast_horizon, model):

            # Get predictions
            eval_targets, eval_prediction = DartsHelpers.get_output_for_darts_torch_model(model, eval_manager, 
                                                                                            eval_dataset_dictionary=eval_dataset_dictionary,
                                                                                            forecast_horizon_chunks=forecast_horizon)
            
            #: change predictions >3 sigma to mean for stability
            logging.info("For stability, replace all >3 sigma predictions with mean")
            eval_prediction = DartsHelpers.turn_all_over_3_sigma_predictions_to_mean(eval_prediction, statistics_dic)

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



        validation_targets, validation_prediction, validation_meta_data = evaluate_and_record(eval_dataset_dictionary=validation_full_ts, 
                                                                                                eval_set_name=validation_set, 
                                                                                                eval_meta_data=validation_full_meta, 
                                                                                                forecast_horizon=forecast_horizon, 
                                                                                                model=model)
        
        test_targets, test_prediction, test_meta_data = evaluate_and_record(eval_dataset_dictionary=test_full_ts, 
                                                                            eval_set_name=test_set, 
                                                                            eval_meta_data=test_full_meta, 
                                                                            forecast_horizon=forecast_horizon, 
                                                                            model=model)


        ############################################################ Finish run ############################################################
        wandb.run.finish()
        del experiment




    ##################################################### Run actual models ###########################################################
    

    #: TCN - with 20 epoch patience
    curr_wandb_name = "TCN - CM - 20 epoch patience - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_TCN, curr_wandb_name, "TCN - Full", {
        "gradient_cliping": 1,
        "early_stopping_patients" : 20,
        "max_nr_epochs": 100,
    })
    
    #: TCN - with 10 epoch patience
    curr_wandb_name = "TCN - CM - 20 epoch patience - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_TCN, curr_wandb_name, "TCN - Full", {
        "gradient_cliping": 1,
        "early_stopping_patients" : 10,
        "max_nr_epochs": 100,
    })
    

    # For now only 1 model
    return

    #: TCN
    curr_wandb_name = "TCN - CM - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_TCN, curr_wandb_name, "TCN - Full", {
        "gradient_cliping": 1,
        "early_stopping_patients" : 3,
        "max_nr_epochs": 100,
    })

    #: Transformer
    curr_wandb_name = "Transformer - CM - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_transformer, curr_wandb_name, "Transformer - Full", {
        "gradient_cliping": 1,
        "early_stopping_patients" : 3,
        "max_nr_epochs": 100,
    })

    #: RNN
    curr_wandb_name = "RNN - CM - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_rnn, curr_wandb_name, "RNN - Full", {
        "gradient_cliping": 1,
        "early_stopping_patients" : 3,
        "max_nr_epochs": 100,
    })

    # : LSTM
    curr_wandb_name = "LSTM - CM - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_lstm, curr_wandb_name, "LSTM - Full", {
        "gradient_cliping": 1,
        "early_stopping_patients" : 3,
        "max_nr_epochs": 100,
    })


    # For now only running the new models!
    return




# Call main runner
if __name__ == "__main__":
    main()




 











