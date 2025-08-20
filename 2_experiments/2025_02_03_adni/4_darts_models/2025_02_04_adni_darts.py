import __init__
from EvaluationManager import EvaluationManager
from Experiment import Experiment
import wandb
import pandas as pd
import numpy as np
from Splitters import After1VisitSplitter
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
from NormalizationFilterManager import Only_Standardization
from MetricManager import MetricManager
from PlottingHelpers import PlotHelper
from darts.models import TiDEModel
from darts.models import LightGBMModel
from darts.models import LinearRegressionModel
from darts.models import TiDEModel, TransformerModel, RNNModel


WANDB_DEBUG_MODE = False



def setup_model_transformer(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up Transformer model with following params: " + str(params))

     # Setting up model
    model_name = "model"
    input_chunk_length = int(max_lookback_window)
    forecast_horizon = int(nr_days_forecast)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    if "early_stopping_patients" in params:
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
            "callbacks": [early_stopping_callback] if "early_stopping_patients" in params else [],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Train model
    model.fit(training_full_ts["target_ts"], 
            past_covariates=training_full_ts["past_covariate_ts"], 
            verbose=True)

    return model






def setup_model_lstm(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up LSTM model with following params: " + str(params))

     # Setting up model
    model_name = "model"
    input_chunk_length = int(max_lookback_window)
    forecast_horizon = int(nr_days_forecast)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    if "early_stopping_patients" in params:
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
        training_length=input_chunk_length + forecast_horizon - 1,
        model="LSTM",
        batch_size=64,
        n_epochs=params["max_nr_epochs"],
        add_encoders=None,
        likelihood=None, 
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
        pl_trainer_kwargs={
            'precision': '32-true',
            "callbacks": [early_stopping_callback] if "early_stopping_patients" in params else [],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"],
            verbose=True)

    return model








def setup_model_rnn(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up RNN model with following params: " + str(params))

     # Setting up model
    model_name = "model"
    input_chunk_length = int(max_lookback_window)
    forecast_horizon = int(nr_days_forecast)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    if "early_stopping_patients" in params:
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
        training_length=input_chunk_length + forecast_horizon - 1,
        model="RNN",
        batch_size=64,
        n_epochs=params["max_nr_epochs"],
        add_encoders=None,
        likelihood=None, 
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
        pl_trainer_kwargs={
            'precision': '32-true',
            "callbacks": [early_stopping_callback] if "early_stopping_patients" in params else [],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"],
            verbose=True)

    return model






def setup_model_tide(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up TiDE model with following params: " + str(params))

     # Setting up TFT model
    model_name = "model"
    input_chunk_length = int(max_lookback_window)
    forecast_horizon = int(nr_days_forecast)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    if "early_stopping_patients" in params:
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00001,
            patience=params["early_stopping_patients"],
            verbose=True,
            mode='min'
        )


    # Setup up actual model
    model = TiDEModel(
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
            "callbacks": [early_stopping_callback] if "early_stopping_patients" in params else [],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Add saving of losses to wandb
    #wandb.tensorboard.patch(root_logdir=f'{model_name}/logs/')


    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"], 
            past_covariates=training_full_ts["past_covariate_ts"],
            verbose=True)

    return model






def setup_model_tft(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up TFT model with following params: " + str(params))

     # Setting up TFT model
    model_name = "model"
    input_chunk_length = int(max_lookback_window)
    forecast_horizon = int(nr_days_forecast)
    GRADIENT_CLIPPING = params["gradient_cliping"]


    #: set up early stopping
    if "early_stopping_patients" in params:
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00001,
            patience=params["early_stopping_patients"],
            verbose=True,
            mode='min'
        )

    # Setup up actual model
    model = TFTModel(
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
            "callbacks": [early_stopping_callback] if "early_stopping_patients" in params else [],
            "gradient_clip_val": GRADIENT_CLIPPING,
        },
        work_dir=experiment.get_experiment_folder(),
        log_tensorboard=True,
        model_name=model_name,
    )

    # Add saving of losses to wandb
    #wandb.tensorboard.patch(root_logdir=f'{model_name}/logs/')


    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"], 
            past_covariates=training_full_ts["past_covariate_ts"],
            verbose=True)

    return model





def setup_model_lightgbm(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):
    
    logging.info("Setting up LightGBM model with following params: " + str(params))
    
    # Setting up TFT model
    model_name = "model"
    input_chunk_length = int(max_lookback_window)
    forecast_horizon = int(nr_days_forecast)
    
    # Setup up actual model
    model = LightGBMModel(
        output_chunk_length=forecast_horizon,
        lags=input_chunk_length,
        lags_past_covariates=input_chunk_length,
        lags_future_covariates=(input_chunk_length, forecast_horizon),
    )

    # Add saving of losses to wandb
    #wandb.tensorboard.patch(root_logdir=f'{model_name}/logs/')

    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"], 
            past_covariates=training_full_ts["past_covariate_ts"],)
    
    return model





def setup_model_linreg(params, max_lookback_window, nr_days_forecast, experiment, training_full_ts, validation_full_ts):

    logging.info("Setting up Linear Regression model with following params: " + str(params))

    # Setting up TFT model
    model_name = "model"
    input_chunk_length = int(max_lookback_window )
    forecast_horizon = int(nr_days_forecast )
    
    # Setup up actual model
    model = LinearRegressionModel(
        output_chunk_length=forecast_horizon,
        lags=input_chunk_length,
        lags_past_covariates=input_chunk_length,
        lags_future_covariates=(input_chunk_length, forecast_horizon),
    )


    # Add saving of losses to wandb
    #wandb.tensorboard.patch(root_logdir=f'{model_name}/logs/')

    # Train model
    model.fit(training_full_ts["target_ts"], 
            future_covariates=training_full_ts["future_covariates_ts"], 
            past_covariates=training_full_ts["past_covariate_ts"])
    
    return model
    








def main():


    ############################################################ Setup Experiment ############################################################


    NR_DAYS_FORECAST = 4   
    MAX_LOOKBACK_WINDOW = 1
    CONSTANT_ROW_COLUMNS_TO_USE =[
        "DX_bl_EMCI",
        "PTMARRY_Never married",
        "PTGENDER_Male",
        "PTMARRY_Widowed",
        "AGE",
        "DX_bl_AD",
        "PTMARRY_Unknown",
        "dataset_split",
        "APOE4",
        "PTMARRY_Married",
        "DX_bl_LMCI",
        "PTGENDER_Female",
        "PTMARRY_Divorced",
        "DX_bl_CN"
    ]
    MAX_NR_EPOCHS = 100


    eval_manager = EvaluationManager("2025_02_03_adni")

    experiment = Experiment("setup")

    # Uncomment for debug mode of WandB
    if WANDB_DEBUG_MODE:
        experiment.setup_wandb_debug_mode()
    else:
        experiment.setup_wandb("All Darts Models Setup - Full - Lookback: " + str(MAX_LOOKBACK_WINDOW), "Setup", project="UC - ADNI")

    #: Add hyperparameters to wandb
    wandb.config.nr_days_forecast = NR_DAYS_FORECAST
    wandb.config.max_lookback_window = MAX_LOOKBACK_WINDOW
    wandb.config.constant_row_columns_to_use = CONSTANT_ROW_COLUMNS_TO_USE
    wandb.config.max_nr_epochs = MAX_NR_EPOCHS



    ############################################################ Load & Split Data ############################################################

    # Get paths patientids to datasets

    logging.info("======================================================= RUNNING FULL ===================================================================")

    training_set = "TRAIN"
    test_set = "TEST"

    # Get patientids
    training_full_paths, training_full_patientids = eval_manager.get_paths_to_events_in_split(training_set)
    test_full_paths, test_full_patientids = eval_manager.get_paths_to_events_in_split(test_set)


    # Load data
    training_full_constants, training_full_events = eval_manager.load_list_of_patient_dfs_and_constants(training_full_patientids)
    test_full_constants, test_full_events = eval_manager.load_list_of_patient_dfs_and_constants(test_full_patientids)

    # Setup splitter object
    splitter = After1VisitSplitter()
    
    # Setup splits
    training_full_events_split, training_full_meta_data = splitter.setup_split_indices(training_full_events, eval_manager)
    test_full_events, test_full_meta = splitter.setup_split_indices(test_full_events, eval_manager)
    
    path_to_statistics_file = experiment.base_path + "2_experiments/2025_02_03_adni/1_data/1_final_data/dataset_statistics.json"
    with open(path_to_statistics_file) as f:
        statistics_dic = json.load(f)
    
    
    ############################################################ Convert to Darts format ############################################################
        
    training_full_ts = DartsHelpers.convert_to_darts_dataset_ADNI(training_full_events_split, training_full_meta_data,
                                                                    statistics_dic=statistics_dic,
                                                                        eval_manager=eval_manager,
                                                                        forecast_horizon=NR_DAYS_FORECAST,
                                                                        max_look_back_window=365, 
                                                                        n_jobs=1,
                                                                        constant_row_columns=CONSTANT_ROW_COLUMNS_TO_USE,
                                                                        freq="6MS")

    test_full_ts = DartsHelpers.convert_to_darts_dataset_ADNI(test_full_events, test_full_meta, 
                                                                statistics_dic=statistics_dic,
                                                                    max_look_back_window=365, 
                                                                    eval_manager=eval_manager,
                                                                    forecast_horizon=NR_DAYS_FORECAST,
                                                                    save_target_dfs=True,
                                                                    n_jobs=1,
                                                                    constant_row_columns=CONSTANT_ROW_COLUMNS_TO_USE,
                                                                    pipeline_targets=training_full_ts["pipeline_targets"], 
                                                                    pipeline_past_covariates=training_full_ts["pipeline_past_covariates"], 
                                                                    pipeline_static_covariates=training_full_ts["pipeline_static_covariates"], 
                                                                    past_covariate_encoder=training_full_ts["past_covariate_encoder"],
                                                                    freq="6MS")

    #: log missing data stats to wandb
    DartsHelpers.log_to_wandb_missing_data_statistics(training_full_ts, training_set)
    DartsHelpers.log_to_wandb_missing_data_statistics(test_full_ts, test_set)

    #: print for target ts missingness stats
    logging.info("Missingness ratio in target: " + str(training_full_ts["missing_data_statistics"]["target_ts_missing_ratio"]))
    logging.info("Nr of TS in training dataset: " + str(len(training_full_ts["target_ts"])))
    logging.info("Nr of TS in test dataset: " + str(len(test_full_ts["target_ts"])))

    # Kill off old experiment and finish wandb
    wandb.run.finish()
    del experiment


    ############################################################ Model setup and training ############################################################


    def run_and_eval_model(model_setup_and_fit_func, model_wandb_name, model_wandb_group, curr_params):

        ############################################################ Experiment Setup ############################################################
        experiment = Experiment("adni_darts")

        # Uncomment for debug mode of WandB
        if WANDB_DEBUG_MODE:
            experiment.setup_wandb_debug_mode()
        else:
            experiment.setup_wandb(model_wandb_name, model_wandb_group, project="UC - ADNI")


        logging.info("======================================================= RUNNING Full ===================================================================")

        
        input_chunk_length = int(MAX_LOOKBACK_WINDOW)
        forecast_horizon = int(NR_DAYS_FORECAST)
        inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
        target_cols_raw = target_cols.copy()


        ############################################################ Model setup and fitting ############################################################

        model = model_setup_and_fit_func(curr_params, MAX_LOOKBACK_WINDOW, NR_DAYS_FORECAST, experiment, training_full_ts, None)
        
        #: add hyperparameters to wandb
        wandb.config.update({"model_config": str(model)}, allow_val_change=True)

        
        ############################################################ Evaluate model ############################################################


        path_to_statistics_file = experiment.base_path + "2_experiments/2025_02_03_adni/1_data/1_final_data/dataset_statistics.json"
        with open(path_to_statistics_file) as f:
            statistics_dic = json.load(f)

            
        only_standardize = Only_Standardization(path_to_statistics_file)
        metric_manager = MetricManager(path_to_statistics_file)


        def evaluate_and_record(eval_dataset_dictionary, eval_set_name, eval_meta_data, forecast_horizon, model):

            # Get predictions
            eval_targets, eval_prediction = DartsHelpers.get_output_for_darts_torch_model(model, eval_manager, 
                                                                                            eval_dataset_dictionary=eval_dataset_dictionary,
                                                                                            forecast_horizon_chunks=forecast_horizon)
            
            # Apply numeric
            eval_prediction[target_cols_raw] = eval_prediction[target_cols_raw].apply(pd.to_numeric, errors='raise')
            eval_targets[target_cols_raw] = eval_targets[target_cols_raw].apply(pd.to_numeric, errors='raise')
            
            #: change predictions >3 sigma to mean for stability
            logging.info("For stability, replace all >3 sigma predictions with mean")
            eval_prediction = DartsHelpers.turn_all_over_3_sigma_predictions_to_mean(eval_prediction, statistics_dic)

            # Do filtering without standardizing
            eval_targets_filtered, eval_prediction_filtered = only_standardize.normalize_and_filter(eval_targets, eval_prediction)
            
            #: set grouping by therapy
            eval_targets_filtered_with_meta_data = experiment.join_meta_data_to_targets(eval_targets_filtered, eval_meta_data)

            # Calculate performance metrics
            eval_performance = metric_manager.calculate_metrics(eval_targets_filtered, eval_prediction_filtered)

            # Save tables locally and record in wandb
            experiment.save_df_targets_predictions_locally_and_statistics_to_wandb(eval_set_name, eval_targets_filtered, eval_prediction_filtered, meta_data_df=eval_targets_filtered_with_meta_data)

            # Save performance to wandb
            experiment.save_to_wandb_final_performances(eval_performance, eval_set_name)

            return eval_targets_filtered, eval_prediction_filtered, eval_targets_filtered_with_meta_data
        
        test_targets, test_prediction, test_meta_data = evaluate_and_record(eval_dataset_dictionary=test_full_ts, 
                                                                            eval_set_name=test_set, 
                                                                            eval_meta_data=test_full_meta, 
                                                                            forecast_horizon=forecast_horizon, 
                                                                            model=model)


        ############################################################ Finish run ############################################################
        wandb.run.finish()
        del experiment




    ##################################################### Run actual models ###########################################################
    
    
    #: RNN
    curr_wandb_name = "RNN - CM - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_rnn, curr_wandb_name, "RNN - Full", {
        "gradient_cliping": 1,
        "max_nr_epochs": 100,
    })

    # : LSTM
    curr_wandb_name = "LSTM - CM - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_lstm, curr_wandb_name, "LSTM - Full", {
        "gradient_cliping": 1,
        "max_nr_epochs": 100,
    })
    
    # TiDE default
    curr_wandb_name = "TiDE - CM - Full Validation & Training -"
    run_and_eval_model(setup_model_tide, curr_wandb_name, "TiDE - Full", {
        "gradient_cliping": 100,
        "max_nr_epochs": 100,
    })

    # TiDE long
    curr_wandb_name = "TiDE - CM - Full Validation & Training - 20 Epochs Patience -"
    run_and_eval_model(setup_model_tide, curr_wandb_name, "TiDE - Full", {
        "gradient_cliping": 100,
        "max_nr_epochs": 100,
    })


    # LightGBM
    curr_wandb_name = "LightGBM - CM - Full Validation & Training -"
    run_and_eval_model(setup_model_lightgbm, curr_wandb_name, "LightGBM - Full", {})
    
    # LinReg
    curr_wandb_name = "LinReg - CM - Full Validation & Training -"
    run_and_eval_model(setup_model_linreg, curr_wandb_name, "Linear Regression - Full", {})
     


    # TFT default
    curr_wandb_name = "TFT - CM - Full Validation & Training -"
    run_and_eval_model(setup_model_tft, curr_wandb_name, "TFT - Full", {
        "gradient_cliping": 100,
        "max_nr_epochs": 100,
    })

    # TFT long
    curr_wandb_name = "TFT - CM - Full Validation & Training - 20 Epochs Patience -"
    run_and_eval_model(setup_model_tft, curr_wandb_name, "TFT - Full", {
        "gradient_cliping": 100,
        "max_nr_epochs": 100,
    })


    #: Transformer
    curr_wandb_name = "Transformer - CM - Full Validation & Training - Forecast: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    run_and_eval_model(setup_model_transformer, curr_wandb_name, "Transformer - Full", {
        "gradient_cliping": 1,
        "max_nr_epochs": 100,
    })

    




# Call main runner
if __name__ == "__main__":
    main()




 











