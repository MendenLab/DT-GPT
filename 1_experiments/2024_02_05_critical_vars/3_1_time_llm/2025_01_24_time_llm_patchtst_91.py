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
import pipeline.DartsHelpers as DartsHelpers
import NeuralForecastHelpers
import math
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
import json
from pipeline.NormalizationFilterManager import Double3_sigma_Filtering_And_Standardization
from pipeline.MetricManager import MetricManager
from PlottingHelpers import PlotHelper
from darts.models import TiDEModel
from darts.models import LightGBMModel
from darts.models import LinearRegressionModel
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM, PatchTST, MLP


#: setup correctly
WANDB_DEBUG_MODE = False

# Set various consts
GPT_2 = "openai-community/gpt2"
LLAMA2_7B = "meta-llama/Llama-2-7b-hf"





def setup_model_mlp(params, max_lookback_window, nr_days_forecast, train_dataset, val_dataset):
    """
    Sets up and trains an MLP model for time series forecasting using NeuralForecast.

    Args:
        params (dict): Dictionary containing model parameters.
            Expected keys:
                - "model": Name or configuration of the MLP model.
                - "max_training_steps": Maximum number of training steps.
                - Optional keys for MLP architecture, e.g., "layers", "activation", "dropout".
                - "prompt_prefix": (If applicable) Prefix for any prompts used.
        max_lookback_window (int): The maximum number of past time steps to consider.
        nr_days_forecast (int): The number of days to forecast.
        train_dataset (dict): Training dataset containing:
            - "df": Training dataframe.
            - "static_df": Static features dataframe.
            - "future_covariate_cols": List of future covariate column names.
            - "past_covariate_cols": List of past covariate column names.
            - "static_covariate_cols": List of static covariate column names.
            - "freq": Frequency of the time series data.
        val_dataset (dict): Validation dataset (not used in this setup but can be utilized for validation).

    Returns:
        MLP: Trained MLP model instance.
    """

    logging.info("Setting up MLP model with the following params: %s", params)

    # Extract parameters with defaults for optional MLP configurations


    max_training_steps = params.get("max_training_steps", 1000)
    
    # Initialize the MLP model
    mlp_model = MLP(
        h=nr_days_forecast // 7,
        input_size=max_lookback_window // 7,
        num_layers=2,
        hidden_size=64,
        batch_size=16,               # Smaller batch size for CPU efficiency
        valid_batch_size=16,
        max_steps=max_training_steps,
        futr_exog_list=train_dataset["future_covariate_cols"],
        hist_exog_list=train_dataset["past_covariate_cols"],
        stat_exog_list=train_dataset["static_covariate_cols"],
        scaler_type='robust',        # Robust scaler as in TimeLLM setup
        random_seed=7862,  # For reproducibility
    )

    # Initialize NeuralForecast with the MLP model
    nf = NeuralForecast(
        models=[mlp_model],
        freq=train_dataset.get("freq", "D")  # Default to daily frequency if not specified
    )

    logging.info("Starting to fit the MLP model...")

    # Fit the model on the training data
    nf.fit(
        df=train_dataset["df"],
        static_df=train_dataset.get("static_df"),
        val_size=0,
    )

    logging.info("Finished fitting the MLP model.")

    return nf




def setup_model_time_llm(params, max_lookback_window, nr_days_forecast, train_dataset, val_dataset):

    logging.info("Setting up time-llm model with following params: " + str(params))

    prompt_prefix = params["prompt_prefix"]
    model = params["model"]
    max_training_steps = params["max_training_steps"]

    #: setup model
    timellm = TimeLLM(h=nr_days_forecast // 7,
                input_size=max_lookback_window // 7,
                llm=model,
                patch_len=nr_days_forecast // 7,  # Since of patching lengths
                top_k=params["top_k"],        # 5 gets error since out of bounds
                d_llm = 4096 if model != GPT_2 else 768,
                n_heads=32  if model != GPT_2 else 12,
                prompt_prefix=prompt_prefix,
                batch_size=params["batch_size"],
                valid_batch_size=params["batch_size"],
                windows_batch_size=params["windows_batch_size"],
                max_steps=max_training_steps,
                scaler_type = 'robust',
                random_seed=7862,
                )

    nf = NeuralForecast(
        models=[timellm],
        freq=train_dataset["freq"],
    )

    #: TimeLLM does not support future, historical or static exogenous variables.
    train_dataset_df = train_dataset["df"][["unique_id", "ds", "y"]]

    logging.info("Starting to fit model")

    #: fit model
    nf.fit(df=train_dataset_df,
           val_size = 0)

    logging.info("Finished fitting model")
    
    return nf



def setup_model_patchtst(params, max_lookback_window, nr_days_forecast, train_dataset, val_dataset):

    logging.info("Setting up PatchTST model with following params: " + str(params))

    max_training_steps = params["max_training_steps"]

    #: setup model
    patchtst = PatchTST(h=nr_days_forecast // 7,
                input_size=max_lookback_window // 7,
                batch_size=params["batch_size"],
                valid_batch_size=params["batch_size"],
                windows_batch_size=params["windows_batch_size"],
                max_steps=max_training_steps,
                scaler_type = 'robust',
                random_seed=7862,
    )

    nf = NeuralForecast(
        models=[patchtst],
        freq=train_dataset["freq"],
    )

    #: TimeLLM does not support future, historical or static exogenous variables.
    train_dataset_df = train_dataset["df"][["unique_id", "ds", "y"]]

    #: fit model
    logging.info("Starting to fit model")

    nf.fit(df=train_dataset_df,
           val_size = 0)

    logging.info("Finished fitting model")
    
    return nf






def main():


    ############################################################ Setup Experiment ############################################################


    NR_DAYS_FORECAST = 91   # We want to forecast up to the first visit after this value, or until the start of the next therapy (which ever comes first) - using 91 since it is the closest multiple of 7 to 90 days - often used for meds
    MAX_LOOKBACK_WINDOW = 91    # Max 91 days before LoT to use as input
    CONSTANT_ROW_COLUMNS_TO_USE = ["histology", "groupstage", "smokingstatus", "isadvanced", "sesindex2015_2019", "birthyear", "gender", "race", "ethnicity"]  # These are the columns that will be used as static covariates from the constant row
    MAX_NR_EPOCHS = 100


    eval_manager = EvaluationManager("2024_02_05_critical_vars")

    experiment = Experiment("setup_neuralforecast")

    # Uncomment for debug mode of WandB
    if WANDB_DEBUG_MODE:
        experiment.setup_wandb_debug_mode()
    else:
        experiment.setup_wandb("All NeuralForecast Models Setup - Lookback: " + str(MAX_LOOKBACK_WINDOW), "Setup", project="UC2 - NSCLC- Critical Vars")

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

    
    ############################################################ Convert to NeuralForecast format ############################################################


    logging.info("Converting to NeuralForecast format")
    
    # Taken from NeuralForecastHelpers
    training_full_nf = NeuralForecastHelpers.convert_to_neuralforecast_dataset(training_full_ts)
    validation_full_nf = NeuralForecastHelpers.convert_to_neuralforecast_dataset(validation_full_ts)
    test_full_nf = NeuralForecastHelpers.convert_to_neuralforecast_dataset(test_full_ts)

    logging.info("Finished converting to NeuralForecast format")

    
    # Kill off old experiment and finish wandb
    wandb.run.finish()
    del experiment

    ############################################################ Model setup and training ############################################################


    def run_and_eval_model(model_setup_and_fit_func, model_wandb_name, model_wandb_group, curr_params):

        ############################################################ Experiment Setup ############################################################
        experiment = Experiment("setup_neuralforecast")

        # Uncomment for debug mode of WandB
        if WANDB_DEBUG_MODE:
            experiment.setup_wandb_debug_mode()
        else:
            model_wandb_group = model_wandb_group
            experiment.setup_wandb(model_wandb_name, model_wandb_group, project="UC2 - NSCLC- Critical Vars")
        
        input_chunk_length = int(MAX_LOOKBACK_WINDOW / 7)
        forecast_horizon = int(NR_DAYS_FORECAST / 7)


        ############################################################ Model setup and fitting ############################################################

        model = model_setup_and_fit_func(curr_params, MAX_LOOKBACK_WINDOW, NR_DAYS_FORECAST, training_full_nf, validation_full_nf)
        
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
            eval_targets, eval_prediction = NeuralForecastHelpers.get_output_for_neuralforecast_model(model, eval_manager, 
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



        validation_targets, validation_prediction, validation_meta_data = evaluate_and_record(eval_dataset_dictionary=validation_full_nf, 
                                                                                                eval_set_name=validation_set, 
                                                                                                eval_meta_data=validation_full_meta, 
                                                                                                forecast_horizon=forecast_horizon, 
                                                                                                model=model)
        
        test_targets, test_prediction, test_meta_data = evaluate_and_record(eval_dataset_dictionary=test_full_nf, 
                                                                            eval_set_name=test_set, 
                                                                            eval_meta_data=test_full_meta, 
                                                                            forecast_horizon=forecast_horizon, 
                                                                            model=model)

        # No plotting
            


        ############################################################ Finish run ############################################################
        wandb.run.finish()
        del experiment



    ##################################################### Run actual models ###########################################################
    

    # MLP for debugging
    curr_wandb_name = "MLP: " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
    params_mlp = {
        "max_training_steps": 50,
    }
    #run_and_eval_model(setup_model_mlp, curr_wandb_name, "MLP", params_mlp)
    #return


    # PatchTST
    def run_patchtst(num_epochs):

        curr_wandb_name = "PatchTST: epochs:" + str(num_epochs) + " " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
        params = {
            "max_training_steps": round(num_epochs * 2856),  # Since we can't set up nr epochs in NeuralForecast
            "batch_size" : 32,  # Number of instances to get
            "windows_batch_size": 1024,  # Number of batches per sequence to get at once
        }
        run_and_eval_model(setup_model_patchtst, curr_wandb_name, "PatchTST", params)
    
    # Using 100 epochs from paper
    run_patchtst(100)
    
    
    logging.info("Finished PatchTST grid search")


    # Time-LLM
    def run_time_llm(num_epochs):

        is_gpu_over_60GB = False
        if torch.cuda.is_available():
            gpu_size = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            is_gpu_over_60GB = gpu_size > 60


        curr_wandb_name = "Time-LLM: epochs:" + str(num_epochs) + " " + str(NR_DAYS_FORECAST) + " Lookback: " + str(MAX_LOOKBACK_WINDOW)
        prompt = "The Flatiron Health non-small cell lung cancer dataset includes patient time series for lactate dehydrogenase, leukocytes, lymphocytes, lymphocyte/leukocyte ratio, neutrophils, and hemoglobin. \nDomain: Trends are primarily influenced by medication and patient condition, with significant changes occurring early and stabilizing over time."
        window_batch_size = 1   # We tune primarily via batch_size, since we have many relatively short sequences
        batch_size = 32   # (2048 works on 40GB GPU) - Using 32 as in paper
        params = {
            "prompt_prefix": prompt,
            "max_training_steps": round(num_epochs * 2856),  # Providing since we can't just give nr of epochs in NeuralForecast
            "model": LLAMA2_7B,  # GPT_2 or LLAMA2_7B
            "top_k": 5,  # From paper
            "batch_size" : batch_size,  # Number of instances to get
            "windows_batch_size": window_batch_size,  # Number of batches per sequence to get at once
        }
        run_and_eval_model(setup_model_time_llm, curr_wandb_name, "Time-LLM", params)


    # Running for 1 epoch and 50 (as in paper)
    run_time_llm(1)
    run_time_llm(50)



# Call main runner
if __name__ == "__main__":
    main()



