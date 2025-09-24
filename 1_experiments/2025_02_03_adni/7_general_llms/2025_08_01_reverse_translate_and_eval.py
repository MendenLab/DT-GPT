import __init__
import logging
from openai import (AsyncOpenAI, OpenAIError, AuthenticationError,
                    RateLimitError, APIConnectionError)
import asyncio
import pandas as pd
import os
import sys
import numpy as np
import traceback
from pandas.api.types import is_numeric_dtype
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
from pipeline.Experiment import Experiment
import wandb
import os
# To overcome issues with CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2000"
import __init__
from pipeline.EvaluationManager import EvaluationManager
from pipeline.Experiment import Experiment
import wandb
import pandas as pd
import numpy as np
from pipeline.Splitters import After1VisitSplitter
import logging
from pipeline.DataFrameConvertTemplateTextBasicDescription import DTGPTDataFrameConverterTemplateTextBasicDescription
from pipeline.DFConversionHelpers import process_all_tuples_multiprocessing, process_all_tuples
from pipeline.DataProcessorBiomistral import DataProcessorBiomistral
from pipeline.NormalizationFilterManager import Only_Double3_sigma_Filtering
from pipeline.NormalizationFilterManager import Only_Double3_sigma_Filtering
from pipeline.MetricManager import MetricManager
from PlottingHelpers import PlotHelper
from plotnine import *
from trl import SFTTrainer
from pipeline.NormalizationFilterManager import Double3_sigma_Filtering_And_Standardization
import json
from datetime import datetime
import argparse
import re



COLUMN_MAPPING_PATH = "/home/makaron1/dt-gpt/uc2_nsclc/2_experiments/2025_02_03_adni/1_data/1_final_data/column_descriptive_mapping.csv"
PATH_TO_STATISTICS_FILE = "/home/makaron1/dt-gpt/uc2_nsclc/2_experiments/2025_02_03_adni/1_data/1_final_data/dataset_statistics.json"
EVAL_MANAGER_SETTING = "2025_02_03_adni"
WANDB_PROJECT = "UC - ADNI"


def preprocess_test_set_data(llm_name, debug=False):
    
    test_set = "TEST"

    eval_manager = EvaluationManager(EVAL_MANAGER_SETTING)
    experiment = Experiment("dt_gpt_general_llm_eval")

    if debug:
        experiment.setup_wandb_debug_mode()
    else:
        experiment.setup_wandb(llm_name + " - reverse translation & eval", llm_name, project=WANDB_PROJECT)

    # Get patientids
    test_full_paths, test_full_patientids = eval_manager.get_paths_to_events_in_split(test_set)

    # Load data
    test_full_constants, test_full_events = eval_manager.load_list_of_patient_dfs_and_constants(test_full_patientids)

    # Setup splitter object
    splitter = After1VisitSplitter()
    
    # Setup splits
    test_events, test_meta = splitter.setup_split_indices(test_full_events, eval_manager)

    return test_events, test_meta, eval_manager, experiment




def convert_strings_to_df(test_events, test_meta, load_path, eval_manager):

    #: load in the data and convert it to a dataframe and concat
    raw_predicted_strings = pd.read_csv(load_path + "predicted_strings.csv")
    raw_meta_data = pd.read_csv(load_path + "prompts_with_meta.csv")
    predicted_data = pd.concat([raw_predicted_strings, raw_meta_data], axis=1)

    #: preprocess strings and clean them
    def clean_string(raw_s):
        # Slice only before the first "]}"
        if isinstance(raw_s, str):

            # Nothing special for now
            s = raw_s.lower().strip()
        else:
            s = str(raw_s)
        return s

    predicted_data['cleaned_prediction'] = predicted_data['response'].apply(clean_string)

    #: convert strings via JSON to dataframe
    def convert_to_df(row):
        """
        Parses a string of model predictions into a long-format DataFrame.
        
        This function is designed to work with pandas' groupby().apply(), where 'row'
        is a DataFrame group. It extracts day, variable, and value from each
        prediction line, handles malformed strings, and adds metadata.
        """
        try:
            # Get the prediction string from the first entry of the group
            raw_string = row['cleaned_prediction']

            # Regex to find all lines with the pattern: "day <day> - <variable> - <value>"
            # It robustly handles variable names with spaces, numbers, or symbols.
            pattern = re.compile(r"month\s+(\d+)\s+-\s+(.*?)\s+-\s+([\d.]+)")
            matches = pattern.findall(raw_string)
            
            # If no valid prediction lines are found, return NA
            if not matches:
                return pd.NA

            # Create a list of dictionaries from all found matches
            long_data = [
                {'month': int(day), 'variable': var.strip(), 'value': float(val)}
                for day, var, val in matches
            ]

            # Create the DataFrame from the list of records
            df = pd.DataFrame(long_data)

            # Add the metadata columns from the input group
            df["patientid"] = str(row["patientid"].iloc[0])
            df["patient_sample_index"] = "split_0"  # From bug of original code
            df["trajectory_index"] = row["trajectory_index"].iloc[0]

            # Sort by variable and day to ensure chronological order
            df = df.sort_values(by=['variable', 'month']).reset_index(drop=True)
            
            # Create prediction_index based on the position within each variable group
            df["prediction_index"] = df.groupby('variable').cumcount()

            # Replace spaces in variable names for edge case handling
            df["variable"] = df["variable"].str.replace(" ", "", regex=False)
            
            return df
        except Exception as e:
            # If any error occurs (e.g., missing keys, parsing issues), return NA
            return pd.NA
        
    predicted_data["prediction_as_df"] = predicted_data.apply(convert_to_df, axis=1)
    nr_not_converted = predicted_data["prediction_as_df"].isna().sum()
    logging.info(f"Number of not converted predictions: {nr_not_converted}")
    logging.info(f"Number of converted predictions: {len(predicted_data['prediction_as_df'].dropna())}")

    predictions = predicted_data["prediction_as_df"].dropna().tolist()
    predicted_df = pd.concat(predictions, ignore_index=True)

    #: aggregate them using mean on patientid, patient_sample_index across all trajectories
    predicted_df["value"] = pd.to_numeric(predicted_df["value"], errors='coerce')
    predicted_df = predicted_df.groupby(['patientid', 'patient_sample_index', "variable", "prediction_index"])[['value']].mean().reset_index()
    
    #: prep meta to be useful
    meta_df = pd.DataFrame(test_meta)
    meta_df["patientid"] = meta_df["patientid"].astype(str)
    meta_df["patient_sample_index"] = meta_df["patient_sample_index"].astype(str)
    target_events = pd.concat([x[3] for x in test_events])
    target_events["patientid"] = target_events["patientid"].astype(str)
    target_events["patient_sample_index"] = target_events["patient_sample_index"].astype(str)
    target_events_long = pd.melt(target_events, id_vars=['patientid', 'patient_sample_index', 'date'],
                                 var_name='variable', value_name='value')
    target_events_long = target_events_long.dropna(subset=['value'])
    target_events_long = target_events_long.drop(columns=['value'])

    #: map to the names, then remove spaces for easier matching
    column_mapping = pd.read_csv(COLUMN_MAPPING_PATH)
    column_mapping = column_mapping[["original_column_names", "descriptive_column_name"]]
    target_events_long = target_events_long.merge(column_mapping, how='left', 
                                                  left_on="variable", right_on="original_column_names")
    target_events_long["descriptive_column_name"] = target_events_long["descriptive_column_name"].str.replace(" ", "").str.lower()

    #: get order of the dates and add them as a column
    target_events_long["date"] = pd.to_datetime(target_events_long["date"], errors='coerce')
    target_events_long = target_events_long.sort_values(by=['patientid', 'patient_sample_index', 'date'])
    target_events_long['date_order'] = target_events_long.groupby(['patientid', 'patient_sample_index', 'variable']).cumcount()


    #: map target with predicted to add empty for any missing values & implicitly handle closest dates
    predicted_df = predicted_df.rename(columns={"variable": "descriptive_column_name"})
    full_prediction = target_events_long.merge(predicted_df, how="left",
                                               left_on=['patientid', 'patient_sample_index', 'descriptive_column_name', 'date_order'],
                                               right_on=['patientid', 'patient_sample_index', 'descriptive_column_name', 'prediction_index'])

    #: setup last observed values
    last_observed_list = meta_df[["patientid", "patient_sample_index", "last_input_values_of_targets"]].values.tolist()

    last_observed = []
    for patientid, patient_sample_index, last_values in last_observed_list:
        for i, value in enumerate(last_values):
            if not pd.isna(value[1]):
                last_observed.append({
                    "patientid": patientid,
                    "patient_sample_index": patient_sample_index,
                    "last_observed_value": value[1],
                    "variable": value[2],
                })
    last_observed_df = pd.DataFrame(last_observed)
    full_prediction["patientid"] = full_prediction["patientid"].astype(str)
    full_prediction["patient_sample_index"] = full_prediction["patient_sample_index"].astype(str)
    full_prediction["variable"] = full_prediction["variable"].astype(str)
    last_observed_df["patientid"] = last_observed_df["patientid"].astype(str)
    last_observed_df["patient_sample_index"] = last_observed_df["patient_sample_index"].astype(str)
    last_observed_df["variable"] = last_observed_df["variable"].astype(str)
    full_prediction = full_prediction.merge(last_observed_df, how="left",
                                           left_on=['patientid', 'patient_sample_index', 'variable'],
                                           right_on=['patientid', 'patient_sample_index', 'variable'])

    
    #: If the value is NaN for date_order_0, fill it with the last observed value
    date_order_0_mask = full_prediction['date_order'] == 0
    date_order_0_values =  full_prediction["value"].isna() & date_order_0_mask
    date_order_0 = full_prediction[date_order_0_values]
    full_prediction.loc[date_order_0_values, 'value'] = date_order_0['last_observed_value']

    #: Then for all other date_orders, forward fill NaN values with the last observed value for that patient, sample and variable
    full_prediction['value'] = full_prediction.groupby(['patientid', 'patient_sample_index', 'variable'])['value'].ffill()

    #: format into correct format (back into wide format)
    full_prediction_subset = full_prediction[["patientid", "patient_sample_index", "date", "variable", "value"]]
    full_prediction_wide = full_prediction_subset.pivot_table(index=['patientid', 'patient_sample_index', 'date'],
                                                              columns='variable', values='value',
                                                              aggfunc='first').reset_index()

    #: we need to include fully NA rows from target as well
    missing_entries = target_events[['patientid', 'patient_sample_index', 'date']].drop_duplicates()
    if missing_entries.shape[0] > full_prediction_wide.shape[0]:
        # Merge missing entries with the full prediction wide dataframe#
        missing_entries['patientid'] = missing_entries['patientid'].astype(str)
        missing_entries['patient_sample_index'] = missing_entries['patient_sample_index'].astype(str)
        full_prediction_wide = pd.merge(missing_entries, full_prediction_wide, on=['patientid', 'patient_sample_index', 'date'], how='left')
        assert full_prediction_wide.shape[0] == missing_entries.shape[0], "Mismatch in number of rows after merging missing entries."
    
    #: check for missing, rare columns and add them in
    all_columns = test_events[0][3].columns.tolist()
    missing_columns = set(all_columns) - set(full_prediction_wide.columns.tolist())
    if missing_columns:
        logging.warning(f"Missing columns in the prediction: {missing_columns}")
        # Add missing columns with NaN values
        for col in missing_columns:
            full_prediction_wide[col] = pd.NA

    #: process via eval manager and then get concatenated dfs
    # This ensures everything is in correct format
    eval_manager.evaluate_split_stream_start()
    for constant_df, input_df, empty_df, target_df in test_events:
        #: get patientid and patient_sample_index
        patientid = str(input_df['patientid'].iloc[0])
        patient_sample_index = str(input_df['patient_sample_index'].iloc[0])
        curr_prediction = full_prediction_wide[(full_prediction_wide['patientid'] == patientid) & 
                                                (full_prediction_wide['patient_sample_index'] == patient_sample_index)]
        
        eval_manager.evaluate_split_stream_prediction(curr_prediction, target_df, patientid, patient_sample_index)
    full_df_targets, full_df_prediction = eval_manager.concat_eval()

    # Convert target columns to numeric
    original_columns = column_mapping["original_column_names"].tolist()
    subset_cols = [col for col in full_df_targets.columns if col in original_columns]
    for col in subset_cols:
        full_df_targets[col] = pd.to_numeric(full_df_targets[col], errors='coerce')
        full_df_prediction[col] = pd.to_numeric(full_df_prediction[col], errors='coerce')

    # Return
    return full_df_targets, full_df_prediction



def evaluate(eval_meta_data, eval_targets, eval_prediction, experiment):

    #: reuse previous code (including filtering of values)

    eval_set_name = "TEST"
    double_3_sigma_and_standardize = Double3_sigma_Filtering_And_Standardization(PATH_TO_STATISTICS_FILE)
    metric_manager = MetricManager(PATH_TO_STATISTICS_FILE)

    # Do filtering without standardizing
    eval_targets_filtered, eval_prediction_filtered = double_3_sigma_and_standardize.normalize_and_filter(eval_targets, eval_prediction)

    #: set grouping by therapy
    eval_targets_filtered_with_meta_data = experiment.join_meta_data_to_targets(eval_targets_filtered, eval_meta_data)

    # Calculate performance metrics
    eval_performance = metric_manager.calculate_metrics(eval_targets_filtered, eval_prediction_filtered)

    # Save tables locally and record in wandb
    experiment.save_df_targets_predictions_locally_and_statistics_to_wandb(eval_set_name, eval_targets_filtered, eval_prediction_filtered, meta_data_df=eval_targets_filtered_with_meta_data)

    # Save performance to wandb
    experiment.save_to_wandb_final_performances(eval_performance, eval_set_name)

    # Print for debug
    print(eval_performance[list(eval_performance.keys())[-1]]["mae"]["overall"])




def run(debug, curr_model, load_path):

    # Prepare all strings
    llm_name = curr_model.split("/")[-1]
    test_events, test_meta, eval_manager, experiment = preprocess_test_set_data(llm_name, debug=debug)

    # Load in the strings
    full_df_targets, full_df_prediction = convert_strings_to_df(test_events, test_meta, load_path, eval_manager)

    # Evaluate the strings
    evaluate(test_meta, full_df_targets, full_df_prediction, experiment)

    # Wrap up
    print("Evaluation completed successfully.")
    wandb.finish()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the reverse translation and evaluation script.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--load_path", type=str, default="/home/makaron1/dt-gpt/uc2_nsclc/2_experiments/2025_02_03_adni/7_general_llms/results/2025_07_31_09_24_52/", help="Path to load the predicted strings and meta data from.")
    parser.add_argument("--curr_model", type=str, default="/home/makaron1/dt-gpt/cache/hub/models--Qwen--Qwen3-32B", help="Name of the current model to evaluate.")
    args = parser.parse_args()

    run(args.debug, args.curr_model, args.load_path)



