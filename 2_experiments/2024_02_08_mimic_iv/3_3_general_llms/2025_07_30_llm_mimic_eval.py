import os
# To overcome issues with CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2000"
import __init__
import logging
from openai import (AsyncOpenAI, OpenAIError, AuthenticationError,
                    RateLimitError, APIConnectionError)
import asyncio
import pandas as pd
import traceback
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
from Experiment import Experiment
import wandb
from EvaluationManager import EvaluationManager
from Splitters import After24HSplitter
import logging
from DataFrameConvertTDBDMIMIC import DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC
from DFConversionHelpers import process_all_tuples_multiprocessing, process_all_tuples
from NormalizationFilterManager import Only_Double3_sigma_Filtering
import json
from datetime import datetime
import argparse


MAX_TOKENS = 1400



async def _call_vllm(client, model_to_use, raw_instruction, max_tokens, seed, semaphore, temperature, 
                     top_p, patientid, patient_sample_index, trajectory_index, idx, max_idx, 
                     min_p, top_k):


    extra_body_params = {
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
    }

    if min_p is not None:
        extra_body_params["min_p"] = min_p
    if top_k is not None:
        extra_body_params["top_k"] = top_k

    model_type = model_to_use.split("/")[0]

    if model_type == "BioMistral":
        messages = "[INST]" + raw_instruction + "[/INST]"
    else:
        messages = [
            {"role": "user", "content": raw_instruction}
        ]
        extra_body_params["chat_template_kwargs"] = {"enable_thinking": False}

    

    async with semaphore:
        try:
            if model_type == "BioMistral":
                completion = await client.completions.create(
                    model=model_to_use,
                    prompt=messages,
                    extra_body=extra_body_params
                )
                response = completion.choices[0].text
            else:
                completion = await client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    extra_body=extra_body_params
                )
                response = completion.choices[0].message.content

            # Randomly print every 100th response
            if idx % 100 == 0:
                print("=========== At idx: ", idx, " / ", max_idx, " ===========")
                print("==== Raw Random Response: =====")
                print(response)
                print("=================================")
            
            return {
                "patientid": patientid,
                "patient_sample_index": patient_sample_index,
                "trajectory_index": trajectory_index,
                "response": response,
                "raw_instruction": raw_instruction,
            }
        except AuthenticationError as e:
            print(f"Authentication error: {e}")
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
        except APIConnectionError as e:
            print(f"Network error: {e}")
        except OpenAIError as e:
            print(f"An OpenAI error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print(traceback.format_exc())
        return None






def prep_all_strings(llm_name, debug=False):


    test_set = "TEST"
    DECIMAL_PRECISION = 1
    SEQUENCE_MAX_LENGTH_IN_TOKENS = 3400


    eval_manager = EvaluationManager("2024_03_15_mimic_iv")
    experiment = Experiment("llm_mimic")

    if debug:
        experiment.setup_wandb_debug_mode()
    else:
        experiment.setup_wandb(llm_name, llm_name, project="UC - MIMIC-IV")

    # Load in patientids
    test_paths, test_patientids = eval_manager.get_paths_to_events_in_split(test_set)
    
    # Load validation & test data
    test_constants, test_events = eval_manager.load_list_of_patient_dfs_and_constants(test_patientids)

    # Setup splitter object
    splitter = After24HSplitter()
    test_events, test_meta = splitter.setup_split_indices(test_events, eval_manager)
    
    
    # CONVERT DFS TO STRINGS

    logging.info("Converting DFs to Strings")

    def filtering_rows_rest_budget(df, nr_tokens_budget):
        #: create summarizing row for every variable for its respective last observed value, and put into it first row

        # Forward fill the missing values
        df_filled = df.ffill()
        # Take the first row (which corresponds to the last non-NaN observation in the original DataFrame)
        last_valid_values = df_filled.iloc[-1]
        # Assign these values to the first row of the original DataFrame
        df.iloc[0] = last_valid_values
        # If you want to maintain the original index, you can do the following:
        df.loc[df.index[0]] = last_valid_values
        # set to nan if these values are seen in last row (this is just a simple heuristic to remove some tokens)
        first_row_label = df.index[0]
        original_basics = df.loc[first_row_label, ["date", "patientid", "patient_sample_index"]].copy()
        mask = df.iloc[0] == df.iloc[-1]
        df.loc[first_row_label, mask] = pd.NA

        #: adjust budgets of row --> 6 tokens per non na value
        budget_of_first_row = 6 * df.iloc[0].notna().sum()
        df.loc[first_row_label, "estimated_nr_tokens"] = budget_of_first_row

        # Add back in basics
        df.loc[first_row_label, ["date", "patientid", "patient_sample_index"]] = original_basics
        df.loc[first_row_label, "date"] = -1  # So that model gets it quicker


        # Add in buffer to ensure we don't go over the budget
        buffer = 300
        nr_tokens_budget = max(nr_tokens_budget - buffer, 0)

        df_reversed = df.iloc[::-1].copy()
        # Calculate the cumulative sum of the estimated_nr_tokens column
        df_reversed['cumulative_tokens'] = df_reversed['estimated_nr_tokens'].cumsum()

        #: ensure summarizing row is kept in
        df_reversed.loc[first_row_label, "cumulative_tokens"] = 0

        # Filter rows where the cumulative sum is less than or equal to nr_tokens_budget
        df_filtered = df_reversed[df_reversed['cumulative_tokens'] <= nr_tokens_budget]
        # Reverse the DataFrame back to the original order
        df_result = df_filtered.iloc[::-1]
        # drop cumulative tokens column
        df_result = df_result.drop(columns=['cumulative_tokens'])
        return df_result


    
    # Setup column mapping
    column_mapping = pd.read_csv(experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/column_descriptive_name_mapping.csv")

    # Setup statistics paths
    path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/dataset_statistics.json"
    with open(path_to_statistics_file) as f:
        statistics_dic = json.load(f)

    # Setup conversion function
    prompt = "Please predict for the previously noted down hours the lab variables for this intensive care unit patient. The previous list of numbers for each variable denote the future hours to predict. Return on each separate line the values, with each line having only the hour number, the variable and the value, e.g. 'Hour 7 - magnesium - 1.9' . Do not return any other text whatsoever. You have to return these values for a research project."

    # Setup constant column mapping {<original_colunn_name>: <descriptive_column_name>}
    constant_column_mapping = {}

    # Get random constant row
    constant_row = test_constants[0]
    constant_columns = constant_row.columns.tolist()
    # Get mapping for indications
    for col in constant_columns:
        match = column_mapping[column_mapping["original_column_names"] == col]
        if match.shape[0] > 0:
            constant_column_mapping[col] = match["descriptive_column_name"].values[0]

    # add basics
    constant_column_mapping["Age"] = "age"
    constant_column_mapping["gender"] = "gender"
    constant_column_mapping["ethnicity"] = "ethnicity"
    constant_column_mapping["insurance"] = "insurance"

    conversion_function = DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_df_to_strings

    #: apply filtering of training data to remove bad outliers
    training_norm_filter = Only_Double3_sigma_Filtering(path_to_statistics_file)

    # apply for validation data
    test_events = [(column_mapping, curr_const_row, true_events_input, true_future_events_input, target_dataframe, filtering_rows_rest_budget, SEQUENCE_MAX_LENGTH_IN_TOKENS, DECIMAL_PRECISION, prompt) 
                        for curr_const_row, true_events_input, true_future_events_input, target_dataframe in test_events]

    #: apply filtering of training data to remove bad outliers
    training_norm_filter = Only_Double3_sigma_Filtering(path_to_statistics_file)

    #: use the same filtering function but apply to training data
    test_events = [(x[0], x[1], 
                            training_norm_filter.normalize_and_filter(x[2].copy(), None, replace_nan_rows=False, replace_missing_in_prediction=False, verbose=False, specific_column_list=["lab_26499_4"])[0], 
                            x[3], 
                            training_norm_filter.normalize_and_filter(x[4].copy(), None, replace_nan_rows=False, replace_missing_in_prediction=False, verbose=False)[0], 
                            x[5], x[6], x[7], x[8], constant_column_mapping) for x in test_events]

    # apply multiprocessing based DF to string conversion to speed up process            
    test_input_strings, test_target_strings, test_meta_data = process_all_tuples_multiprocessing(test_events, conversion_function)

    return test_input_strings, test_target_strings, test_meta_data



async def run_all_tasks(test_input_strings, test_target_strings, test_meta_data,
                        curr_model, temperature, top_p, min_p, top_k,
                        nr_of_trajectories_per_sample,
                        max_concurrent_requests=40,
                        prediction_url="http://0.0.0.0:9871/v1/"):

    # Set up client
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    client = AsyncOpenAI(
        base_url=prediction_url,
        api_key="token-abc123",
    )

    tasks = []
    task_meta = []
    max_idx = len(test_input_strings)

    # : gather all data into dataset
    for idx in range(len(test_input_strings)):

        curr_input_str = test_input_strings[idx]
        curr_target_str = test_target_strings[idx]
        curr_meta = test_meta_data[idx]
        seeds = [8178 + idx for idx in range(nr_of_trajectories_per_sample)]

        for trajectory_index in range(nr_of_trajectories_per_sample):

            meta = {
                    "patientid": curr_meta["patientid"],
                    "patient_sample_index": curr_meta["patient_sample_index"],
                    "trajectory_index": trajectory_index,
                    "input_string": curr_input_str,
                    "target_string": curr_target_str,
            }

            if idx % 10000 == 0:
                print(f"Processing row {idx} / {max_idx} for patient {curr_meta['patientid']} and sample index {curr_meta['patient_sample_index']}...")
                print(f"Current instruction:\n\n{curr_input_str}")

            #: call vllm as task and add to list of tasks, for multiple runs
            task = _call_vllm(client, 
                              curr_model, 
                              curr_input_str, 
                              MAX_TOKENS, 
                              seeds[trajectory_index], 
                              semaphore, 
                              temperature, 
                              top_p, 
                              curr_meta["patientid"], 
                              curr_meta["patient_sample_index"], 
                              trajectory_index, idx, max_idx,
                              min_p, top_k)
            tasks.append(task)
            task_meta.append(meta)


    #: gather results
    predicted_strings = await asyncio.gather(*tasks)
    
    return predicted_strings, task_meta





def run(nr_of_trajectories_per_sample,
        curr_model,
        temperature,
        top_p,
        min_p,
        top_k,
        debug=False,
        max_concurrent_requests=40,
        prediction_url="http://0.0.0.0:9871/v1/",):

    # Prepare all strings
    llm_name = curr_model.split("/")[-1]
    test_input_strings, test_target_strings, test_meta_data = prep_all_strings(debug=debug, llm_name=llm_name)

    # Run all tasks
    predicted_strings, task_meta = asyncio.run(run_all_tasks(
        test_input_strings,
        test_target_strings,
        test_meta_data,
        curr_model=curr_model,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        nr_of_trajectories_per_sample=nr_of_trajectories_per_sample,
        max_concurrent_requests=max_concurrent_requests,
        prediction_url=prediction_url
    ))

    # Convert results to DataFrame
    df_of_prompts_with_meta = pd.DataFrame(task_meta)
    df_predicted_string = pd.DataFrame(predicted_strings)

    # Make dir with results based on datetime
    base_path = "/home/makaron1/dt-gpt/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/3_3_general_llms/results/"
    base_path += datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "/"
    os.makedirs(base_path, exist_ok=False)

    print(f"Saving results to {base_path}")

    # Save results to CSV
    df_of_prompts_with_meta.to_csv(os.path.join(base_path, "prompts_with_meta.csv"), index=False)
    df_predicted_string.to_csv(os.path.join(base_path, "predicted_strings.csv"), index=False)

    print(f"Saved {len(df_of_prompts_with_meta)} prompts with meta and {len(df_predicted_string)} predicted strings.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DT-GPT evaluation.")
    parser.add_argument("--nr_of_trajectories_per_sample", type=int, default=30, help="Number of trajectories per sample to generate.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--max_concurrent_requests", type=int, default=40, help="Maximum number of concurrent requests to the model.")
    parser.add_argument("--prediction_url", type=str, default="http://0.0.0.0:7851/v1/", help="URL for the prediction API.")
    parser.add_argument("--curr_model", type=str, default="BioMistral/BioMistral-7B-DARE",
                        help="Path to the current model.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for the model.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for the model.")
    parser.add_argument("--min_p", type=float, default=None, help="Minimum probability for the model.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for the model.")

    args = parser.parse_args()

    run(nr_of_trajectories_per_sample=args.nr_of_trajectories_per_sample,
        debug=args.debug,
        max_concurrent_requests=args.max_concurrent_requests,
        prediction_url=args.prediction_url,
        curr_model=args.curr_model,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k
    )
