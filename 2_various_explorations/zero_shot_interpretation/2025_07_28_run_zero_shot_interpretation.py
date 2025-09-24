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
import random
from datetime import datetime
import wandb
import argparse


COL_INITIAL_INSTRUCTION = "input_strings"
COL_INITIAL_RESPONSE = "response_1"
COL_ZERO_SHOT_INSTRUCTION = "prompt_two_steps"  # Alternatively: prompt_two_steps_uncertain
COL_ZERO_SHOT_RESPONSE = "response_2"

SEED = 9178  # Fixed seed for reproducibility, can be changed as needed
temperature = 1.0
top_p = 0.9



async def _call_vllm(client, semaphore, model_to_use, instruction, patientid, patient_sample_index, idx, max_idx,
                     seed, temperature, top_p):

    async with semaphore:
        try:

            modified_instruction_1 = instruction
            modified_instruction_1 = modified_instruction_1.replace('"', '')

            completion_1 = await client.completions.create(
                model=model_to_use,
                prompt=modified_instruction_1,   
                extra_body={            
                    "seed": seed,  # We want a fixed seed so that the effect of different outputs for the same variable can be compared
                    "temperature": temperature,
                    "top_p": top_p,
                    "include_stop_str_in_output": True,
                    "max_tokens": 125,
                }
            )
            response = completion_1.choices[0].text

            # Randomly print every 100th response
            if idx % 100 == 0:
                print("=========== At idx: ", idx, " / ", max_idx, " ===========")
                print("==== Raw Random Response: =====")
                print(response)
                print("=================================")

            return response
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



def make_instruction_string(initial_instruction, initial_response, zero_shot_instruction, zero_shot_response, number_of_features):
    
    query_instruction = f"What are the {number_of_features} most important variables or patient's baseline characteristics for your prediction?"
    
    new_instruction = "[INST]" + initial_instruction + "[/INST]" + \
                    initial_response + \
                    "[INST]" + zero_shot_instruction + "[/INST]" + \
                        zero_shot_response + \
                    "[INST]" + query_instruction + "[/INST]"
    return new_instruction




async def _run_across_all_patients(df_of_prompts_with_meta,
                                   number_of_features, creative,
                                   max_concurrent_requests=40,
                                   prediction_url="http://0.0.0.0:18101/v1/",
                                   model_to_use="/home/makaron1/dt-gpt/models/2024_04_03_coolguy/checkpoint-44028",
                                   ):


    # Setting up async
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    client = AsyncOpenAI(
        base_url=prediction_url,
        api_key="token-abc123",
    )

    tasks = []
    task_meta = []
    max_idx = len(df_of_prompts_with_meta)
    all_seeds = [9178 + i for i in range(len(df_of_prompts_with_meta))]

    # : gather all data into dataset
    for idx in range(len(df_of_prompts_with_meta)):
        
        curr_row = df_of_prompts_with_meta.iloc[idx]
        
        curr_instruction = make_instruction_string(
            curr_row[COL_INITIAL_INSTRUCTION],
            curr_row[COL_INITIAL_RESPONSE],
            curr_row[COL_ZERO_SHOT_INSTRUCTION],
            curr_row[COL_ZERO_SHOT_RESPONSE],
            number_of_features
        )

        if idx % 10000 == 0:
            print(f"Processing row {idx} / {max_idx} for patient {curr_row['patientid']} and sample index {curr_row['patient_sample_index']}...")
            print(f"Current instruction:\n\n{curr_instruction}")

        meta = curr_row.copy()
        meta = meta.drop([COL_INITIAL_INSTRUCTION, COL_INITIAL_RESPONSE, 
                            COL_ZERO_SHOT_INSTRUCTION, COL_ZERO_SHOT_RESPONSE,
                          "prompt_two_steps_uncertain"],
                         axis=0,
                         errors="ignore")
        
        # Use different or same seed
        if creative:
            curr_seed = all_seeds[idx]
        else:
            curr_seed = SEED

        #: call vllm as task and add to list of tasks, for multiple runs
        task = _call_vllm(client, semaphore, model_to_use, curr_instruction,
                          curr_row["patientid"], curr_row["patient_sample_index"],
                          idx, max_idx, curr_seed, temperature, top_p)
        tasks.append(task)
        task_meta.append(meta)


    #: gather results
    predicted_strings = await asyncio.gather(*tasks)
    
    return predicted_strings, task_meta




def get_output_for_model(creative, number_of_features_to_predict, prediction_url):

    wandb.init(project="UC2 - NSCLC - Zero Shot & Interpretation")
    wandb.run.name = "Zero shot intepretration - creative: " + str(creative) + " number of features: " + str(number_of_features_to_predict)

    # Load in correct dataframes
    df_of_prompts_with_meta = pd.read_csv("/home/makaron1/dt-gpt/uc2_nsclc/6_zero_shot_exploration/2024_04_19_generate_data/2024_07_23_whole_test_set.csv")
    df_subset = pd.read_csv("/home/makaron1/dt-gpt/uc2_nsclc/6_zero_shot_exploration/2024_05_21_evaluate/2025_02_07_zero_shot_results_final_uncertain.csv")
    df_subset = df_subset[["patientid", "patient_sample_index"]].drop_duplicates()
    df_of_prompts_with_meta = df_of_prompts_with_meta.merge(df_subset, on=["patientid", "patient_sample_index"], how="inner")

    df_generated_predictions = pd.read_csv("/home/makaron1/dt-gpt/uc2_nsclc/8_revision/zero_shot_rerun/results/2025_02_09_01_33_48/results.csv")
    df_generated_predictions_meta = pd.read_csv("/home/makaron1/dt-gpt/uc2_nsclc/8_revision/zero_shot_rerun/results/2025_02_09_01_33_48/meta.csv")
    df_generated_predictions = pd.concat([df_generated_predictions, df_generated_predictions_meta], axis=1)

    # Load in together
    df_full_prompts = df_generated_predictions.merge(df_of_prompts_with_meta, on=["patientid", "patient_sample_index", "variable"], how="left", 
                                                    suffixes=("", "_meta"))
    
    print(f"Loaded {len(df_full_prompts)} prompts for zero-shot interpretation.")


    # Run
    predicted_strings, meta = asyncio.run(_run_across_all_patients(df_full_prompts, number_of_features_to_predict, creative, prediction_url=prediction_url))

    # Make dir with results based on datetime
    base_path = "/home/makaron1/dt-gpt/uc2_nsclc/8_revision/zero_shot_interpretation/results/"
    base_path += datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "/"
    os.makedirs(base_path, exist_ok=False)

    # Save locally
    curr_results = base_path + "results.csv"
    full_results_df = pd.DataFrame(predicted_strings)
    full_results_df.to_csv(curr_results, index=False)
    
    # Save metadata
    curr_meta = base_path + "meta.csv"
    full_meta_df = pd.DataFrame(meta)
    full_meta_df.to_csv(curr_meta, index=False)

    print(f"Results saved to {curr_results}")

    print("Finished processing all prompts.")
    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run zero-shot interpretation for NSCLC dataset.")
    # Store the creative flag as a boolean
    parser.add_argument("--creative", action="store_true", help="Run in creative mode.")
    parser.add_argument("--number_of_features_to_predict", type=str, default="five", help="Number of features to predict in the zero-shot interpretation.")
    parser.add_argument("--prediction_url", type=str, default="http://0.0.0.0:18101/v1/", help="Number of features to predict in the zero-shot interpretation.")
    

    get_output_for_model(
        creative=parser.parse_args().creative,
        number_of_features_to_predict=parser.parse_args().number_of_features_to_predict,
        prediction_url=parser.parse_args().prediction_url
    )

