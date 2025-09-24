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




async def _call_vllm(client, model_to_use, instruction_1, instruction_2, max_tokens, seed, semaphore, temperature, 
                     top_p, idx, max_idx):


    # Using direct completions, as in paper
    async with semaphore:
        try:

            modified_instruction_1 = instruction_1
            modified_instruction_1 = modified_instruction_1.replace('"', '')

            completion_1 = await client.completions.create(
                model=model_to_use,
                prompt=modified_instruction_1,   
                extra_body={            
                    "seed": seed,
                    "temperature": temperature,
                    "top_p": top_p,
                    "include_stop_str_in_output": True,
                    "max_tokens": 500,
                    # "truncate_prompt_tokens": 4096
                }
            )
            response_1 = completion_1.choices[0].text

            new_response = "[INST]" + instruction_1 + "[/INST]" + response_1 +  "[INST]" + instruction_2 + "[/INST]"

            # Generate second completion
            completion_2 = await client.completions.create(
                model=model_to_use,
                prompt=new_response,
                extra_body={            
                    "seed": seed,
                    "temperature": temperature,
                    "top_p": top_p,
                    "include_stop_str_in_output": True,
                    "max_tokens": 200,
                    "stop": ["[INST]", "}["]
                    # "truncate_prompt_tokens": 4096
                }
            )
            response_2 = completion_2.choices[0].text

            # Randomly print every 100th response
            if random.randint(0, 100) == 0:
                print("=========== At idx: ", idx, " / ", max_idx, " ===========")
                print("==== Raw Random Response: =====")
                print(response_1)

                print("==== New full instruction: =====")
                print(new_response)            

                print("==== Raw Random Response 2: =====")
                print(response_2)
                print("=================================")

            return {"response_1": response_1, "response_2": response_2}
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



async def _run_across_all_patients(df_of_prompts_with_meta,
                                   prompt_1_field, 
                                   prompt_2_field,
                                   max_concurrent_requests=40,
                                   prediction_url="http://0.0.0.0:18101/v1/",
                                   model_to_use="/home/makaron1/dt-gpt/models/2024_04_03_coolguy/checkpoint-44028",
                                   max_tokens=5000,
                                   temperature=1.0,
                                   top_p=0.9,
                                   nr_runs_per_query=30,
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

    # : gather all data into dataset
    for idx in range(len(df_of_prompts_with_meta)):
        
        curr_row = df_of_prompts_with_meta.iloc[idx]
        prompt_1 = curr_row[prompt_1_field]
        prompt_2 = curr_row[prompt_2_field]
        meta = curr_row.copy()
        meta = meta.drop([prompt_1_field, prompt_2_field, "prompt_two_steps_uncertain"],
                         axis=0,
                         errors="ignore")

        #: call vllm as task and add to list of tasks, for multiple runs
        seeds = [9178 + i for i in range(nr_runs_per_query)]
        for i in range(nr_runs_per_query):
            task = _call_vllm(client, model_to_use, prompt_1, prompt_2, max_tokens, seeds[i], 
                              semaphore, temperature, top_p, idx, max_idx)
            tasks.append(task)
            task_meta.append(meta)


    #: gather results
    predicted_strings = await asyncio.gather(*tasks)
    
    return predicted_strings, task_meta








def get_output_for_llm(use_subset=True):
    
    wandb.init(project="UC2 - NSCLC- Zero Shot")
    
    # Load data
    df_of_prompts_with_meta = pd.read_csv("/home/makaron1/dt-gpt/uc2_nsclc/6_zero_shot_exploration/2024_04_19_generate_data/2024_07_23_whole_test_set.csv")

    # Subset
    if use_subset:
        print("!!!!!!!!!!!!!!!!! USING SUBSET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        df_subset = pd.read_csv("/home/makaron1/dt-gpt/uc2_nsclc/6_zero_shot_exploration/2024_05_21_evaluate/2025_02_07_zero_shot_results_final_uncertain.csv")
        df_subset = df_subset[["patientid", "patient_sample_index"]].drop_duplicates()
        df_of_prompts_with_meta = df_of_prompts_with_meta.merge(df_subset, on=["patientid", "patient_sample_index"], how="inner")

    #: call async
    predicted_strings, meta = asyncio.run(_run_across_all_patients(df_of_prompts_with_meta,
                                                                   prompt_1_field="input_strings", 
                                                                    prompt_2_field="prompt_two_steps",
                                                                    nr_runs_per_query=30
    ))

    # Make dir with results based on datetime
    base_path = "/home/makaron1/dt-gpt/uc2_nsclc/8_revision/zero_shot_rerun/results/"
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

    print("Finished processing all prompts.")
    wandb.finish()
   
   

if __name__ == "__main__":
    get_output_for_llm(use_subset=True)

