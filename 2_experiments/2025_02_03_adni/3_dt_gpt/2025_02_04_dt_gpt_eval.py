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
from Experiment import Experiment
import wandb



#: set training
WANDB_DEBUG = False

#: set correct path
MODEL_PATH = "/pstore/data/dt-gpt/raw_experiments/uc2_nsclc/adni_dt_gpt/adni_dt_gpt/2025_02_20___11_20_10_850233/models/final"



async def _call_vllm(client, model_to_use, curr_instruction, max_tokens, seed, semaphore, temperature, 
                     top_p, patientid):


    # Using direct completions, as in paper

    async with semaphore:
        try:
            completion = await client.completions.create(
                model=model_to_use,
                prompt=curr_instruction,
                max_tokens=max_tokens,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
            )
            response = completion.choices[0].text

            print(response)
            
            return (patientid, response)
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



async def run_eval(model,
             num_runs_per_prediction=30, max_concurrent_requests=100, 
             prediction_url="http://0.0.0.0:21783/v1/",
             max_tokens=200, temperature=1.0, top_p=0.9,
             splitting_string="<patient_prediction>"):

    #: setup experiment
    experiment = Experiment("adni_dt_gpt")

    # Set up wandb
   # if WANDB_DEBUG:
   #     experiment.setup_wandb_debug_mode()
   # else:
   #     experiment.setup_wandb("DT-GPT Eval Run", "DT-GPT", project="UC - ADNI")

    #: load dataset
    test_data = pd.read_csv("/pstore/data/dt-gpt/uc4-alzheimers-disease/data/ADNI_DT_GPT/test.csv")


    #: add to prompt the splitting string
    test_data["INPUT"] = test_data["INPUT"] + splitting_string

    #: throw out targets
    test_data = test_data.drop(columns=['TARGET'])

    #: set up openai client to model
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    client = AsyncOpenAI(
        base_url=prediction_url,
        api_key="token-abc123",
    )

    #: generate tasks, num_runs_per_prediction per prompt
    seeds = [8719 + i for i in range(num_runs_per_prediction)]
    tasks = []
    for i in range(len(test_data)):
        curr_instruction = test_data.iloc[i]['INPUT']
        patientid = test_data.iloc[i]['PATIENT_ID']
        for j in range(num_runs_per_prediction):
            tasks.append(_call_vllm(client, model, curr_instruction, max_tokens, seeds[j], semaphore, temperature, top_p, patientid))

    #: get responses and save as DF
    responses = await asyncio.gather(*tasks)
    responses_df = pd.DataFrame(responses, columns=['patientid', 'responses'])

    #: save DF
    save_path = experiment.evaluation_meta_data_path + "predictions_raw_all_adni.csv"
    responses_df.to_csv(save_path, index=False)
    logging.info(f"Saved predictions to {save_path}")
    wandb.config.update({
        "num_runs_per_prediction": num_runs_per_prediction,
        "max_concurrent_requests": max_concurrent_requests,
        "prediction_url": prediction_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "response_save_path": save_path,
    }, allow_val_change=True)

    # Clean up wandb
    wandb.finish()



if __name__ == "__main__":

    asyncio.run(run_eval(num_runs_per_prediction=30,
                         model=MODEL_PATH))






