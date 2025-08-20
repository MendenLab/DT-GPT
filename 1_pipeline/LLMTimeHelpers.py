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


DELIMITER = ","
NR_DIGITS_AFTER_COMMA = 3


def _process_string_back_to_dataframe(response, target_dataframe, target_column, max_nr_steps_forward, 
                                      last_input_date, rescaling_factor, freq):
    
    #: prep return target df
    prediction_df = target_dataframe[["patientid", "patient_sample_index", "date"]].copy()
    patientid = prediction_df["patientid"].iloc[0]
    patient_sample_index = prediction_df["patient_sample_index"].iloc[0]
    interested_dates = target_dataframe["date"].tolist()
    interested_dates.sort()

    #: add in all dates
    #offset_max = pd.DateOffset(days=(max_nr_steps_forward) * 7) if freq == "W" else pd.DateOffset(hours=(max_nr_steps_forward))
    #offset_min = pd.DateOffset(days=7) if freq == "W" else if freq == "h" pd.DateOffset(hours=1) elif freq == "6MS" pd.DateOffset(months=6)
    offset_max = (
        pd.DateOffset(days=max_nr_steps_forward * 7) if freq == "W" else
        pd.DateOffset(hours=max_nr_steps_forward) if freq == "h" else
        pd.DateOffset(months=6 * max_nr_steps_forward) if freq == "6MS" else
        None  # Default case
    )
    offset_min = (
        pd.DateOffset(days=7) if freq == "W" else
        pd.DateOffset(hours=1) if freq == "h" else
        pd.DateOffset(months=6) if freq == "6MS" else
        None
    )

    #first_target_date = last_input_date + offset_min
    #last_target_date = last_input_date + offset_max
    first_target_date = interested_dates[0]
    last_target_date = interested_dates[-1]
    
    all_dates = pd.date_range(start=first_target_date, end=last_target_date, freq="7D" if freq == "W" else freq)
    all_dates_df = pd.DataFrame({'date': all_dates})
    filled_data = pd.merge(all_dates_df, prediction_df, on='date', how='left')
    prediction_df = filled_data
    prediction_df = prediction_df.sort_values('date').reset_index(drop=True)

    assert prediction_df["date"].max() >= target_dataframe["date"].max()
    assert prediction_df["date"].min() <= target_dataframe["date"].min()
    assert set(interested_dates).issubset(set(prediction_df["date"].tolist()))


    #: slice reponse by DELIMITER
    sliced_response = response.split(DELIMITER)
    sliced_response = [x for x in sliced_response if x != ""]

    def is_convertible_to_number(s):
        """Checks if a string can be converted to a number (int or float)."""
        try:
            if pd.isna(s):
                return True

            float(s)
            return True
        except ValueError:
            return False

    # Convert NaN to pd.NA, and throw out anything which isn't numeric
    sliced_response = [pd.NA if x == "NaN" else x for x in sliced_response]
    sliced_response = [x if is_convertible_to_number(x) else pd.NA for x in sliced_response]
    

    #: fill in with NAs for rest
    len_response = len(sliced_response)
    num_timepoints = len(prediction_df)
    if len_response < num_timepoints:
        sliced_response.extend([pd.NA] * (num_timepoints - len_response))
    if len_response > num_timepoints:
        sliced_response = sliced_response[:num_timepoints]
    
    #: assign to correct column
    prediction_df[target_column] = sliced_response
    prediction_df["patientid"] = patientid
    prediction_df["patient_sample_index"] = patient_sample_index

    #: rescale
    q, min_ = rescaling_factor
    prediction_df[target_column] = pd.to_numeric(prediction_df[target_column], errors="coerce")
    prediction_df[target_column] = (prediction_df[target_column] * q) + min_

    #: do forward filling for NAs
    prediction_df[target_column] = prediction_df[target_column].ffill()

    #: extract only needed dates
    prediction_df = prediction_df[prediction_df["date"].isin(interested_dates)]

    # Sort by dates
    prediction_df = prediction_df.sort_values("date").reset_index(drop=True)

    # Do forward and backward filling for NAs, if any are there
    if prediction_df[target_column].isnull().any():
        prediction_df[target_column] = prediction_df[target_column].ffill()
        prediction_df[target_column] = prediction_df[target_column].bfill()

    #: return dataframe
    return prediction_df




async def _call_vllm(client, model_to_use, curr_instruction, max_tokens, seed, semaphore, temperature, 
                     top_p, target_dataframe, target_column, max_nr_steps_forward, rescaling_factor,
                     last_input_date, freq):


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

            # Randomly print every 100th response
            if random.randint(0, 100) == 0:
                print("==== Raw Random Response: =====")
                print(response)
            
            #: process to dataframe
            response_df = _process_string_back_to_dataframe(response, target_dataframe, target_column, max_nr_steps_forward, 
                                                            last_input_date, rescaling_factor, freq)

            return response_df
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




async def _run_across_all_patients(list_of_split_dfs, eval_manager,
                                   freq, max_lookback_positions,
                                   dataset_stats,
                                   max_concurrent_requests=100,
                                   prediction_url="http://0.0.0.0:8001/v1/",
                                   model_to_use="meta-llama/Llama-2-7b-hf",
                                   max_tokens=130,
                                   temperature=1.0,
                                   top_p=0.9,
                                   alpha=0.99,
                                   beta=0.3,
                                   nr_runs_per_query=20,
                                   target_cols_override=None,
                                   add_trailing_comma=False,
    ):

     # Init eval manager for streaming
    eval_manager.evaluate_split_stream_start()

    # Setup cols - using all
    inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
    if target_cols_override is None:
        target_cols = target_cols.copy()
    else:
        target_cols = target_cols_override.copy()
    target_cols.extend(["date", "patientid", "patient_sample_index"])

    # Output saving
    tasks = []
    task_meta = []
    last_observed_values = {}

    # Setting up async
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    client = AsyncOpenAI(
        base_url=prediction_url,
        api_key="token-abc123",
    )



    # : gather all data into dataset
    for idx, (constants_row, true_events_input, true_future_events_input, target_dataframe) in enumerate(list_of_split_dfs):

        if idx % 100 == 0:
            logging.info("Generating data - current idx: " + str(idx) + " / " + str(len(list_of_split_dfs)))


        patientid = constants_row["patientid"].tolist()[0]
        patient_sample_index = true_events_input["patient_sample_index"].tolist()[0]

        # Setup last observed
        if patientid not in last_observed_values:
            last_observed_values[patientid] = {}
        last_observed_values[patientid][patient_sample_index] = {}

        #: extract columns to use
        true_events_input = true_events_input.loc[:, inputs_cols]
        true_future_events_input = true_future_events_input.loc[:, future_known_inputs_cols]
        target_dataframe = target_dataframe.loc[:, target_cols]
        target_dataframe_no_empty_rows = target_dataframe.dropna(axis=0, how='all', subset=target_dataframe.columns.difference(["patientid", "patient_sample_index", "date"]))
        samples_for_patient_sample = 0

        #: Call to model
        for target_column in target_cols:
            if target_column in ["date", "patientid", "patient_sample_index"]:
                continue

            # Skip if empty for this column
            if target_dataframe_no_empty_rows[target_column].isnull().all():
                continue
            
            #: extract input data
            input_data = true_events_input[[target_column, "date"]].copy()

            #: fill in missing dates using max_lookback and frequency
            if freq == "W":
                offset = pd.DateOffset(days=(max_lookback_positions - 1) * 7)
                date_range_freq = "7D"
            elif freq == "6MS":
                offset = pd.DateOffset(months=(max_lookback_positions - 1) * 6)
                date_range_freq = "6MS"
            elif freq == "h":
                offset = pd.DateOffset(hours=(max_lookback_positions - 1))
                date_range_freq = freq
            else:
                raise NotImplementedError("Freq not implemented: " + str(freq))

            latest_date = input_data['date'].max() 
            earliest_date_to_fill_in = latest_date - offset
            all_dates = pd.date_range(
                start=earliest_date_to_fill_in,
                end=latest_date,
                freq=date_range_freq
            )
            all_dates_df = pd.DataFrame({'date': all_dates})   # TODO: need to make this work correctly for 6MS
            filled_data = pd.merge(all_dates_df, input_data, on='date', how='left')
            filled_data = filled_data.sort_values('date').reset_index(drop=True)
            filled_data[target_column] = filled_data[target_column].astype(float)

            assert filled_data["date"].max() == true_events_input["date"].max()
            assert set(filled_data[target_column].dropna().tolist()) == set(true_events_input[true_events_input["date"] >= earliest_date_to_fill_in][target_column].dropna().tolist())
            assert len(filled_data[target_column].dropna()) == len(true_events_input[true_events_input["date"] >= earliest_date_to_fill_in][target_column].dropna())

            # Deal with edge cases
            last_input_data = true_events_input[true_events_input["date"] < earliest_date_to_fill_in]
            if filled_data[target_column].isnull().all():
                if true_events_input[target_column].isnull().all():
                    # In the rare case that there is no input data, fill in with mean of column
                    filled_data[target_column] = dataset_stats[target_column]["mean_double_3_sigma_filtered"]

                elif not last_input_data.empty:
                    # In the rare edge case that there are no values since the lookback period, 
                    # we fill in artifically the first value with the last value in the input 
                    # (so that the model has at least something to work with for fairness)
                    last_value = last_input_data[target_column].dropna().tolist()[-1]
                    filled_data.loc[filled_data.index[0], target_column] = last_value
                else:
                    raise Exception("Should not happen")

            #: scaling as in paper
            history = filled_data[target_column].values
            history = history[~np.isnan(history)]
            min_ = np.min(history) - beta*(np.max(history)-np.min(history))
            q = np.quantile(history-min_, alpha)
            if q == 0:
                # Fallback to 1
                q = 1
            filled_data[target_column] = (filled_data[target_column] - min_) / q

            #: convert to text (truncate to NR_DIGITS_AFTER_COMMA), by adding DELIMITER for target column
            ret_str = ""
            for idx, row in filled_data.iterrows():
                ret_str += DELIMITER if idx > 0 else ""
                
                # Add content
                if pd.isnull(row[target_column]):
                    ret_str += "NaN"
                else:
                    ret_str += str(round(row[target_column], NR_DIGITS_AFTER_COMMA))
            
            if add_trailing_comma:
                ret_str += DELIMITER

            # Add meta
            if true_events_input[target_column].isnull().all():
                # median backup, as in copy forward baseline
                last_observed_values[patientid][patient_sample_index][target_column] = dataset_stats[target_column]["median"]
            else:
                # Last observed
                last_observed_values[patientid][patient_sample_index][target_column] = true_events_input[target_column].dropna().tolist()[-1]

            #: call vllm as task and add to list of tasks, for multiple runs
            seeds = [9178 + i for i in range(nr_runs_per_query)]
            for i in range(nr_runs_per_query):
                task = _call_vllm(client, model_to_use, ret_str, max_tokens, seeds[i], semaphore, 
                                temperature, top_p, target_dataframe_no_empty_rows, target_column, len(all_dates), (q, min_),
                                latest_date, freq)
                tasks.append(task)
                samples_for_patient_sample += 1

        if samples_for_patient_sample > 0:
            task_meta.append((target_dataframe_no_empty_rows, patientid, patient_sample_index))


    #: gather results
    predicted_dfs = await asyncio.gather(*tasks)
    
    #: sort into dictionary for easy access
    predicted_df_dic = {}
    for predicted_df in predicted_dfs:
        if predicted_df is None:
            continue
        patientid = predicted_df["patientid"].iloc[0]
        patient_sample_index = predicted_df["patient_sample_index"].iloc[0]
        predicted_column = predicted_df.columns.difference(["patientid", "patient_sample_index", "date"])[0]
        if patientid not in predicted_df_dic:
            predicted_df_dic[patientid] = {}
        if patient_sample_index not in predicted_df_dic[patientid]:
            predicted_df_dic[patientid][patient_sample_index] = {}
        if predicted_column not in predicted_df_dic[patientid][patient_sample_index]:
            predicted_df_dic[patientid][patient_sample_index][predicted_column] = []
        
        predicted_df_dic[patientid][patient_sample_index][predicted_column].append(predicted_df)


    #: median predictions for same patient and patient_sample_index and column, and also rescale back to normal
    averaged_columns = {}
    for patientid, patient_sample_index_dic in predicted_df_dic.items():
        averaged_columns[patientid] = {}
        for patient_sample_index, predicted_dfs in patient_sample_index_dic.items():
            averaged_columns[patientid][patient_sample_index] = []
            for predicted_column, predicted_df_list in predicted_dfs.items():
                
                # Concat all datasets
                concat_datasets = pd.concat(predicted_df_list)

                # In case of all NaNs for a specific date, use last observed of column for that entry
                for date in concat_datasets["date"].unique():
                    if concat_datasets[concat_datasets["date"] == date][predicted_column].isnull().all():
                        concat_datasets.loc[concat_datasets["date"] == date, predicted_column] = last_observed_values[patientid][patient_sample_index][predicted_column] 

                #: extract only numeric and non nan
                concat_datasets_numeric_nonna = concat_datasets[concat_datasets[predicted_column].notnull()]

                # Check that no more nans
                if concat_datasets_numeric_nonna[predicted_column].isnull().all():
                    raise Exception("Should not happen")
                    
                # Apply median
                averaged_dfs = concat_datasets_numeric_nonna.groupby(["date", "patientid", "patient_sample_index"]).median().reset_index()
                averaged_columns[patientid][patient_sample_index].append(averaged_dfs)

    # In case of empty meta, just return empty dataframes
    if len(task_meta) == 0:
        logging.info("No samples to generate!!!!!!!!!")
        return pd.DataFrame(), pd.DataFrame()

    #: send to eval manager
    for target_dataframe_no_empty_rows, patientid, patient_sample_index in task_meta:
        
        dates = averaged_columns[patientid][patient_sample_index][0]["date"].tolist()
        col_dfs = [x[x.columns.difference(["patientid", "patient_sample_index", "date"])] 
                   for x in averaged_columns[patientid][patient_sample_index]]
        pred = pd.concat(col_dfs, axis=1)
        pred["patientid"] = patientid
        pred["patient_sample_index"] = patient_sample_index
        pred["date"] = dates
        
        #: fill in missing columns
        for col in target_dataframe_no_empty_rows.columns.difference(pred.columns):
            pred[col] = pd.NA

        # Reorder for better debug
        pred = pred[target_dataframe_no_empty_rows.columns]

        assert len(pred) == len(target_dataframe_no_empty_rows)

        eval_manager.evaluate_split_stream_prediction(pred, target_dataframe_no_empty_rows, patientid, patient_sample_index)

    #: post process predictions
    logging.info("Finished generating samples")

    #: do full eval
    full_df_targets, full_df_prediction = eval_manager.concat_eval()

    return full_df_targets, full_df_prediction




def get_output_for_llm(list_of_split_dfs, eval_manager,
                       freq, max_lookback_positions, dataset_stats,
                       model_to_use,
                    max_concurrent_requests=100,
                    prediction_url="http://0.0.0.0:8001/v1/",
                    max_tokens=100,
                    temperature=1.0,
                    top_p=0.9,
                    alpha=0.99,
                    beta=0.3,
                    nr_runs_per_query=20,
                    target_cols_override=None,
                    add_trailing_comma=False
    ):

    logging.info("Starting to generate samples")
    logging.info("Freq: " + str(freq))
    logging.info("Max lookback positions: " + str(max_lookback_positions))
    logging.info("Model to use: " + str(model_to_use))
    logging.info("Max concurrent requests: " + str(max_concurrent_requests))
    logging.info("Prediction URL: " + str(prediction_url))
    logging.info("Max tokens: " + str(max_tokens))
    logging.info("Temperature: " + str(temperature))
    logging.info("Top p: " + str(top_p))
    logging.info("Alpha: " + str(alpha))
    logging.info("Beta: " + str(beta))
    logging.info("Nr runs per query: " + str(nr_runs_per_query))
    logging.info("Target cols override: " + str(target_cols_override))
    

    #: call async
    full_df_targets, full_df_prediction = asyncio.run(_run_across_all_patients(list_of_split_dfs, eval_manager,
                                                                               freq, max_lookback_positions,
                                                                               dataset_stats=dataset_stats,
                                                                                max_concurrent_requests=max_concurrent_requests,
                                                                                prediction_url=prediction_url,
                                                                                model_to_use=model_to_use,
                                                                                max_tokens=max_tokens,
                                                                                temperature=temperature,
                                                                                top_p=top_p,
                                                                                alpha=alpha,
                                                                                beta=beta,
                                                                                nr_runs_per_query=nr_runs_per_query,
                                                                                target_cols_override=target_cols_override,
                                                                                add_trailing_comma=add_trailing_comma)
    )
   
    # Return
    return full_df_targets, full_df_prediction


