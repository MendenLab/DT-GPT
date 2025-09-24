import __init__  # Do all imports
from pipeline.data_generators.DataFrameConverters import DataFrameConverter
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import random
import pandas as pd
import numpy as np
import json
import re
import math
from datetime import datetime
import traceback
import decimal
from collections.abc import Container
import numbers
import logging
from transformers import LlamaTokenizer
import os



class DTGPTDataFrameConverterTemplateTextBasicDescription(DataFrameConverter):

    tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", truncation_side="left")
    base_path = os.path.dirname(__file__).split("/uc2_nsclc")[0] + "/uc2_nsclc/"  # Hacky way to get the base path
    lot = pd.read_csv(base_path + "2_experiments/2023_11_07_neutrophils/1_data/line_of_therapy.csv")
    lot['startdate'] = pd.to_datetime(lot['startdate'], errors='coerce')
    lot = lot.dropna(subset=['startdate'])


    def _estimate_nr_tokens_per_row(df, column_name_mapping):

        #: create new DF with numbers same size as original DF - filled with 0s by default
        df_tokens = pd.DataFrame(0, index=df.index, columns=df.columns)

        # Set constants
        per_value_est_nr_tokens = 4 + 1 + 1   # 4 for value, 1 for comma, 1 for double dots
        json_per_row_est_tokens = 6

        #: for every variable, get its estimated nr of tokens from column_name_mapping["nr_tokens"] + per_value_est_nr_tokens
        for col in df.columns:

            if col in DTGPTDataFrameConverterTemplateTextBasicDescription.skip_columns:
                # Set to 0 for skip columns
                df_tokens.loc[:, col] = 0

            else:
                #: get how much to fill
                curr_fill_value = column_name_mapping.loc[column_name_mapping["original_column_names"] == col, "nr_tokens"].tolist()[0] + per_value_est_nr_tokens

                #: mask all nans with 0
                df_tokens.loc[~pd.isnull(df[col]), col] = curr_fill_value

        #: sum across row
        df_tokens["estimated_nr_tokens"] = df_tokens.sum(axis=1, numeric_only=True)

        #: add json extra costs
        df_tokens["estimated_nr_tokens"] = df_tokens["estimated_nr_tokens"] + json_per_row_est_tokens

        # Apply to original DF
        df_tokens = df_tokens.copy()
        df = df.copy()
        df["estimated_nr_tokens"] = df_tokens["estimated_nr_tokens"].copy()

        #: return
        return df

    def _calculate_nr_tokens_in_string(output_string):
        tokens = DTGPTDataFrameConverterTemplateTextBasicDescription.tokenizer(text=output_string)["input_ids"]
        num_tokens = len(tokens) - 1  # -1 since there is end of line
        return num_tokens



    def _get_columns_descriptive_mapping(original_columns, column_name_mapping):
        #: go through every column
        col_mapping = {}
        
        for col in original_columns:

            #: get mapping
            descriptive_info = column_name_mapping.loc[column_name_mapping["original_column_names"] == col]
            descriptive_info_name = descriptive_info["descriptive_column_name"].item()
                    
            #: note down
            col_mapping[col] = descriptive_info_name
        
        #: return mapping
        return col_mapping
    
    def _get_nr_tokens_in_string(str_input):

        tokens = re.split(r'[ :.{}]+', str_input)
        tokens = [token for token in tokens if token] # remove empty tokens
        nr_tokens = len(tokens)

        return nr_tokens


    def _convert_df_to_json(dataframe, skip_nan_values, column_name_mapping, max_token_input_length, decimal_precision, use_accumulated_dates=False, json_dict_row_wise=True):
        
        # Setup function
        assert skip_nan_values == True, "DataFrameConverter: currently only implemented that we skip nan values"
        if max_token_input_length is None:
            max_token_input_length = math.inf

        #: accumulate dates if needed
        if use_accumulated_dates:
            acc_dates = dataframe["date"].tolist()
            for idx in range(1, len(acc_dates)):
                acc_dates[idx] = acc_dates[idx] + acc_dates[idx - 1]
            dataframe["date"] = acc_dates

        #: convert date to string
        dataframe["date"] = dataframe["date"].astype(int).astype(str)
    
        #: append row of date to date string (to keep JSON unique)
        dataframe["date"] = dataframe["date"] + [" days{" + str(i) + "}" for i in range(1, len(dataframe["date"]) + 1)]

        #: convert date to index of df
        dataframe = dataframe.set_index('date', inplace=False, drop=True)

        #: drop skip columns
        dataframe = dataframe.drop(DTGPTDataFrameConverterTemplateTextBasicDescription.skip_columns, axis=1, errors="ignore", inplace=False)
        
        #: convert <UNK> to "Unknown"
        dataframe = dataframe.replace("<UNK>", "Unknown", inplace=False)  

        #: convert variable names to descriptive versions
        new_column_names = DTGPTDataFrameConverterTemplateTextBasicDescription._get_columns_descriptive_mapping(dataframe.columns.tolist(), column_name_mapping)
        dataframe.rename(columns=new_column_names, inplace=True)

        # setup basics
        float_conversion_string = '{:.' + str(decimal_precision) + 'f}'

        def _attempt_json_conversion(curr_dataframe):
            
            if json_dict_row_wise:
                
                #: convert DF to dictionary and skip NAs (axis = 1 means it works row-wise)
                # convert all floats to max 2 decimal precision
                df_dic_series = curr_dataframe.apply(lambda x : x.dropna().apply(lambda x: float_conversion_string.format(x).rstrip('0').rstrip('.') if isinstance(x, float) else x).to_dict(), axis=1)
                df_dic = df_dic_series.to_dict()

            else:
                # Convert for target based on each variable, to be more concise and focused on one string
                processed_df = curr_dataframe.apply(lambda x : x.dropna().apply(lambda x: float_conversion_string.format(x).rstrip('0').rstrip('.') if isinstance(x, float) else x), axis=1)
                df_dic_unsorted = processed_df.to_dict(orient="list")  # This then converts formated by {column -> [values]}

                #: order alphabetically
                sorted_keys = sorted(df_dic_unsorted.keys())
                df_dic = {key: df_dic_unsorted[key] for key in sorted_keys}

                #: remove nans
                df_dic = {key: [x for x in df_dic[key] if not pd.isnull(x)] for key in df_dic.keys()}

                
            #: convert to JSON string, with orientated day first (i.e. index)
            df_json_str = json.dumps(df_dic)

            #: return
            return df_json_str, df_dic

        #: call _attempt_json_conversion, check if within nr of token limits, else call again with one day less in the beggining
        curr_dataframe = dataframe
        nr_iterations = min([dataframe.shape[0] if dataframe.shape[0] >= 2 else 2, 10])
        final_json = None
        final_dic = None
        final_nr_days = None

        for i in range(0, nr_iterations):

            # Attempt another time with fewer days
            curr_dataframe = dataframe.iloc[i:, :]
            
            # Try conversion
            df_json, df_dic_curr = _attempt_json_conversion(curr_dataframe)

            # Calculate nr tokens
            nr_tokens = DTGPTDataFrameConverterTemplateTextBasicDescription._get_nr_tokens_in_string(df_json)

            # Check if short enough
            if nr_tokens <= max_token_input_length:
                final_json = df_json
                final_dic = df_dic_curr
                break
        
        # Fall back to using 1 row as input, even if it is longer than specified length
        if final_json is None:
            final_json = df_json
            final_dic = df_dic_curr

        final_nr_days = curr_dataframe.shape[0]

        #: post process string json to remove "{<index>}" from keys to prevent overfitting. We do not do it on the dictionary here because then it collapses entries
        final_json = re.sub(r'\{(\d+)\}', '', final_json)

        # Setup meta data
        str_meta = {
            "nr_days" : final_nr_days,
        }

        #: return string, and return nr of days in resulting JSON as metadata
        return final_json, str_meta, final_dic





    def _get_patient_input_string(dataframe, skip_nan_values, column_name_mapping, max_token_input_length, decimal_precision,
                                    use_accumulated_dates=False, json_dict_row_wise=True):

        #: call _convert_df_to_json to get json out
        _, _, patient_history_dic = DTGPTDataFrameConverterTemplateTextBasicDescription._convert_df_to_json(dataframe, skip_nan_values=skip_nan_values, column_name_mapping=column_name_mapping, max_token_input_length=max_token_input_length, decimal_precision=decimal_precision, use_accumulated_dates=use_accumulated_dates, json_dict_row_wise=json_dict_row_wise)

        #: systematically convert to string
        final_string = ""

        #: go through every day and add to string
        for day in patient_history_dic.keys():
            
            if day.startswith("0 days"):
                added_string = "Patient visits for the first time, with the following values:"
            else:
                added_string = day + " after previous visit, patient visits again, with the following values:"

            for lab in patient_history_dic[day].keys():

                added_string += " " + str(lab) + " is " + str(patient_history_dic[day][lab]) + ","

            added_string = added_string[:-1]  # remove last comma
            added_string += ".\n"  #
            final_string += added_string
        
        patient_history_string = final_string        

        return patient_history_string





    
    def _get_columns_to_predict(target_dataframe, column_name_mapping):

        #: use accumulated dates
        target_dataframe_acc = target_dataframe.copy()
        acc_dates = target_dataframe["date"].tolist()
        for idx in range(1, len(acc_dates)):
            acc_dates[idx] = acc_dates[idx] + acc_dates[idx - 1]
        target_dataframe_acc["date"] = acc_dates

        #: init return dic, which which days to predict into the future
        basic_key_name = "Variables to predict for respective days"
        return_dic = {
            basic_key_name : {},
        }

        #: skip bad columns
        target_dataframe_conversion = target_dataframe_acc.drop(DTGPTDataFrameConverterTemplateTextBasicDescription.skip_columns, axis=1, errors="ignore", inplace=False)


        #: go through every variable note down the days that it is not nan
        for col in target_dataframe_conversion.columns:

            #: skip diagnosis
            if col.startswith("diagnosis."):
                continue

            #: get descriptive column name
            descriptive_column_name = DTGPTDataFrameConverterTemplateTextBasicDescription._get_columns_descriptive_mapping([col], column_name_mapping)[col]

            #: get non-na days as ints
            non_na_days_mask = ~pd.isnull(target_dataframe_conversion[col])
            non_na_days = target_dataframe_acc.loc[non_na_days_mask, "date"].tolist()

            #: note down in dic, if any days
            if len(non_na_days) > 0:
                return_dic[basic_key_name][descriptive_column_name] = [int(x) for x in non_na_days]
        
        #: order alphabetically
        sorted_keys = sorted(return_dic[basic_key_name].keys())

        # To create a new dictionary that is sorted, you can use a dictionary comprehension
        return_dic[basic_key_name] = {key: return_dic[basic_key_name][key] for key in sorted_keys}

        #: return only inner dictionary
        return return_dic[basic_key_name], return_dic


    
    def _convert_constant_row_to_string(constants_row,  curr_lot, curr_line_num):
        
        #: remove unnecessary columns
        constants_row = constants_row.drop(DTGPTDataFrameConverterTemplateTextBasicDescription.skip_columns, axis=1, errors="ignore", inplace=False)
        
        #: apply DataFrameConverter.constant_columns_mapping 
        curr_mapping = {}

        for col in DataFrameConverter.constant_columns_mapping.keys():

            col_descriptive_name = DataFrameConverter.constant_columns_mapping[col]
            
            curr_mapping[col] = col_descriptive_name

        constants_row = constants_row[list(DataFrameConverter.constant_columns_mapping.keys())] 
        constants_row = constants_row.rename(columns=curr_mapping, inplace=False)
        constants_row = constants_row.iloc[0]  # Turn into single row

        #: convert to dic
        constants_row_dic = constants_row.to_dict()

        #: add  curr_lot, curr_line_num
        constants_row_dic["Current line of therapy"] = curr_lot
        constants_row_dic["Current line number"] = curr_line_num
        
        #: make into string
        added_string = ""

        for lab in constants_row_dic.keys():

            added_string += " " + str(lab) + " is " + str(constants_row_dic[lab]) + ","

        added_string = added_string[:-1]  # remove last comma
        added_string += ".\n" 

        #: return
        return added_string

    
    def _convert_unnecessary_input_columns_to_nans(input_dataframe):
                
        #: convert all <UNK> values to nans
        input_dataframe = input_dataframe.replace("<UNK>", np.nan, inplace=False)   

        #: replace diagnosed stuff with "diagnosed"
        diagnosis_cols = [col for col in input_dataframe.columns.tolist() if col.startswith("diagnosis.")]
        input_dataframe[diagnosis_cols] = input_dataframe[diagnosis_cols].apply(lambda x: x.fillna(np.nan).mask(x.notna(), 'diagnosed'))

        #: all False values in death
        if "disease_death.death" in input_dataframe.columns:
            input_dataframe["disease_death.death"] = input_dataframe["disease_death.death"].replace(False, np.nan, inplace=False)

        #: return
        return input_dataframe
    

    def _get_current_lot_(input_dataframe):
        
        lot = DTGPTDataFrameConverterTemplateTextBasicDescription.lot

        # Step 1: Find the latest date in 'input_dataframe'
        latest_input_date = input_dataframe['date'].max()

        matching_patientids = input_dataframe['patientid'].unique()
        filtered_lot_by_patientid = lot[lot['patientid'].isin(matching_patientids)]

        # Step 3: Further filter 'lot' dataframe for startdates less than or equal to the latest date from 'input_dataframe'
        filtered_lot_by_date = filtered_lot_by_patientid[filtered_lot_by_patientid['startdate'] <= latest_input_date]

        # Step 4: Sort the filtered 'lot' dataframe by 'startdate' in descending order
        sorted_filtered_lot = filtered_lot_by_date.sort_values(by='startdate', ascending=False)

        # Step 5: Select the top row to get the 'linename' with the latest 'startdate'
        latest_linename = sorted_filtered_lot.iloc[0]['linename'] if not sorted_filtered_lot.empty else ""
        if latest_linename == "":
            print("No line name found!")

        latest_linenumber = int(sorted_filtered_lot.iloc[0]['linenumber']) if not sorted_filtered_lot.empty else -1

        return latest_linename, latest_linenumber
    

    def convert_df_to_strings(column_name_mapping, constants_row, true_events_input, true_future_events_input, true_events_output, input_filtering_function,
                               max_token_full_length, decimal_precision, 
                               prompt="Given the non small cell lung cancer patient's history, please predict for this patient the previously noted down variables and future days, in the same JSON format."):
        
        # assert that dates are actually dates
        assert is_datetime(true_events_input["date"]), "DataFrameConverter: Date needs to be a datetime object!"
        assert is_datetime(true_future_events_input["date"]), "DataFrameConverter: Date needs to be a datetime object!"
        assert is_datetime(true_events_output["date"]), "DataFrameConverter: Date needs to be a datetime object!"

        # Do checks
        if true_events_input.shape[0] == 0 or true_events_output.shape[0] == 0:
            if true_events_input.shape[0] > 0:
                patientid = true_events_input["patientid"].tolist()[0]
                patient_sample_index = true_events_input["patient_sample_index"].tolist()[0]
                print("Bad DF, returning empty for patientid & patient_sample_index" + str((patientid, patient_sample_index)))
            else:
                print("Bad DF, returning empty")
            return "", "", None

        # Get basics
        patientid = true_events_input["patientid"].tolist()[0]
        patient_sample_index = true_events_input["patient_sample_index"].tolist()[0]

        #: sort by date
        true_events_input = true_events_input.sort_values(by=['date'])
        true_events_input_original = true_events_input.copy()
        true_future_events_input = true_future_events_input.sort_values(by=['date'])
        true_events_output = true_events_output.sort_values(by=['date'])

        #: randomly sample UNK columns in output for non diagnosis columns and set progression to "Not Documented" in unknown case
        non_diagnosis_cols = [col for col in true_events_output.columns.tolist() if not col.startswith("diagnosis.")]
        if "progression_progression" in true_events_output.columns:
            true_events_output["progression_progression"] = true_events_output["progression_progression"].replace("<UNK>", "Not documented", inplace=False)
        true_events_output[non_diagnosis_cols] = true_events_output[non_diagnosis_cols].applymap(lambda x: x if x != "<UNK>" else np.nan)

        #  convert all <UNK> in diagnosis to NA, and all diagnosed
        diagnosis_cols = [col for col in true_events_output.columns.tolist() if col.startswith("diagnosis.")]
        true_events_output[diagnosis_cols] = true_events_output[diagnosis_cols].applymap(lambda x: "diagnosed" if x != "<UNK>" else np.nan)

        #: convert all "unknown" in drug dosage to "administered" - drugs only in input
        drug_cols = [col for col in true_events_input.columns.tolist() if col.startswith("drug.")]
        true_events_input[drug_cols] = true_events_input[drug_cols].applymap(lambda x: "administered" if x == "unknown" else x)


        # : get difference in days from output seq to last of the input
        input_days = true_events_input["date"].tolist()
        output_days = true_events_output["date"].tolist()
        last_day_input = true_events_input["date"].tolist()[-1]
        first_day_output = true_events_output["date"].tolist()[0]
        diff_input_output_days = (first_day_output - last_day_input).days

        # : convert dates into relative days from previous for columns
        true_events_input['date'] = true_events_input['date'].diff().dt.days
        true_future_events_input['date'] = true_future_events_input['date'].diff().dt.days
        true_events_output['date'] = true_events_output['date'].diff().dt.days

        # set first day to 0 or diff_input_output_days respectively
        true_events_input.iloc[0, true_events_input.columns.get_loc('date')] = 0
        true_future_events_input.iloc[0, true_future_events_input.columns.get_loc('date')] = diff_input_output_days
        true_events_output.iloc[0, true_events_output.columns.get_loc('date')] = diff_input_output_days
        
        #: convert unnecessary columns in true_events_input, true_future_events_input to nans via _convert_unnecessary_input_columns_to_nans
        true_events_input = DTGPTDataFrameConverterTemplateTextBasicDescription._convert_unnecessary_input_columns_to_nans(true_events_input)
        true_future_events_input = DTGPTDataFrameConverterTemplateTextBasicDescription._convert_unnecessary_input_columns_to_nans(true_future_events_input)

        #: first generate output
        true_events_output_str, true_events_output_meta, true_events_output_dic = DTGPTDataFrameConverterTemplateTextBasicDescription._convert_df_to_json(true_events_output.copy(), skip_nan_values=True, column_name_mapping=column_name_mapping, max_token_input_length=None,
                                                                                                                                       use_accumulated_dates=True, json_dict_row_wise=False, decimal_precision=decimal_precision)

        #: get columns to predict via _get_columns_to_predict
        output_columns_to_predict, output_columns_to_predict_dic = DTGPTDataFrameConverterTemplateTextBasicDescription._get_columns_to_predict(true_events_output, column_name_mapping)

        # get current LoT
        curr_lot, curr_line_num = DTGPTDataFrameConverterTemplateTextBasicDescription._get_current_lot_(true_events_input_original)

        #: add constant entry value via _convert_constant_row_to_string
        constants_row_str = DTGPTDataFrameConverterTemplateTextBasicDescription._convert_constant_row_to_string(constants_row, curr_lot, curr_line_num)


        #: estimate remaining budget
        nr_tokens_used_for_output = DTGPTDataFrameConverterTemplateTextBasicDescription._calculate_nr_tokens_in_string(true_events_output_str)
        nr_tokens_used_for_prompt = DTGPTDataFrameConverterTemplateTextBasicDescription._calculate_nr_tokens_in_string(prompt)
        nr_tokens_used_for_constant_row = DTGPTDataFrameConverterTemplateTextBasicDescription._calculate_nr_tokens_in_string(constants_row_str)
        nr_tokens_used_for_col_to_predict = DTGPTDataFrameConverterTemplateTextBasicDescription._calculate_nr_tokens_in_string(json.dumps(output_columns_to_predict))
        nr_tokens_budget = max_token_full_length - nr_tokens_used_for_output - nr_tokens_used_for_prompt - nr_tokens_used_for_constant_row - nr_tokens_used_for_col_to_predict

        # Estimate nr of tokens per row and apply input filtering
        true_events_input = DTGPTDataFrameConverterTemplateTextBasicDescription._estimate_nr_tokens_per_row(true_events_input, column_name_mapping)
        true_events_input = input_filtering_function(true_events_input, nr_tokens_budget)   
        true_events_input = true_events_input.drop(["estimated_nr_tokens"], axis=1, errors="ignore", inplace=False)  # Drop token column before conversion

        #: get strings for patient history
        true_events_input_str = DTGPTDataFrameConverterTemplateTextBasicDescription._get_patient_input_string(true_events_input, skip_nan_values=True, column_name_mapping=column_name_mapping, max_token_input_length=None, decimal_precision=decimal_precision)


        #: build input string as JSON as well
        string_input = "First, patient chronological patient history up until the current day. " + true_events_input_str
        string_input = string_input + "Next, the baseline data for the patient: " + constants_row_str + "\n"
        string_input = string_input + "Finally, the variables which you should predict, and for which days in the future from the current day: " + json.dumps(output_columns_to_predict) + "\n"
        string_input = string_input + "Now, your task is as follows: " + prompt

        string_input = re.sub(r'\{(\d+)\}', '', string_input)  # Remove the indices here of the days to ensure that the model doesn't overfit on those

        #: build output string, which is already directly JSON string
        string_output = true_events_output_str

        #: build meta data
        meta_data = {
            "output_data_meta" : true_events_output_meta,
            "first_date_output" : first_day_output.strftime("%Y/%m/%d %H:%M:%S"),
            "prediction_columns" : output_columns_to_predict_dic,
            "all_days_output": [x.strftime("%Y/%m/%d %H:%M:%S") for x in output_days],
            "nr_days_output" : true_events_output.shape[0],
            "nr_tokens_output": nr_tokens_used_for_output,
            "patientid": patientid,
            "patient_sample_index": patient_sample_index,
            "lot": curr_lot,
            "linenumber": curr_line_num,
        }

        #: return all values and meta data
        return string_input, string_output, meta_data


    def _get_columns_short_mapping(descriptive_columns, column_name_mapping):
        
        #: go through every column
        col_mapping = {}
        
        for col in descriptive_columns:

            #: get mapping
            descriptive_info = column_name_mapping.loc[column_name_mapping["descriptive_column_name"] == col]

            try:
                descriptive_info_column = descriptive_info["original_column_names"].item()

            except Exception:

                # try with two spaces -> issue with csv
                try:
                    print("Trying with 2 spaces")
                    column_name_descriptive_attempt_2 = col.replace(" ", "  ")
                    descriptive_info = column_name_mapping.loc[column_name_mapping["descriptive_column_name"] == column_name_descriptive_attempt_2]
                    descriptive_info_column = descriptive_info["original_column_names"].item()

                except Exception:
                    # In case of error, let it propagate up
                    print("Error with column: " + str(col))
                    raise ValueError()
                
                    
            #: note down
            col_mapping[col] = descriptive_info_column
        
        #: return mapping
        return col_mapping


    def convert_from_strings_to_df(column_name_mapping, string_output, all_prediction_days, patientid, patient_sample_index, 
                                   all_column_names, all_unk_columns, prediction_days_column_wise=None):

        #: basic assertions
        assert "patientid" in all_column_names and "patient_sample_index" in all_column_names and "date" in all_column_names, "DataFrameConverter: PatientID or patient_sample_index or date not in all_column_names!"

        #: in case of errors, send back empty df with patientid & patient_sample_index
        empty_df = pd.DataFrame(columns=all_column_names)
        empty_df.loc[0:len(all_prediction_days)] = [None] * len(empty_df.columns)
        empty_df["date"] = all_prediction_days                            
        empty_df['date'] = pd.to_datetime(empty_df['date'])
        empty_df["patientid"] = patientid
        empty_df["patient_sample_index"] = patient_sample_index
        empty_df = empty_df.reset_index()
        empty_df = empty_df.drop(["index"], axis=1, errors="ignore", inplace=False)


        #: try to convert from JSON string to dataframe, using orientation 'index'
        try:
            
            # Convert json string to dic
            data_as_dic = json.loads(string_output)

            #: fill in correct dates with NaNs
            if prediction_days_column_wise is not None:
                column_wise_date_mapping = prediction_days_column_wise['Variables to predict for respective days']
                
                #: first get all possible positions
                all_positions = sorted(list(set([y for x in column_wise_date_mapping.values() for y in x])))

                for col in column_wise_date_mapping.keys():
                    
                    positions = column_wise_date_mapping[col]
                    indices = [all_positions.index(x) for x in positions]

                    for idx in range(len(all_positions)):
                        if idx not in indices:
                            data_as_dic[col].insert(idx, np.nan)
            
            # Convert from dic to DF
            prediction_df = pd.DataFrame.from_dict(data_as_dic, orient='columns')

            #: sort by index
            prediction_df.sort_index(inplace=True)

        except Exception as e:
            print("Making empty dataframe due to JSON error for patientid: " + str(patientid) + " and patient_sample_index: " + str(patient_sample_index))
            print(string_output)
            raise e


        #: convert columns back to short name using _get_columns_short_mapping
        try:
            new_column_names = DTGPTDataFrameConverterTemplateTextBasicDescription._get_columns_short_mapping(prediction_df.columns.to_list(), column_name_mapping)
            prediction_df = prediction_df.rename(columns=new_column_names, inplace=False)
        
        except Exception:
            print("Making empty dataframe due to column error for patientid: " + str(patientid) + " and patient_sample_index: " + str(patient_sample_index))
            return empty_df
        

        #: fill in with in with NAs for all_column_names
        missing_columns = [col for col in all_column_names if col not in prediction_df.columns.to_list()]
        new_df = pd.DataFrame(index=prediction_df.index, columns=missing_columns)
        prediction_df = pd.concat([prediction_df, new_df], axis=1)

        #: convert Unknown to <UNK>
        prediction_df[all_unk_columns] = prediction_df[all_unk_columns].replace("Unknown", "<UNK>", inplace=False)

        #: convert NAs in all_unk_columns to <UNK>
        prediction_df[all_unk_columns] = prediction_df[all_unk_columns].replace(np.nan, "<UNK>", inplace=False)

        #: convert progression "Not Documented" to "<UNK>"
        if "progression_progression" in prediction_df.columns:
            prediction_df["progression_progression"] = prediction_df["progression_progression"].replace("Not documented", "<UNK>", inplace=False)

        # set diagnosed from "diagnosed" to their respective correct value
        diagnosis_columns = [col for col in prediction_df.columns.tolist() if col.startswith("diagnosis_")]
        prediction_df[diagnosis_columns] = prediction_df[diagnosis_columns].where(prediction_df[diagnosis_columns] != "diagnosed", 
                                                                                  [x.split("diagnosis.icd10.")[1] for x in prediction_df[diagnosis_columns].columns.to_list()], 
                                                                                  axis=1)

        #: set and convert date
        prediction_df["date"] = all_prediction_days
        prediction_df['date']= pd.to_datetime(prediction_df['date'])
        
        #: set correct order of columns
        prediction_df = prediction_df.loc[:, all_column_names]

        #: set patientid and patient_sample_index
        prediction_df['patientid'] = patientid
        prediction_df["patient_sample_index"] = patient_sample_index

        #: reset index and drop "index column"
        prediction_df = prediction_df.reset_index()
        prediction_df = prediction_df.drop(["index"], axis=1, errors="ignore", inplace=False)
        
        #: return
        return prediction_df






