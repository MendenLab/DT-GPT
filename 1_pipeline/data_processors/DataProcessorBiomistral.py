import __init__  # Do all imports
import logging
import wandb
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, LongT5Model, DataCollatorForSeq2Seq, T5Tokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from DataFrameConvertJSONLlama import DTGPTDataFrameConverterLlama_1_0
import re
from trl import DataCollatorForCompletionOnlyLM


class DataProcessorBiomistral():

    def __init__(self, experiment, path_to_statistics_file, column_name_mapping, model_to_use,
                 max_total_length=4000, collator_setting="completion"):


        #: make tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_to_use, 
                                                        force_download=False,
                                                        truncation_side="left",
                                                        padding_side="left",
                                                        add_eos_token=True,
                                                        add_bos_token=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.chat_template = ""

        # Set constants
        self.max_total_length = max_total_length
        self.column_name_mapping = column_name_mapping

        self.all_columns = None
        self.all_unk_columns = None

        # Processed data
        self.processed_dataset = {}

        self.collator_setting = collator_setting
        self.response_template = "<patient_prediction>"
        self.data_collator = None

        self.current_converter = DTGPTDataFrameConverterLlama_1_0

        # Load statistics file
        with open(path_to_statistics_file) as f:
            self.statistics = json.load(f)

    def setup_cols(self, target_cols):
        #: set up columns
        self.all_columns = target_cols.copy()

        self.all_columns.insert(0, "patientid")
        self.all_columns.insert(1, "patient_sample_index")
        self.all_columns.insert(2, "date")
        self.all_unk_columns = [col for col in self.all_columns if col.startswith("diagnosis") or col.startswith("discontinuationreason")] #: get all <UNK> columns

    ##################################### MODES ##################################################
    
    def set_for_training(self):
        self.tokenizer.add_eos_token = True
    
    def set_for_inference(self):
        self.tokenizer.add_eos_token = False

    def set_converter(self, converter):
        self.current_converter = converter


    ##################################### DATASET PREPROCESSING ##################################################

    def preprocess_dataset(self, list_of_input_strings, list_of_target_strings, tokenize=True):
        
        #: call _preprocess_dic
        l = self._preprocess_dic(list_of_input_strings, list_of_target_strings, tokenize=tokenize)
        return l
    
    def _preprocess_dic(self, list_of_input_strings, list_of_target_strings, tokenize):
        
        # : see https://discuss.huggingface.co/t/longt5-fine-tunning/22650
        logging.info("DataProcessor: tokenizing dataset!")

        #: gather inputs and outputs
        all_inputs = list_of_input_strings
        all_outputs = list_of_target_strings

        # Preprocess inputs
        logging.info("Started preprocessing inputs for basic stuff")
        all_inputs = self.preprocess_inputs(all_inputs)
        logging.info("Finished preprocessing inputs for basic stuff")

        #: preprocess outputs
        logging.info("Started preprocessing outputs for basic stuff")
        all_outputs = self.preprocess_outputs(all_outputs)
        logging.info("Finished preprocessing outputs for basic stuff")

        # Setup dataset
        curr_dataset = Dataset.from_dict({"input_text": all_inputs, "target_text": all_outputs})

        #: preprocess function
        def preprocess_function(samples):

            inputs = samples['input_text']
            targets = samples['target_text']

            #: concatenate inputs & outputs, since we're doing causal LM
            concat_text = [str(inputs[x]) + " " + str(targets[x]) for x in range(len(inputs))]
            
            #: preprocess output using tokenizer
            if tokenize:
                model_inputs = self.tokenizer(text=concat_text, max_length=self.max_total_length, truncation=True)
            else:
                model_inputs = {}

            model_inputs["concatenated_text"] = concat_text

            #: return lists
            return model_inputs

        # Perform actual tokenization
        tokenized_dataset = curr_dataset.map(preprocess_function, batched=True)

        logging.info("DataProcessor: finished tokenizing dataset!")

        return tokenized_dataset
    
    def preprocess_inputs(self, input_string_list):
        
        # In completion loss setting we need the response template in the input
        if self.collator_setting == "completion":
            input_string_list = [x + " " + self.response_template for x in input_string_list]

        #: remove " since it creates unnecessary labels in the input
        ret_string_list = [x.replace('"', '') for x in input_string_list]

        return ret_string_list

    def preprocess_outputs(self, output_string_list):

        return output_string_list
    


    ##################################### DF -> STRING ##################################################

    def convert_to_string_single_patient(self, constants_row, true_events_input, true_future_events_input, target_dataframe, input_filtering_function, prompt, decimal_precision=2,
                                         index_version=False, generated_trajectories_df_path=None, **kwargs):
        
        assert generated_trajectories_df_path is None
        assert index_version is False

        str_input, str_output, meta_data = self.current_converter.convert_df_to_strings(self.column_name_mapping, 
                                                                                        constants_row, true_events_input, true_future_events_input, target_dataframe,
                                                                                        input_filtering_function=input_filtering_function,
                                                                                        max_token_full_length=self.max_total_length,
                                                                                        decimal_precision=decimal_precision,
                                                                                        prompt=prompt,
                                                                                        **kwargs)


        #: return strings and meta data
        return str_input, str_output, meta_data

    
    ##################################### STRING -> DF ##################################################

    
    def convert_to_df_single_patient(self, prediction_string, all_prediction_days, patientid, patient_sample_index, prediction_days_column_wise=None):

        # NOTE: all_prediction_days can be gotten from the meta_data["all_days_output"] output from preprocess_single_patient

        assert self.all_columns is not None and self.all_unk_columns is not None, "DataProcessor: Columns not set up correctly"

        #: call converter to convert to DF
        predicted_dataframe = self.current_converter.convert_from_strings_to_df(self.column_name_mapping, 
                                                                                prediction_string, 
                                                                                all_prediction_days, 
                                                                                patientid, patient_sample_index, 
                                                                                self.all_columns, self.all_unk_columns,
                                                                                prediction_days_column_wise=prediction_days_column_wise)


        #: return DF
        return predicted_dataframe
    

    ##################################### TOKENIZATION ##################################################
    
    def encode_input_string(self, str_input):
        str_input_encoded = self.tokenizer(text=str_input, return_tensors="pt", max_length=self.max_total_length, truncation=True, padding="longest")
        return str_input_encoded
    

    def decode_generated_string(self, model_output):
        output_text = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        return output_text
    
    ##################################### POST PROCESSING GENERATED STRING ##################################################


    def post_process_string(self, output_text):

        # Extract second dic (first dic is input dic)        

        def extract_second_top_level_curly_braces(text):
            braces_count = 0
            first_brace_index = None
            second_brace_content = ''
            second_brace_found = False

            for i, char in enumerate(text):
                if char == '{':
                    braces_count += 1
                    if braces_count == 1:
                        first_brace_index = i
                elif char == '}':
                    if braces_count == 1:
                        if second_brace_found:
                            second_brace_content = text[first_brace_index:i+1]
                            break
                        second_brace_found = True
                    braces_count -= 1

            return second_brace_content

        # Example usage:
        result = extract_second_top_level_curly_braces(output_text)

        return result

    ##################################### STANDARDIZING DFS ##################################################
    
    def standardize_numeric_columns(self, df):
        #: standardize all numeric columns
        for col in df.columns:
            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":
                    #: apply standardization
                    df[col] = (df[col] - self.statistics[col]["mean_3_sigma_filtered"]) / self.statistics[col]["std_3_sigma_filtered"]

        return df
    

    def destandardize_df(self, df):
        #: destandardize all numeric columns
        for col in df.columns:
            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":
                    
                    #: first convert to numeric
                    df[col] = pd.to_numeric(df[col])
                    
                    #: apply standardization
                    df[col] = (df[col] * self.statistics[col]["std_3_sigma_filtered"]) + self.statistics[col]["mean_3_sigma_filtered"]

        return df

    ##################################### DATASET LOADER ##################################################

    def get_collator(self, model):

        if self.collator_setting == "causal_lm":
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        elif self.collator_setting == "completion":
            self.data_collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)
            

        return self.data_collator
    
    