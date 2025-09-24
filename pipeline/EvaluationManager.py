import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import json
import glob
import logging
import math
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import os


class EvaluationManager:

    def __init__(self, dataset_name, load_statistics_file=True, base_path=None):
        
        # Public variables
        self.dataset_name = dataset_name
        
        # Dictionary with all available datasets
        if base_path is not None:
            self.base_path = base_path
        else:
            self.base_path = os.path.dirname(__file__).split("/DT-GPT")[0] + "/DT-GPT/"  # Hacky way to get the base path

        self._datasets_available = {
            "2023_11_07_neutrophils" : {
                "path_constant": self.base_path + "1_experiments/2023_11_07_neutrophils/1_data/constant.csv",
                "path_to_events_folder" : self.base_path + "1_experiments/2023_11_07_neutrophils/1_data/patient_events/",
                "path_to_column_mapping_file": self.base_path + "1_experiments/2023_11_07_neutrophils/1_data/column_mapping.json",
                "path_to_dataset_cache": self.base_path + "3_cache",
                "patientid_splits":{
                    "2023_11_08_1k_train": self.base_path + "1_experiments/2023_11_07_neutrophils/1_data/patient_subsets/2023_11_08_1k_train.json",  
                    "2023_11_08_100_validation": self.base_path + "1_experiments/2023_11_07_neutrophils/1_data/patient_subsets/2023_11_08_100_validation.json",  
                    "2023_11_08_100_test": self.base_path + "1_experiments/2023_11_07_neutrophils/1_data/patient_subsets/2023_11_08_100_test.json", 
                },
                "TRAIN": {
                    "input_output_splitting": None,
                    "skip_patient_event_dataset_cache" : True,
                    "path_to_majority_row": "",
                    "path_to_statistics_json": self.base_path + "1_experiments/2023_11_07_neutrophils/1_data/dataset_statistics.json",
                },
                "VALIDATION": {
                    "input_output_splitting": "",
                    "skip_patient_event_dataset_cache" : False,
                    "path_to_high_quality_patient_ids": "",
                },
                "TEST": {
                    "input_output_splitting": "",
                    "skip_patient_event_dataset_cache" : False,
                    "path_to_high_quality_patient_ids": "",
                }
            },
            "2024_02_05_critical_vars" : {
                "path_constant": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/constant.csv",
                "path_to_events_folder" : self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_events/",
                "path_to_column_mapping_file": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/column_mapping.json",
                "path_to_dataset_cache": self.base_path + "3_cache",
                "patientid_splits":{
                    "2023_11_08_1k_train": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2023_11_08_1k_train.json",  
                    "2023_11_08_100_validation": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2023_11_08_100_validation.json",  
                    "2023_11_08_100_test": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2023_11_08_100_test.json", 
                    "2024_06_17_randomized_1_train": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2024_06_17_randomized_1_training.json",
                    "2024_06_17_randomized_1_validation": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2024_06_17_randomized_1_validation.json",
                    "2024_06_17_randomized_1_test": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2024_06_17_randomized_1_test.json",
                    "2024_06_17_randomized_2_train": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2024_06_17_randomized_2_training.json",
                    "2024_06_17_randomized_2_validation": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2024_06_17_randomized_2_validation.json",
                    "2024_06_17_randomized_2_test": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/patient_subsets/2024_06_17_randomized_2_test.json",
                },
                "TRAIN": {
                    "input_output_splitting": None,
                    "skip_patient_event_dataset_cache" : True,
                    "path_to_majority_row": "",
                    "path_to_statistics_json": self.base_path + "1_experiments/2024_02_05_critical_vars/1_data/dataset_statistics.json",
                },
                "VALIDATION": {
                    "input_output_splitting": "",
                    "skip_patient_event_dataset_cache" : False,
                    "path_to_high_quality_patient_ids": "",
                },
                "TEST": {
                    "input_output_splitting": "",
                    "skip_patient_event_dataset_cache" : False,
                    "path_to_high_quality_patient_ids": "",
                }
            },

            "2024_03_15_mimic_iv" : {
                "path_constant": self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/constants.csv",
                "path_to_events_folder" : self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/events/",
                "path_to_column_mapping_file": self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/column_mapping.json",
                "path_to_dataset_cache": self.base_path + "3_cache",
                "patientid_splits": {
                    "2024_03_15_1k_train" : self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_03_15_1k_train.json",
                    "2024_03_15_100_validation" : self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_03_15_100_validation.json",
                    "2024_03_15_100_test": self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_03_15_100_test.json",
                    "2024_06_17_randomized_1_train" : self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_06_17_randomized_1_training.json",
                    "2024_06_17_randomized_1_validation" : self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_06_17_randomized_1_validation.json",
                    "2024_06_17_randomized_1_test": self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_06_17_randomized_1_test.json",
                    "2024_06_17_randomized_2_train" : self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_06_17_randomized_2_training.json",
                    "2024_06_17_randomized_2_validation" : self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_06_17_randomized_2_validation.json",
                    "2024_06_17_randomized_2_test": self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/patient_subsets/2024_06_17_randomized_2_test.json",
                },
                "TRAIN": {
                    "input_output_splitting": None,
                    "skip_patient_event_dataset_cache" : True,
                    "path_to_majority_row": "",
                    "path_to_statistics_json": self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/dataset_statistics.json",
                },
                "VALIDATION": {
                    "input_output_splitting": "",
                    "skip_patient_event_dataset_cache" : False,
                    "path_to_high_quality_patient_ids": "",
                },
                "TEST": {
                    "input_output_splitting": "",
                    "skip_patient_event_dataset_cache" : False,
                    "path_to_high_quality_patient_ids": "",
                }
            },


             "2025_02_03_adni" : {
                "path_constant": self.base_path + "1_experiments/2025_02_03_adni/1_data/1_final_data/constant.csv",
                "path_to_events_folder" : self.base_path + "1_experiments/2025_02_03_adni/1_data/1_final_data/patient_events/",
                "path_to_column_mapping_file": self.base_path + "1_experiments/2025_02_03_adni/1_data/1_final_data/column_mapping.json",  
                "path_to_dataset_cache": self.base_path + "3_cache",
                "patientid_splits": {
                },
                "TRAIN": {
                    "input_output_splitting": None,
                    "skip_patient_event_dataset_cache" : True,
                    "path_to_majority_row": "",
                    "path_to_statistics_json": self.base_path + "1_experiments/2024_02_08_mimic_iv/1_data/0_final_data/dataset_statistics.json", # TODO
                },
                "TEST": {
                    "input_output_splitting": "",
                    "skip_patient_event_dataset_cache" : False,
                    "path_to_high_quality_patient_ids": "",
                }
            },

        }

        # Setup important constants
        self.path_prefix = ""
        self._dataset_splits = None
        self._patient_sample_index_mapping = {}
        self._patient_events_cache = {}
        self._current_dataset_metadata = self._datasets_available[self.dataset_name]  # Current dataset info
        self._all_patient_sample_indices = []

        # Setup constants
        self._load_constants_table()  # Current master.constants table
        self.column_mapping_input_col_names = None
        self.column_mapping_known_future_input_col_names = None
        self.column_mapping_input_target_col_names = None
        with open(self.path_prefix + self._current_dataset_metadata["path_to_column_mapping_file"], "r") as read_file:
            self.column_mapping = json.load(read_file)
        self._setup_columns_usage()
        
        # Setup rest of smaller things
        self._load_rest(load_statistics_file=load_statistics_file)

        # Setup streaming of predictions
        self._eval_streaming = {}
        self._eval_cols_to_assess = None
        self.skip_if_empty_target = False



    ########################################### PATH AND DATASET LOADING FUNCTIONS ################################################################


    def get_dataset_split_names(self):
        assert self._dataset_splits is not None, "EvaluationManager: dataset not initialized yet!"
        return self._dataset_splits

    def get_constant_table(self):
        return self._current_master_constants_table

    def get_paths_to_events_in_split(self, split_name):
    
        #: first get all patient ids for current split using self.get_dataset_split_patientids
        ret_list_patientids = self.get_dataset_split_patientids(split_name)

        #: then get paths - see previous version
        ret_list_paths = []
        for patientid in ret_list_patientids:
            curr_patient_path = self._current_master_constants_table.loc[self._current_master_constants_table["patientid"] == patientid]["path_to_events_file"].tolist()[0]
            ret_list_paths.append(self.path_prefix + self._datasets_available[self.dataset_name]["path_to_events_folder"] + curr_patient_path)

        return ret_list_paths, ret_list_patientids
    

    ########################################### CACHE MANAGER ################################################################

    def check_exists_in_cache(self, name, return_path=False):
        ret_list = glob.glob(self.path_prefix + self._current_dataset_metadata["path_to_dataset_cache"] + name + ".*")
        if len(ret_list) > 0:
            if return_path:
                return True, ret_list
            else:
                return True
        else:
            if return_path:
                return False, []
            else:
                return False


    def load_from_cache(self, name, mode):
        # Modes implemented: json
        assert mode in ["json"], "EvaluationManager: Cache Manager: unknown mode"

        if mode == "json":
            path_to_file = self.path_prefix + self._current_dataset_metadata["path_to_dataset_cache"] + name + ".json"
            with open(path_to_file, "r") as read_file:
                ret_file = json.load(read_file)
        
        return ret_file


    def save_to_cache(self, name, object_to_save, mode):
        # Modes implemented: json
        assert mode in ["json"], "EvaluationManager: Cache Manager: unknown mode"

        if mode == "json":   
            # save as json
            path_to_file = self.path_prefix + self._current_dataset_metadata["path_to_dataset_cache"] + name + ".json"
            with open(path_to_file, 'w') as f:
                json.dump(object_to_save, f, indent=4, default=str)
            return path_to_file
    
    def get_path_to_cache(self):
        return self.path_prefix + self._current_dataset_metadata["path_to_dataset_cache"]


    ########################################### PATIENT SPLITS ################################################################

    def get_dataset_split_patientids(self, split_name):

        # First check in normal splitting        
        if split_name in self._current_master_constants_table["dataset_split"].unique().tolist():
            # Get patientids in given split
            patientids_split = self._current_master_constants_table.loc[self._current_master_constants_table["dataset_split"] == split_name]["patientid"]
            return patientids_split
        
        # Next check check in patientid_splits
        if split_name in self._datasets_available[self.dataset_name]["patientid_splits"]:
            
            #: load in as json
            with open(self._datasets_available[self.dataset_name]["patientid_splits"][split_name], "r") as read_file:
                dic_with_patientids = json.load(read_file)
            
            return dic_with_patientids["patientids"]

        raise Exception("Eval Manager: split not found!")
    

    def get_events_table(self, patientid):
        
        # Then check if data already in cache
        if patientid in self._patient_events_cache:
            return self._patient_events_cache[patientid]

        # load data
        patient_path = self._current_master_constants_table.loc[self._current_master_constants_table["patientid"] == patientid]["path_to_events_file"].tolist()[0]
        path = self.path_prefix + self._current_dataset_metadata["path_to_events_folder"] + str(patient_path)
        patient_events_table = pd.read_csv(path)

        # convert date to date
        patient_events_table['date'] = pd.to_datetime(patient_events_table['date'])

        # save to cache
        self._patient_events_cache[patientid] = patient_events_table

        return patient_events_table


    def get_full_patient_info(self, patientid):
        
        # return constant and full info
        constants_row = self._current_master_constants_table.loc[self._current_master_constants_table["patientid"] == patientid]
        events_table = self.get_events_table(patientid)

        return constants_row, events_table
    
    def load_list_of_patient_dfs_and_constants(self, list_of_patient_ids):

        ret_list_constants = []
        ret_list_dfs = []

        for idx, patientid in enumerate(list_of_patient_ids):

            if idx % 100 == 0:
                logging.info("Loading patient: " + str(idx) + " / " + str(len(list_of_patient_ids)))

            curr_const, curr_events = self.get_full_patient_info(patientid)

            ret_list_constants.append(curr_const)
            ret_list_dfs.append(curr_events)

        return ret_list_constants, ret_list_dfs

    def get_column_usage(self):
        # Use to get list of col names: input, known_future_inputs, targets
        return self.column_mapping_input_col_names, self.column_mapping_known_future_input_col_names, self.column_mapping_input_target_col_names


    def make_empty_df(self, df):
        
        empty_target_dataframe = df.copy()
        empty_target_dataframe.loc[:, [col for col in empty_target_dataframe.columns if col not in ["date", "patientid", "patient_sample_index"]]] = np.nan

        return empty_target_dataframe
        





    ########################################### STREAMING EVALUATION FUNCTIONS - ASSUMING NEXT VISIT PREDICTION ################################################################
    
    def evaluate_split_stream_start(self, cols_to_asses=None, skip_if_empty_target=True):
        self._eval_streaming = {}
        self._eval_cols_to_assess = cols_to_asses
        self.skip_if_empty_target = skip_if_empty_target


    def evaluate_split_stream_prediction(self, processed_prediction, target_df, patientid, patient_sample_index):

        events_output = target_df

        #: convert to string for more stable metric calculation whilst retaining NAs
        processed_prediction_nas = pd.isnull(processed_prediction)
        events_output_nas = pd.isnull(events_output)

        #: sort by date to ensure consistent order
        processed_prediction = processed_prediction.sort_values(by=['date'])
        events_output = events_output.sort_values(by=['date'])

        #: convert correctly each column
        for col in events_output.columns: 
            if col in self.column_statistics and self.column_statistics[col]["type"] == "numeric":
                # Convert to float
                events_output[col] = pd.to_numeric(events_output[col])
                processed_prediction[col] = pd.to_numeric(processed_prediction[col])
            else:
                # Convert to str
                events_output[col] = events_output[col].astype(str)
                processed_prediction[col] = processed_prediction[col].astype(str)

        #: ensure correct column order
        processed_prediction = processed_prediction[events_output.columns]

        # Process nans
        processed_prediction[processed_prediction_nas] = np.nan
        events_output[events_output_nas] = np.nan
        
        # add the prediction and corresponding targets to the list
        if patientid not in self._eval_streaming:
            self._eval_streaming[patientid] = {}

        self._eval_streaming[patientid][patient_sample_index] = {
            "prediction" : processed_prediction,
            "target": events_output,
        }


    def concat_eval(self):
        
        #: get evaluation of all predictions
        full_evaluation = self._eval_streaming

        # calculate samples in list

        #: combine all rows into single dataframe - generate multiindex for later use
        full_df_prediction = pd.concat([full_evaluation[patientid][patient_sample_index]["prediction"] for patientid in full_evaluation.keys() for patient_sample_index in full_evaluation[patientid].keys()], ignore_index=True)
        new_index_prediction = pd.MultiIndex.from_frame(full_df_prediction.loc[:, ["patientid", "patient_sample_index"]])
        full_df_prediction = full_df_prediction.set_index(new_index_prediction)  # Can now access it with full_df[(patientid, patient_sample_index)]

        #: make corresponding target dataframe
        full_df_targets = pd.concat([full_evaluation[patientid][patient_sample_index]["target"] for patientid in full_evaluation.keys() for patient_sample_index in full_evaluation[patientid].keys()], ignore_index=True)
        new_index_targets = pd.MultiIndex.from_frame(full_df_targets.loc[:, ["patientid", "patient_sample_index"]])
        full_df_targets = full_df_targets.set_index(new_index_targets)  # Can now access it with full_df[(patientid, patient_sample_index)]

        # Logging
        logging.info("Full number of samples to evaluate in target: " + str(full_df_targets.shape[0]))
        logging.info("Full number of samples to evaluate in prediction: " + str(full_df_prediction.shape[0]))
        assert full_df_targets.shape[0] == full_df_prediction.shape[0], "Eval Manager: df target and prediction not same length!"
        assert full_df_targets[["patientid", "patient_sample_index"]].values.tolist() == full_df_prediction[["patientid", "patient_sample_index"]].values.tolist(), "Eval Manager: df target and prediction not same patientid and patient_sample_index!"


        return full_df_targets, full_df_prediction




    ########################################### PRIVATE HELPER FUNCTIONS ################################################################

    
    def _load_constants_table(self):

        # Load in main table
        self._current_master_constants_table = pd.read_csv(self.path_prefix + self._current_dataset_metadata["path_constant"])


    def _load_rest(self, load_statistics_file=True):
        # Set some further constants
        self._dataset_splits = self._current_master_constants_table["dataset_split"].unique()

        # load statistics
        if load_statistics_file:
            with open(self._datasets_available[self.dataset_name]["TRAIN"]["path_to_statistics_json"]) as f:
                self.column_statistics = json.load(f)


   
    def _setup_columns_usage(self):
        
        input_col_names = []
        known_future_input_col_names = []
        target_col_names = []
     
        for current_col_name in self.column_mapping.keys():

            if self.column_mapping[current_col_name]["input"]:
                input_col_names.append(current_col_name)

            if self.column_mapping[current_col_name]["known_future_input"]:
                known_future_input_col_names.append(current_col_name)

            if self.column_mapping[current_col_name]["target"]:
                target_col_names.append(current_col_name)

        self.column_mapping_input_col_names = input_col_names
        self.column_mapping_known_future_input_col_names = known_future_input_col_names
        self.column_mapping_input_target_col_names = target_col_names 

        # do assertions that is ok
        assert set(self.column_mapping_known_future_input_col_names).isdisjoint(self.column_mapping_input_target_col_names), "Evaluation Manager: known future inputs and target cols intersect!"



