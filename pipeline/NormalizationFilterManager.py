import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
from pandas.api.types import CategoricalDtype
import logging
from abc import ABC, abstractmethod
import json
import wandb




class NormalizeAndFilterBase(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def normalize_and_filter(self, target_dataframe, prediction_dataframe):
        # Each target and prediction is DF with rows of patientid, patient_sample_index, date, vars
        # Note: can be multiple rows for each patient_sample_index, ordered by date

        # Return: target_dataframe, prediction_dataframe in same general format but with bad rows filtered out
        pass

    @abstractmethod
    def get_name(self):
        # Return string name of method
        pass

    @abstractmethod
    def denormalize(self, target_dataframe, prediction_dataframe):
        # Denormalize both dataframes back into original space
        pass

    


def replace_nan_rows_with_majority_mean(df, statistics, verbose=True):

    skip_columns = ["patientid", "patient_sample_index", "date", "X", "X.1", "X.2", "X.3", "X.4", "X.5", "X.6"]

    #: make median/majority row
    df_dic = {}
    for col in statistics.keys():
        if col in df.columns:
            if statistics[col]["type"] == "numeric":
                df_dic[col] = [statistics[col]["median"]]
            else:
                df_dic[col] = [max(statistics[col]["counts"], key=statistics[col]["counts"].get)]
    df_mean = pd.DataFrame(df_dic)
    
    #: find all nan rows
    df_all_nas = df.drop(skip_columns, axis=1, errors="ignore")
    df_all_nas_select = df_all_nas.isnull().all(axis=1)
    df_all_nas_select = np.flatnonzero(df_all_nas_select)

    #: fill in missing columns in median
    cols_to_use = [col for col in df.columns if col in df_mean.columns]
    cols_to_use_index = [df.columns.get_loc(col) for col in df.columns if col in df_mean.columns]
    df_mean = df_mean[cols_to_use]

    #: replace nan rows
    df.iloc[df_all_nas_select, cols_to_use_index] = df_mean

    if verbose:
        logging.info("Nr of rows only with nans replaced with median: " + str(df_all_nas_select.sum()))

    return df



def replace_nan_columns_in_prediction_df_with_median(target_df, prediction_df, statistics, verbose=True):

    data_rec = {}

    #: replace for every column where nan in prediction but value in target with median/majority class
    for col in prediction_df.columns:

        if col in statistics.keys():

            #: extract non-nas
            current_col_targets_non_na = ~pd.isnull(target_df[col])
            current_col_pred_na = pd.isnull(prediction_df[col])

            preds_to_change = current_col_targets_non_na & current_col_pred_na

            data_rec[col] = preds_to_change.sum()

            if statistics[col]["type"] == "numeric":
                prediction_df.loc[preds_to_change, col] = statistics[col]["median"]
            else:
                prediction_df.loc[preds_to_change, col] = max(statistics[col]["counts"], key=statistics[col]["counts"].get)

    if verbose:
        #: log to wandb nr of missing and to console
        logging.info("Nr of data changed to median for nan column in prediction but not in target: " + str(data_rec))

    return prediction_df



class MetaFilter(NormalizeAndFilterBase):
    """A meta filter class to select different filters for different columns"""

    def __init__(self, filter_to_column_dictionary):
        # filter_to_column_dictionary - should be of the format {<FilterInstance> : [<list of columns to apply to>]}
        # e.g. {Only_3_sigma_Filtering() : ["col1", "col2"], Only_3_sigma_Filtering() : ["col3", "col4"]}

        super().__init__()

        self.filter_to_column_dictionary = filter_to_column_dictionary

    
    def normalize_and_filter(self, target_dataframe, prediction_dataframe):

        # Setup return instances
        ret_target = []
        ret_prediction = []

        #: go through every filter
        for filter_instance, columns_to_apply_to in self.filter_to_column_dictionary.items():
            
            # Select columns to apply to
            temp_target = target_dataframe[columns_to_apply_to]
            temp_prediction = prediction_dataframe[columns_to_apply_to]
            temp_filtered_target, temp_filtered_prediction = filter_instance.normalize_and_filter(temp_target, temp_prediction)
            ret_target.append(temp_filtered_target)
            ret_prediction.append(temp_filtered_prediction)

        # concat target and prediction dataframes        
        target_dataframe = pd.concat(ret_target, axis=1)
        prediction_dataframe = pd.concat(ret_prediction, axis=1)
        
        return target_dataframe, prediction_dataframe


    def get_name(self):
        return "MetaFilter"
    
    def denormalize(self, target_dataframe, prediction_dataframe):
        raise NotImplementedError("Denormalization not implemented for this normalizer & filter")




class Only_Double3_sigma_Filtering(NormalizeAndFilterBase):
    "Applied 2x 3 sigma rule, clipping the outputs"

    def __init__(self, path_to_statistics_file):
        self.path_to_statistics_file = path_to_statistics_file

        #: load column statistics from file
        with open(self.path_to_statistics_file) as f:
            self.statistics = json.load(f)


    def normalize_and_filter(self, target_dataframe, prediction_dataframe, replace_nan_rows=True, replace_missing_in_prediction=True, verbose=True, specific_column_list=None):
        
        #: replace nan rows with meam/majority class row
        if replace_nan_rows and prediction_dataframe is not None:
            prediction_dataframe = replace_nan_rows_with_majority_mean(prediction_dataframe.copy(), self.statistics, verbose=verbose)

        # Fill in rest of missing in prediction
        if replace_missing_in_prediction and prediction_dataframe is not None:
            prediction_dataframe = replace_nan_columns_in_prediction_df_with_median(target_dataframe, prediction_dataframe, statistics=self.statistics, verbose=verbose)

        if verbose:
            logging.info("Filtering with 3 Sigma Rule")
        
        #: go through every target column
        for col in target_dataframe.columns:

            # In case sprecific_column_list is given, only apply to those columns
            if specific_column_list is not None:
                if col not in specific_column_list:
                    continue

            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":
                    
                    #: calculate 3 sigma
                    lower_bound = self.statistics[col]["mean"] - (3 * self.statistics[col]["std"])
                    upper_bound = self.statistics[col]["mean"] + (3 * self.statistics[col]["std"])

                    #: turn all target values out of bounds to NAs
                    rows_to_select = (target_dataframe[col] > upper_bound) | (target_dataframe[col] < lower_bound)
                    if verbose and rows_to_select.sum() > 0:
                        logging.info("For column: " + str(col) + " filtering out # rows: " + str(rows_to_select.sum()))
                    target_dataframe.loc[rows_to_select, col] = np.nan
                    if prediction_dataframe is not None:
                        prediction_dataframe.loc[rows_to_select, col] = np.nan

                    # Now in second round clip to cleaned 3 sigma
                    lower_bound = self.statistics[col]["mean_3_sigma_filtered"] - (3 * self.statistics[col]["std_3_sigma_filtered"])
                    upper_bound = self.statistics[col]["mean_3_sigma_filtered"] + (3 * self.statistics[col]["std_3_sigma_filtered"])

                    #: clip values to the bounds
                    target_dataframe[col] = target_dataframe[col].clip(lower=lower_bound, upper=upper_bound)
                    if prediction_dataframe is not None:
                        prediction_dataframe[col] = prediction_dataframe[col].clip(lower=lower_bound, upper=upper_bound)

                    #: turn all values out of bounds to mean
                    rows_clipped_target = ((target_dataframe[col] == lower_bound) | (target_dataframe[col] == upper_bound)).sum()
                    if prediction_dataframe is not None:
                        rows_clipped_predic = ((prediction_dataframe[col] == lower_bound) | (prediction_dataframe[col] == upper_bound)).sum()
                    
                    if verbose:
                        if prediction_dataframe is not None and rows_clipped_target > 0:
                            logging.info("For column: " + str(col) + " clipping # still too large targets: " + str(rows_clipped_target) + " and # cliping predictions: " + str(rows_clipped_predic))
                        else:
                            if rows_clipped_target > 0:
                                logging.info("For column: " + str(col) + " clipping # still too large targets: " + str(rows_clipped_target))


                # keep categorical as is

        
        return target_dataframe, prediction_dataframe

    
    def get_name(self):
        return "only_double_3_sigma"

    def denormalize(self, target_dataframe, prediction_dataframe):
        raise NotImplementedError("Denormalization not implemented for this normalizer & filter")







class Double3_sigma_Filtering_And_Standardization(NormalizeAndFilterBase):
    "Applied 2x 3 sigma rule, clipping the outputs, then applying training data based standardization"

    def __init__(self, path_to_statistics_file):
        self.path_to_statistics_file = path_to_statistics_file
        self.only_double3_sigma_filter = Only_Double3_sigma_Filtering(path_to_statistics_file)

        #: load column statistics from file
        with open(self.path_to_statistics_file) as f:
            self.statistics = json.load(f)
    

    def normalize_and_filter(self, target_dataframe, prediction_dataframe, **kwargs):

        if "verbose" in kwargs and kwargs["verbose"]:
            logging.info("Applying Double 3 Sigma Rule and Standardization")
        

        # First apply 3 sigma filtering
        target_dataframe, prediction_dataframe = self.only_double3_sigma_filter.normalize_and_filter(target_dataframe, prediction_dataframe, **kwargs)

        # Then apply standardization
        for col in target_dataframe.columns:

            # In case sprecific_column_list is given, only apply to those columns
            if "specific_column_list" in kwargs.keys():
                if col not in kwargs["specific_column_list"]:
                    continue

            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":
                    
                    #: apply standardization
                    target_dataframe[col] = (target_dataframe[col] - self.statistics[col]["mean_double_3_sigma_filtered"]) / self.statistics[col]["std_double_3_sigma_filtered"]

                    if prediction_dataframe is not None:
                        prediction_dataframe[col] = (prediction_dataframe[col] - self.statistics[col]["mean_double_3_sigma_filtered"]) / self.statistics[col]["std_double_3_sigma_filtered"]

                # keep categorical as is
        

        return target_dataframe, prediction_dataframe
    

    def denormalize(self, target_dataframe, prediction_dataframe, meta_data=None):

        target_dataframe = target_dataframe.copy()
        prediction_dataframe = prediction_dataframe.copy()
        
        for col in target_dataframe.columns:

            if col in self.statistics.keys():
                if self.statistics[col]["type"] == "numeric":

                    target_dataframe[col] = (target_dataframe[col] * self.statistics[col]["std_double_3_sigma_filtered"]) + self.statistics[col]["mean_double_3_sigma_filtered"]
                    
                    if prediction_dataframe is not None:
                        prediction_dataframe[col] = (prediction_dataframe[col] * self.statistics[col]["std_double_3_sigma_filtered"]) + self.statistics[col]["mean_double_3_sigma_filtered"]
                    
                    if meta_data is not None:
                        meta_data[col] = (meta_data[col] * self.statistics[col]["std_double_3_sigma_filtered"]) + self.statistics[col]["mean_double_3_sigma_filtered"]
        
        if meta_data is not None:
            return target_dataframe, prediction_dataframe, meta_data
        else:
            return target_dataframe, prediction_dataframe

    
    def get_name(self):
        return "double_3_sigma_then_standardization"






class Only_Standardization(NormalizeAndFilterBase):
    "Only applying training data based standardization"

    def __init__(self, path_to_statistics_file):
        self.path_to_statistics_file = path_to_statistics_file
        

        #: load column statistics from file
        with open(self.path_to_statistics_file) as f:
            self.statistics = json.load(f)
    

    def normalize_and_filter(self, target_dataframe, prediction_dataframe, replace_nan_rows=True, replace_missing_in_prediction=True, **kwargs):

        if "verbose" in kwargs:
            verbose = kwargs["verbose"]

            if verbose:
                logging.info("Applying only Standardization")
        else:
            verbose = False


        #: replace nan rows with meam/majority class row
        if replace_nan_rows and prediction_dataframe is not None:
            prediction_dataframe = replace_nan_rows_with_majority_mean(prediction_dataframe.copy(), self.statistics, verbose=verbose)

        # Fill in rest of missing in prediction
        if replace_missing_in_prediction and prediction_dataframe is not None:
            prediction_dataframe = replace_nan_columns_in_prediction_df_with_median(target_dataframe, prediction_dataframe, statistics=self.statistics, verbose=verbose)
        
        # Then apply standardization
        for col in target_dataframe.columns:

            # In case sprecific_column_list is given, only apply to those columns
            if "specific_column_list" in kwargs.keys():
                if col not in kwargs["specific_column_list"]:
                    continue

            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":
                    
                    #: apply standardization
                    target_dataframe[col] = (target_dataframe[col] - self.statistics[col]["mean_double_3_sigma_filtered"]) / self.statistics[col]["std_double_3_sigma_filtered"]

                    if prediction_dataframe is not None:
                        prediction_dataframe[col] = (prediction_dataframe[col] - self.statistics[col]["mean_double_3_sigma_filtered"]) / self.statistics[col]["std_double_3_sigma_filtered"]

                # keep categorical as is
        
        return target_dataframe, prediction_dataframe
    

    def denormalize(self, target_dataframe, prediction_dataframe, meta_data=None):

        target_dataframe = target_dataframe.copy()
        prediction_dataframe = prediction_dataframe.copy()
        
        for col in target_dataframe.columns:

            if col in self.statistics.keys():
                if self.statistics[col]["type"] == "numeric":

                    target_dataframe[col] = (target_dataframe[col] * self.statistics[col]["std_double_3_sigma_filtered"]) + self.statistics[col]["mean_double_3_sigma_filtered"]
                    
                    if prediction_dataframe is not None:
                        prediction_dataframe[col] = (prediction_dataframe[col] * self.statistics[col]["std_double_3_sigma_filtered"]) + self.statistics[col]["mean_double_3_sigma_filtered"]
                    
                    if meta_data is not None:
                        meta_data[col] = (meta_data[col] * self.statistics[col]["std_double_3_sigma_filtered"]) + self.statistics[col]["mean_double_3_sigma_filtered"]
        
        if meta_data is not None:
            return target_dataframe, prediction_dataframe, meta_data
        else:
            return target_dataframe, prediction_dataframe

    
    def get_name(self):
        return "only_standardization"















