import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
from pandas.api.types import CategoricalDtype
import logging
import json
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



class MetricManager:

    def __init__(self, path_to_statistics_file, debug_mode=False):

        # Column statistics
        with open(path_to_statistics_file) as f:
            self.column_statistics = json.load(f)
            
        # Some constants
        self.categorical_metric_list = {"f1_macro": self.f1_macro, "accuracy": self.accuracy}
        self.numeric_metric_list = {"r2" : self.r2, "mae" : self.mae, "rmse" : self.rmse, 
                                    "spearman_corr" : self.spearman_correlation, "nrmse" : self.nrmse,
                                    "dir_accuracy" : self.directional_accuracy}
        self.metric_list_requiring_ids_cols = ["dir_accuracy"]

        self.skip_columns = ["patientid", "patientid.1", "patient_sample_index.1", "patient_sample_index", "date", "X", "X.1", "X.2", "X.3", "X.4", "X.5", "X.6"]

        self.debug_mode = debug_mode


    def calculate_metrics(self, target_df, prediction_df, group_by=None):

        # Make checks
        assert target_df.shape[0] == prediction_df.shape[0], "MetricManager: nr of samples different between target_df and prediction_df"
        
        # Convert to numpy
        targets = target_df
        predictions = prediction_df

        # Setup counters
        results = {}
        cat_columns = []
        numeric_columns = []
        
        #: get all distinct groups
        if group_by is None:
            all_groups = []
        else:
            assert len(group_by) == prediction_df.shape[0], "MetricManager: size of group_by list and prediction_df don't match up!"
            all_groups = list(set(group_by))
        all_groups = sorted(all_groups)  # Sort for ease of searching later
        all_groups.append("overall")

        # Go over all columns
        for i in range(targets.shape[1]):

            # Skip poor columns
            if targets.columns[i] in self.skip_columns:
                continue
            
            # Select current column target and predictions
            curr_col = target_df.columns[i]
            current_col_targets = targets[curr_col]
            current_col_predictions = predictions[curr_col]
            curr_col_ids = targets[["patientid", "patient_sample_index", "date"]]

            # Check if we skip
            if curr_col in self.skip_columns:
                continue

            #: extract non-nas
            current_col_targets_non_na = pd.isnull(current_col_targets)
            current_col_targets = current_col_targets[~current_col_targets_non_na]
            current_col_predictions = current_col_predictions[~current_col_targets_non_na]
            curr_col_ids = curr_col_ids[~current_col_targets_non_na]

            # Setup indices of groups
            if group_by is not None:
                group_indices = { group: [i for i, x in enumerate(np.asarray(group_by)[~current_col_targets_non_na]) if x == group] 
                                            if group != "overall" else list(range(len(np.asarray(group_by)[~current_col_targets_non_na])))
                                            for group in all_groups}
            else:
                group_indices = {"overall" : list(range(current_col_targets.shape[0]))}

            # setup results
            results[curr_col] = {}
            results[curr_col]["nr_samples"] = {}

            # Skip in case empty targets
            if current_col_targets.shape[0] == 0:
                continue
            
            #: get correct type
            if self.column_statistics[curr_col]["type"] == "numeric":

                numeric_columns.append(curr_col)
                
                #: run across all numeric metrics and calculate
                for numerical_metric_name in self.numeric_metric_list.keys():

                    results[curr_col][numerical_metric_name] = {}

                    for group in all_groups:

                        #: select those rows which are in the group
                        indices_to_select = group_indices[group]
                        
                        #: select from DF
                        group_current_col_targets = current_col_targets.iloc[indices_to_select]
                        group_current_col_predictions = current_col_predictions.iloc[indices_to_select]
                        group_current_col_ids = curr_col_ids.iloc[indices_to_select]

                        if group_current_col_targets.shape[0] == 0:

                            #: in case of empty dataframe, elegantly skip
                            results[curr_col][numerical_metric_name][group] = np.nan
                            results[curr_col]["nr_samples"][group] = 0
                        
                        else:

                            if numerical_metric_name in self.metric_list_requiring_ids_cols:
                                #: eval per category, and pass the ids as well
                                results[curr_col][numerical_metric_name][group] = self.numeric_metric_list[numerical_metric_name](group_current_col_targets, group_current_col_predictions, group_current_col_ids)
                                results[curr_col]["nr_samples"][group] = group_current_col_targets.shape[0]
                            
                            else:
                                #: eval per category
                                results[curr_col][numerical_metric_name][group] = self.numeric_metric_list[numerical_metric_name](group_current_col_targets, group_current_col_predictions)
                                results[curr_col]["nr_samples"][group] = group_current_col_targets.shape[0]

            else:
                # Go over categorical measures
                cat_columns.append(curr_col)

                # Convert to strings to ensure no problems in calculation later by scikit
                current_col_targets = current_col_targets.astype(str)
                current_col_predictions = current_col_predictions.astype(str)

                #: run across all categorical metrics and calculate
                for cat_metric_name in self.categorical_metric_list.keys():

                    results[curr_col][cat_metric_name] = {}
                    
                    for group in all_groups:
                        #: select those rows which are in the group
                        indices_to_select = group_indices[group]
                        
                        #: select from DF
                        group_current_col_targets = current_col_targets.iloc[indices_to_select]
                        group_current_col_predictions = current_col_predictions.iloc[indices_to_select]

                        # : eval per category
                        results[curr_col][cat_metric_name][group] = self.categorical_metric_list[cat_metric_name](group_current_col_targets, group_current_col_predictions)
                        results[curr_col]["nr_samples"][group] = group_current_col_targets.shape[0]




        #: calculate overall averages for categorical
        avg_cat_results = {x : {curr_group : 0  for curr_group in all_groups} for x in self.categorical_metric_list.keys()}
        avg_cat_counts = {x : {curr_group : 0  for curr_group in all_groups} for x in self.categorical_metric_list.keys()}

        for cat_col in cat_columns:

            #: eval per category

            for cat_metric in results[cat_col].keys():
                if cat_metric == "nr_samples":
                    continue
                
                for curr_group in results[cat_col][cat_metric].keys():

                    if not np.isnan(results[cat_col][cat_metric][curr_group]):
                        avg_cat_results[cat_metric][curr_group] += results[cat_col][cat_metric][curr_group]
                        avg_cat_counts[cat_metric][curr_group] += 1

        avg_cat_results = {x : {curr_group : avg_cat_results[x][curr_group] / avg_cat_counts[x][curr_group] if avg_cat_counts[x][curr_group] > 0 else np.nan for curr_group in all_groups} for x in self.categorical_metric_list.keys()}
        results["all_categorical_columns"] = avg_cat_results

        #: calculate overall averages for numeric
        avg_numeric_results = {x : {curr_group : 0  for curr_group in all_groups} for x in self.numeric_metric_list.keys()}
        avg_numeric_counts = {x : {curr_group : 0  for curr_group in all_groups} for x in self.numeric_metric_list.keys()}

        for num_col in numeric_columns:

            #: eval per category

            for num_metric in results[num_col].keys():
                if num_metric == "nr_samples":
                    continue
                
                for curr_group in results[num_col][num_metric].keys():

                    if not np.isnan(results[num_col][num_metric][curr_group]):
                        avg_numeric_results[num_metric][curr_group] += results[num_col][num_metric][curr_group]
                        avg_numeric_counts[num_metric][curr_group] += 1

        avg_numeric_results = {x : {curr_group : avg_numeric_results[x][curr_group] / avg_numeric_counts[x][curr_group] if avg_numeric_counts[x][curr_group] > 0 else np.nan for curr_group in all_groups} for x in self.numeric_metric_list.keys()}
        results["all_numeric_columns"] = avg_numeric_results
        

        return results



    def f1_macro(self, current_col_targets, current_col_predictions):
        new_f1 = f1_score(current_col_targets, current_col_predictions, average="macro")
        return new_f1
    

    def accuracy(self, current_col_targets, current_col_predictions):
        # Calculate acc score with macro averaging
        new_acc = accuracy_score(current_col_targets, current_col_predictions)
        return new_acc
    

    def r2(self, current_col_targets, current_col_predictions):
        r2 = r2_score(current_col_targets, current_col_predictions)
        return r2

    
    def mae(self, current_col_targets, current_col_predictions):
        mae = mean_absolute_error(current_col_targets, current_col_predictions)
        return mae
    
    def rmse(self, current_col_targets, current_col_predictions):
        rmse = mean_squared_error(current_col_targets, current_col_predictions, squared=False)
        return rmse

    def nrmse(self, current_col_targets, current_col_predictions):
        rmse = mean_squared_error(current_col_targets, current_col_predictions, squared=False)
        nrmse = rmse / np.std(current_col_targets)  # Normalized RMSE
        return nrmse
    
    def spearman_correlation(self, current_col_targets, current_col_predictions):
        res = stats.spearmanr(current_col_targets, current_col_predictions)
        r = res.correlation
        return r
    
    def directional_accuracy(self, current_col_targets, current_col_predictions, current_col_ids):

        # Combine the inputs into a single DataFrame
        data = pd.DataFrame({
            'targets': current_col_targets,
            'predictions': current_col_predictions,
            'patientid': current_col_ids['patientid'],
            'patient_sample_index': current_col_ids['patient_sample_index'],
            'date': current_col_ids['date']
        })
        data = data.reset_index(drop=True)

        # Sort by 'patientid', 'patient_sample_index', and 'date'
        data.sort_values(by=['patientid', 'patient_sample_index', 'date'], inplace=True)

        # Calculate the directional change for targets and predictions
        data['target_change'] = data.groupby(['patientid', 'patient_sample_index'])['targets'].diff()
        data['prediction_change'] = data.groupby(['patientid', 'patient_sample_index'])['predictions'].diff()

        # Define a function to determine if the direction of change matches
        def direction_matches(row):
            # If both changes are NaN (no previous data), we cannot determine direction, so return NaN
            if pd.isna(row['target_change']) or pd.isna(row['prediction_change']):
                return None
            # If both changes have the same sign, the direction matches
            return np.sign(row['target_change']) == np.sign(row['prediction_change'])

        # Apply the function to determine matching directions
        data['direction_match'] = data.apply(direction_matches, axis=1)

        # Calculate directional accuracy
        # We drop NaN values because they represent cases where direction could not be determined
        directional_accuracy = data['direction_match'].dropna().astype(int).mean()

        return directional_accuracy










