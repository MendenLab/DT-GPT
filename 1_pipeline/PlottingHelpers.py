from plotnine import *
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pandas import Timestamp
import matplotlib.image as img
import os
import random
plt.set_loglevel(level = 'warning')
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
                                  


class PlotHelper:

    def __init__(self, dataset_statistics_json_path, column_descriptive_mapping_path) -> None:

        plt.set_loglevel("error")

        with open(dataset_statistics_json_path) as f:
            self.statistics = json.load(f)

        self.descriptive_mapping_df = pd.read_csv(column_descriptive_mapping_path)
        self.descriptive_mapping = {row[1]["original_column_names"] : row[1]["descriptive_column_name"] for row in self.descriptive_mapping_df.iterrows()}


    def convert_column_to_descriptive(self, column):

        col_list = column.tolist()

        ret_list = []

        for col in col_list:
            if col in self.descriptive_mapping:
                ret_list.append(self.descriptive_mapping[col])
            else:
                ret_list.append(col)

        return ret_list

    

    def scatter_plot_all_numeric_values(self, predicted_df, target_df):
        
        flattened_dfs_target = []
        flattened_dfs_predicted = []

        #: get all numeric columns and their respective patient_sample_index into one column
        for col in target_df.columns:

            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":

                    curr_target = target_df.loc[~pd.isnull(target_df[col])]
                    curr_pred = predicted_df.loc[~pd.isnull(target_df[col])]

                    flattened_dfs_target.append(curr_target.rename(columns={col : "values_target"})[["patient_sample_index", "values_target"]])
                    flattened_dfs_predicted.append(curr_pred.rename(columns={col : "values_pred"})[["patient_sample_index", "values_pred"]])


        flattened_dfs_target = pd.concat(flattened_dfs_target, axis=0)
        flattened_dfs_predicted = pd.concat(flattened_dfs_predicted, axis=0)

        flattened_dfs_target = flattened_dfs_target.rename(columns={"patient_sample_index" : "patient_sample_index_targ"})
        flattened_dfs_predicted = flattened_dfs_predicted.rename(columns={"patient_sample_index" : "patient_sample_index_pred"})

        flattened_dfs = pd.concat([flattened_dfs_target, flattened_dfs_predicted], axis=1)

        #: plot
        p = ggplot(aes(x='values_target', y='values_pred'), flattened_dfs)   # Basics
        p = p + geom_point(aes(color='factor(patient_sample_index_targ)'), size=0.1)   # Scatter plot
        p = p + geom_abline(intercept=0, slope=1)   # Diagonal line


        #: return
        return p


    def scatter_plot_all_numeric_values_by_column(self, predicted_df, target_df):
        
        flattened_dfs_target = []
        flattened_dfs_predicted = []

        #: get all numeric columns and their respective patient_sample_index into one column
        for col in target_df.columns:

            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":

                    curr_target = target_df.loc[~pd.isnull(target_df[col])].copy()
                    curr_pred = predicted_df.loc[~pd.isnull(target_df[col])].copy()

                    curr_target["variable_targ"] = col
                    curr_pred["variable_pred"] = col

                    flattened_dfs_target.append(curr_target.rename(columns={col : "values_target"})[["variable_targ", "values_target"]])
                    flattened_dfs_predicted.append(curr_pred.rename(columns={col : "values_pred"})[["variable_pred", "values_pred"]])


        flattened_dfs_target = pd.concat(flattened_dfs_target, axis=0)
        flattened_dfs_predicted = pd.concat(flattened_dfs_predicted, axis=0)


        flattened_dfs = pd.concat([flattened_dfs_target, flattened_dfs_predicted], axis=1)
        flattened_dfs["variable_targ"] = self.convert_column_to_descriptive(flattened_dfs["variable_targ"])

        #: plot
        p = ggplot(aes(x='values_target', y='values_pred'), flattened_dfs)   # Basics
        p = p + geom_point(aes(color='factor(variable_targ)'), size=0.1)   # Scatter plot
        p = p + geom_abline(intercept=0, slope=1)   # Diagonal line


        #: return
        return p
    
    def scatter_plot_all_numeric_values_by_relative_day(self, predicted_df, target_df):
        
        flattened_dfs_target = []
        flattened_dfs_predicted = []

        # Sort by date
        predicted_df['date'] = pd.to_datetime(predicted_df['date'])
        target_df['date'] = pd.to_datetime(target_df['date'])


        #: get all numeric columns and their respective patient_sample_index into one column
        for col in target_df.columns:

            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":


                    curr_target = target_df.loc[~pd.isnull(target_df[col])].copy()
                    curr_pred = predicted_df.loc[~pd.isnull(target_df[col])].copy()

                    curr_target = curr_target.sort_values(by=['date'])
                    curr_target["day_targ"] = curr_target.groupby(['patientid', 'patient_sample_index'])['date'].rank(method='first')

                    
                    curr_pred = curr_pred.sort_values(by=['date'])
                    curr_pred["day_pred"] = curr_pred.groupby(['patientid', 'patient_sample_index'])['date'].rank(method='first')


                    flattened_dfs_target.append(curr_target.rename(columns={col : "values_target"})[["day_targ", "values_target"]])
                    flattened_dfs_predicted.append(curr_pred.rename(columns={col : "values_pred"})[["day_pred", "values_pred"]])


        flattened_dfs_target = pd.concat(flattened_dfs_target, axis=0)
        flattened_dfs_predicted = pd.concat(flattened_dfs_predicted, axis=0)


        flattened_dfs = pd.concat([flattened_dfs_target, flattened_dfs_predicted], axis=1)
        

        #: plot
        p = ggplot(aes(x='values_target', y='values_pred'), flattened_dfs)   # Basics
        p = p + geom_point(aes(color='factor(day_targ)'), size=0.1)   # Scatter plot
        p = p + geom_abline(intercept=0, slope=1)   # Diagonal line


        #: return
        return p


    def facet_scatter_plot_all_numeric_values_by_column(self, predicted_df, target_df, alpha=0.2):
        
        flattened_dfs_target = []
        flattened_dfs_predicted = []

        #: get all numeric columns and their respective patient_sample_index into one column
        for col in target_df.columns:

            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":

                    curr_target = target_df.loc[~pd.isnull(target_df[col])].copy()
                    curr_pred = predicted_df.loc[~pd.isnull(target_df[col])].copy()

                    curr_target["variable_targ"] = col
                    curr_pred["variable_pred"] = col

                    flattened_dfs_target.append(curr_target.rename(columns={col : "values_target"})[["variable_targ", "values_target"]])
                    flattened_dfs_predicted.append(curr_pred.rename(columns={col : "values_pred"})[["variable_pred", "values_pred"]])


        flattened_dfs_target = pd.concat(flattened_dfs_target, axis=0)
        flattened_dfs_predicted = pd.concat(flattened_dfs_predicted, axis=0)
        flattened_dfs = pd.concat([flattened_dfs_target, flattened_dfs_predicted], axis=1)

        # Convert to descriptive
        flattened_dfs["variable_targ"] = self.convert_column_to_descriptive(flattened_dfs["variable_targ"])

        #: plot
        p = ggplot(aes(x='values_target', y='values_pred'), flattened_dfs)   # Basics
        p = p + geom_point(aes(color='factor(variable_targ)'), size=0.1, alpha=alpha)   # Scatter plot
        p = p + facet_wrap('variable_targ')   # Make it faceted
        p = p + geom_abline(intercept=0, slope=1)   # Diagonal line
        p = p + scale_color_discrete(guide=False)  # Hide legend



        #: return
        return p
    


    def facet_scatter_plot_column_across_meta_data(self, predicted_df, target_df, meta_data, column_to_visualize, meta_data_column_with_groups, top_k_most_common_groups):
        
        flattened_dfs_target = []
        flattened_dfs_predicted = []

        #: get all numeric columns and their respective patient_sample_index into one column
        non_na_rows = ~pd.isnull(target_df[column_to_visualize])
        curr_target = target_df.loc[~pd.isnull(target_df[column_to_visualize])].copy()
        curr_pred = predicted_df.loc[~pd.isnull(target_df[column_to_visualize])].copy()

        curr_target["variable_targ"] = column_to_visualize
        curr_pred["variable_pred"] = column_to_visualize

        flattened_dfs_target.append(curr_target.rename(columns={column_to_visualize : "values_target"})[["variable_targ", "values_target"]])
        flattened_dfs_predicted.append(curr_pred.rename(columns={column_to_visualize : "values_pred"})[["variable_pred", "values_pred"]])

        flattened_dfs_target = pd.concat(flattened_dfs_target, axis=0)
        flattened_dfs_predicted = pd.concat(flattened_dfs_predicted, axis=0)
        flattened_dfs = pd.concat([flattened_dfs_target, flattened_dfs_predicted], axis=1)

        meta_col = meta_data[meta_data_column_with_groups][non_na_rows.reset_index(drop=True)].copy()
        flattened_dfs = flattened_dfs.reset_index()
        meta_col = meta_col.reset_index()
        cols = flattened_dfs.columns.to_list()
        flattened_dfs = pd.concat([flattened_dfs, meta_col], axis=1, ignore_index=True)
        cols.append("random_index")
        cols.append("group")
        flattened_dfs.columns = cols
        # Get the most common groups
        common_groups = flattened_dfs['group'].value_counts().nlargest(top_k_most_common_groups).index
        flattened_dfs['group'] = flattened_dfs['group'].where(flattened_dfs['group'].isin(common_groups), 'Other')  # Replace less common groups with 'Other'

        # Convert to descriptive
        flattened_dfs["variable_targ"] = self.convert_column_to_descriptive(flattened_dfs["variable_targ"])
        title_str = self.descriptive_mapping[column_to_visualize]

        #: plot
        p = ggplot(aes(x='values_target', y='values_pred'), flattened_dfs)   # Basics
        p = p + geom_point(aes(color='factor(group)'), size=0.1)   # Scatter plot
        p = p + facet_wrap('group')   # Make it faceted
        p = p + geom_abline(intercept=0, slope=1)   # Diagonal line
        p = p + scale_color_discrete(guide=False)  # Hide legend
        p = p + ggtitle(title_str)

        #: return
        return p
    

    def facet_plot_trajectories_across_meta_data(self, predicted_df, target_df, meta_data, column_to_visualize, 
                                                 meta_data_column_with_groups, top_k_most_common_groups, 
                                                 meta_data_column_with_starting_date,
                                                 xlims, ylims,
                                                 trajectory_alpha=0.1, trajectory_size=0.5):
        
        flattened_dfs_target = []
        flattened_dfs_predicted = []

        #: process last_input_values_of_targets into a list
        idx_to_select = np.argmax(np.asarray([x[2] for x in meta_data["last_input_values_of_targets"].tolist()[0]]) == column_to_visualize)
        last_input_values_of_targets = [(x[idx_to_select][0], x[idx_to_select][1]) for x in meta_data["last_input_values_of_targets"].tolist()] 

        last_input_values_of_targets = pd.DataFrame(last_input_values_of_targets, columns=["last_input_date", "last_input_value"]) 
        last_input_values_of_targets = pd.concat([meta_data[[meta_data_column_with_starting_date, meta_data_column_with_groups, "patientid", "patient_sample_index"]], 
                                                  last_input_values_of_targets], axis=1)

        last_input_values_of_targets["last_input_date"] = pd.to_datetime(last_input_values_of_targets["last_input_date"])
        last_input_values_of_targets[meta_data_column_with_starting_date] = pd.to_datetime(last_input_values_of_targets[meta_data_column_with_starting_date])
        last_input_values_of_targets["relative_days"] = (last_input_values_of_targets["last_input_date"] - last_input_values_of_targets[meta_data_column_with_starting_date]).dt.days


        #: get all numeric columns and their respective patient_sample_index into one column
        non_na_rows = ~pd.isnull(target_df[column_to_visualize])
        curr_target = target_df.loc[~pd.isnull(target_df[column_to_visualize])].copy()
        curr_pred = predicted_df.loc[~pd.isnull(target_df[column_to_visualize])].copy()

        curr_target["variable_targ"] = column_to_visualize
        curr_pred["variable_pred"] = column_to_visualize

        flattened_dfs_target.append(curr_target.rename(columns={column_to_visualize : "values_target", "date" : "date_1"})[["variable_targ", "values_target", "date_1"]])
        flattened_dfs_predicted.append(curr_pred.rename(columns={column_to_visualize : "values_pred", "date" : "date_2"})[["variable_pred", "values_pred", "date_2"]])

        flattened_dfs_target = pd.concat(flattened_dfs_target, axis=0)
        flattened_dfs_predicted = pd.concat(flattened_dfs_predicted, axis=0)
        flattened_dfs = pd.concat([flattened_dfs_target, flattened_dfs_predicted], axis=1)

        meta_col = meta_data[[meta_data_column_with_groups, meta_data_column_with_starting_date]][non_na_rows.reset_index(drop=True)].copy()
        flattened_dfs = flattened_dfs.reset_index()
        meta_col = meta_col.reset_index()
        cols = flattened_dfs.columns.to_list()
        flattened_dfs = pd.concat([flattened_dfs, meta_col], axis=1, ignore_index=True)
        cols.append("random_index")
        cols.append("group")
        cols.append("starting_date")
        flattened_dfs.columns = cols
        # Get the most common groups
        common_groups = flattened_dfs['group'].value_counts().nlargest(top_k_most_common_groups).index
        flattened_dfs['group'] = flattened_dfs['group'].where(flattened_dfs['group'].isin(common_groups), 'Other')  # Replace less common groups with 'Other'

        # Convert to descriptive
        flattened_dfs["variable_targ"] = self.convert_column_to_descriptive(flattened_dfs["variable_targ"])
        
        # Merge patient id name and patient sample index
        flattened_dfs["p_index"] = flattened_dfs["patientid"].astype(str) + "_" + flattened_dfs["patient_sample_index"].astype(str)

        #: get relative LoT day from start
        flattened_dfs["starting_date"] = pd.to_datetime(flattened_dfs["starting_date"])
        flattened_dfs["date_1"] = pd.to_datetime(flattened_dfs["date_1"])
        flattened_dfs["relative_days"] = (flattened_dfs["date_1"] - flattened_dfs["starting_date"]).dt.days
        flattened_dfs["type"] = "Predictions"


        #: add last_input_values_of_targets as entries in flattened_dfs
        last_input_values_of_targets["values_target"] = last_input_values_of_targets["last_input_value"]
        last_input_values_of_targets["values_pred"] = last_input_values_of_targets["last_input_value"]
        last_input_values_of_targets["p_index"] = last_input_values_of_targets["patientid"].astype(str) + "_" + last_input_values_of_targets["patient_sample_index"].astype(str)
        last_input_values_of_targets["group"] = last_input_values_of_targets[meta_data_column_with_groups]
        last_input_values_of_targets["group"] = last_input_values_of_targets['group'].where(last_input_values_of_targets['group'].isin(common_groups), 'Other') 
        last_input_values_of_targets["type"] = "Last Observed Value"
        last_input_values_of_targets = last_input_values_of_targets.drop_duplicates()
        flattened_dfs = pd.concat([flattened_dfs, last_input_values_of_targets], axis=0)


        #: turn into one vertical
        final_df_true = flattened_dfs[["values_target", "p_index", "relative_days", "group", "type"]].reset_index(drop=True)
        final_df_pred = flattened_dfs[["values_pred", "p_index", "relative_days", "group", "type"]].reset_index(drop=True)
        final_df_true["vals"] = final_df_true["values_target"] 
        final_df_pred["vals"] = final_df_pred["values_pred"] 
        final_df_true["col_group"] = final_df_true["group"] 
        final_df_pred["col_group"] = final_df_pred["group"] 
        final_df_true["true_group"] = final_df_true["group"] + " (Labels)"
        final_df_pred["true_group"] = final_df_pred["group"] + " (Predictions)"
        
        final_df = pd.concat([final_df_true, final_df_pred], axis=0)
        final_df = final_df.dropna(how='all')

        title_str = self.descriptive_mapping[column_to_visualize]

        # make two plots next to each other: one for true, another for predicted, return both
        p1 = ggplot(final_df, aes(x='relative_days', y='vals', group='factor(p_index)', color='factor(col_group)', linetype='type'))   # Basics
        p1 = p1 + geom_line(size=trajectory_size, alpha=trajectory_alpha)   # Scatter plot
        p1 = p1 + facet_wrap('~true_group', ncol=2)   # Make it faceted
        p1 = p1 +  xlab("Relative days to start of LoT")
        p1 = p1 +  ylab("True Value, per patient per line of therapy")
        p1 = p1 + coord_cartesian(xlim=xlims, ylim=ylims)
        p1 = p1 + scale_color_discrete(guide=False)  # Hide legend
        p1 = p1 + scale_linetype_manual(values={"Predictions": "solid", "Last Observed Value": "dotted"})
        p1 = p1 + ggtitle(title_str)

        #: return
        return p1


    def plot_individual_trajectories(self, predicted_df, target_df, meta_data,
                                     column_to_visualize = "lab_26499_4",
                                     meta_data_column_with_groups="line_name",
                                     top_k_most_common_groups = 3,
                                    meta_data_column_with_starting_date="lot_start_date",
                                    xlims=(0, 91),
                                    ylims=(0, 20),
                                    trajectory_alpha = 1.0,
                                    trajectory_size= 1.0,
                                    extra_trajctories_column = "used_trajectories",
                                    num_patients_to_show = 3,
                                    selection_strategy = "worst",
                                    inches_vertical_per_plot = 4,
                                    inches_horizontal_per_plot = 10,
                                    eval_manager=None):

        # This is an incredibly hacky plotting function, to show individual trajectories for a number of patients
        # selection_strategy can be "random","worst", "best"


        # Code
        base_path = os.path.dirname(__file__).split("/uc2_nsclc")[0] + "/uc2_nsclc/"  # Hacky way to get the base path
        column_descriptive_mapping_path = base_path + "2_experiments/2023_11_07_neutrophils/1_data/column_descriptive_name_mapping.csv"
        descriptive_mapping_df = pd.read_csv(column_descriptive_mapping_path)
        descriptive_mapping = {row[1]["original_column_names"] : row[1]["descriptive_column_name"] for row in descriptive_mapping_df.iterrows()}


        def convert_column_to_descriptive(column):

                col_list = column.tolist()

                ret_list = []

                for col in col_list:
                    if col in descriptive_mapping:
                        ret_list.append(descriptive_mapping[col])
                    else:
                        ret_list.append(col)

                return ret_list


        flattened_dfs_target = []
        flattened_dfs_predicted = []

        #: process last_input_values_of_targets into a list
        #meta_data["last_input_values_of_targets"] = meta_data["last_input_values_of_targets"].apply(lambda x: eval(x))
        idx_to_select = np.argmax(np.asarray([x[2] for x in meta_data["last_input_values_of_targets"].tolist()[0]]) == column_to_visualize)
        last_input_values_of_targets = [(x[idx_to_select][0], x[idx_to_select][1]) for x in meta_data["last_input_values_of_targets"].tolist()] 
        last_input_values_of_targets = pd.DataFrame(last_input_values_of_targets, columns=["last_input_date", "last_input_value"]) 
        last_input_values_of_targets = pd.concat([meta_data[[meta_data_column_with_starting_date, meta_data_column_with_groups, "patientid", "patient_sample_index"]], 
                                                last_input_values_of_targets], axis=1)

        last_input_values_of_targets["last_input_date"] = pd.to_datetime(last_input_values_of_targets["last_input_date"])
        last_input_values_of_targets[meta_data_column_with_starting_date] = pd.to_datetime(last_input_values_of_targets[meta_data_column_with_starting_date])
        last_input_values_of_targets["relative_days"] = (last_input_values_of_targets["last_input_date"] - last_input_values_of_targets[meta_data_column_with_starting_date]).dt.days


        #: get all numeric columns and their respective patient_sample_index into one column
        non_na_rows = ~pd.isnull(target_df[column_to_visualize])
        curr_target = target_df.loc[~pd.isnull(target_df[column_to_visualize])].copy()
        curr_pred = predicted_df.loc[~pd.isnull(target_df[column_to_visualize])].copy()

        curr_target["variable_targ"] = column_to_visualize
        curr_pred["variable_pred"] = column_to_visualize

        flattened_dfs_target.append(curr_target.rename(columns={column_to_visualize : "values_target", "date" : "date_1"})[["variable_targ", "values_target", "date_1"]])  # , "patientid", "patient_sample_index"
        flattened_dfs_predicted.append(curr_pred.rename(columns={column_to_visualize : "values_pred", "date" : "date_2"})[["variable_pred", "values_pred", "date_2"]])

        flattened_dfs_target = pd.concat(flattened_dfs_target, axis=0)
        flattened_dfs_predicted = pd.concat(flattened_dfs_predicted, axis=0)
        flattened_dfs = pd.concat([flattened_dfs_target, flattened_dfs_predicted], axis=1)

        meta_col = meta_data[[meta_data_column_with_groups, meta_data_column_with_starting_date]][non_na_rows.reset_index(drop=True)].copy()
        flattened_dfs = flattened_dfs.reset_index()
        meta_col = meta_col.reset_index()
        cols = flattened_dfs.columns.to_list()
        flattened_dfs = pd.concat([flattened_dfs, meta_col], axis=1, ignore_index=True)
        cols.append("random_index")
        cols.append("group")
        cols.append("starting_date")
        flattened_dfs.columns = cols
        target_cols = eval_manager.get_column_usage()[2].copy()
        # Get the most common groups
        common_groups = flattened_dfs['group'].value_counts().nlargest(top_k_most_common_groups).index
        flattened_dfs['group'] = flattened_dfs['group'].where(flattened_dfs['group'].isin(common_groups), 'Other')  # Replace less common groups with 'Other'

        # Convert to descriptive
        flattened_dfs["variable_targ"] = convert_column_to_descriptive(flattened_dfs["variable_targ"])

        # Merge patient id name and patient sample index
        flattened_dfs["p_index"] = flattened_dfs["patientid"].astype(str) + "_" + flattened_dfs["patient_sample_index"].astype(str)

        #: get relative LoT day from start
        flattened_dfs["starting_date"] = pd.to_datetime(flattened_dfs["starting_date"])
        flattened_dfs["date_1"] = pd.to_datetime(flattened_dfs["date_1"])
        flattened_dfs["relative_days"] = (flattened_dfs["date_1"] - flattened_dfs["starting_date"]).dt.days


        #: add last_input_values_of_targets
        last_input_values_of_targets["values_target"] = last_input_values_of_targets["last_input_value"]
        last_input_values_of_targets["values_pred"] = last_input_values_of_targets["last_input_value"]
        last_input_values_of_targets["p_index"] = last_input_values_of_targets["patientid"].astype(str) + "_" + last_input_values_of_targets["patient_sample_index"].astype(str)
        last_input_values_of_targets["group"] = last_input_values_of_targets[meta_data_column_with_groups]
        last_input_values_of_targets["group"] = last_input_values_of_targets['group'].where(last_input_values_of_targets['group'].isin(common_groups), 'Other') 
        last_input_values_of_targets["type"] = "Last Observed Value"

        last_input_values_of_targets = last_input_values_of_targets.drop_duplicates()
        flattened_dfs = pd.concat([flattened_dfs, last_input_values_of_targets], axis=0)


        #: turn into one vertical
        final_df_true = flattened_dfs[["values_target", "p_index", "relative_days", "group", "type", "starting_date"]].reset_index(drop=True)
        final_df_pred = flattened_dfs[["values_pred", "p_index", "relative_days", "group", "type", "starting_date"]].reset_index(drop=True)
        final_df_true["vals"] = final_df_true["values_target"] 
        final_df_pred["vals"] = final_df_pred["values_pred"] 
        final_df_true["col_group"] = final_df_true["group"] 
        final_df_pred["col_group"] = final_df_pred["group"] 
        final_df_true["true_group"] = "Observed Values"
        final_df_pred["true_group"] = "Final Predictions"
        final_df_true.loc[final_df_true["type"] != "Last Observed Value", "type"] = "Observed Values"
        final_df_pred.loc[final_df_pred["type"] != "Last Observed Value", "type"] = "Final Predictions"

        final_df = pd.concat([final_df_true, final_df_pred], axis=0)
        final_df = final_df.dropna(how='all')
        

        # LoT stuff
        base_path = os.path.dirname(__file__).split("/uc2_nsclc")[0] + "/uc2_nsclc/"  # Hacky way to get the base path        
        lot = pd.read_csv(base_path + "2_experiments/2023_11_07_neutrophils/1_data/line_of_therapy.csv")



        def get_administrations_for_patient(patient_id, patient_sample_index, eval_manager, split_date):

            constants_row, events_table = eval_manager.get_full_patient_info(patient_id)

            #: extract all rows which begin with "drug."
            drug_rows_list = [x for x in events_table.columns.tolist() if x.startswith("drug_")]
            drug_rows_list.insert(0, "date")
            drug_rows = events_table[drug_rows_list]

            #: get all dates where at least one drug was administered
            columns_to_check = drug_rows.columns.difference(['date'])
            drug_rows = drug_rows.dropna(subset=columns_to_check,how='all')
            drug_dates = drug_rows["date"].tolist()
            drug_dates = [pd.to_datetime(x) for x in drug_dates]
            
            #: filter out in case of switching
            line_setting = int(patient_sample_index.split("_")[1]) + 1
            line_info = lot[(lot["patientid"] == patient_id) & (lot["linenumber"] == line_setting)]
            if len(line_info) > 0:
                latest_date = pd.to_datetime(line_info["enddate"].iloc[0])
                drug_dates = [x for x in drug_dates if x <= latest_date]

            #: offset dates by the starting date
            drug_dates = [(x - split_date).days for x in drug_dates]

            #: filter for those in x lims
            drug_dates = [x for x in drug_dates if x >= xlims[0] and x <= xlims[1]]

            return drug_dates


        def plot_for_patient_trajectories(p_index, curr_annotation):

            #: filter for specific patientid and patient_sample_index
            curr_final_df = final_df[(final_df["p_index"] == p_index)]
            patient_id_to_filter, patient_sample_index_to_filter = p_index.split("_", maxsplit=1)  # Hacky

            # Setup column usage
            all_cols = ["patientid", "patinent_sample_index", "date"]
            all_cols.extend(target_cols)
            base_cols = ["patientid", "patinent_sample_index", "date", column_to_visualize]

            #: add in all the other trajectories - extra_trajctories_column
            trajectory_data = meta_data[(meta_data["patientid"] == patient_id_to_filter) & (meta_data["patient_sample_index"] == patient_sample_index_to_filter)][extra_trajctories_column].tolist()[0]
            trajectory_data = [pd.DataFrame(x, columns=all_cols)[base_cols] for x in trajectory_data]
            trajectory_data = [df.rename(columns={column_to_visualize: 'value'}, inplace=False) for df in trajectory_data if column_to_visualize in df.columns]

            starting_date = curr_final_df["starting_date"].iloc[0]
            values_to_add = []

            for trajectory_idx, x in enumerate(trajectory_data):
                
                #: get relative date
                x["relative_days"] = (pd.to_datetime(x["date"]) - starting_date).dt.days

                #: add all values
                for idx in range(len(x)):
                    values_to_add.append({
                        "values_target": np.nan,
                        "values_pred": np.nan,
                        "p_index": p_index,
                        "relative_days": x.iloc[idx]["relative_days"],
                        "group": "",
                        "type": "Generated Predictions",
                        "starting_date": np.nan,
                        "vals": float(x.iloc[idx]["value"]),
                        "col_group": np.nan,
                        "true_group": "Generated Trajectory: " + str(trajectory_idx + 1),
                    })
            values_to_add = pd.DataFrame(values_to_add)
            curr_final_df = pd.concat([curr_final_df, values_to_add], axis=0)

            meta_info = meta_data[(meta_data["patientid"] == patient_id_to_filter) & (meta_data["patient_sample_index"] == patient_sample_index_to_filter)]
            line_name = meta_info["line_name"].iloc[0]
            line_number = meta_info["line_number"].iloc[0]

            # To make last observed value a dot
            last_observed_df = curr_final_df[curr_final_df['type'] == 'Last Observed Value'].copy()

            curr_final_df_no_last_observed = curr_final_df[curr_final_df['type'] != 'Last Observed Value'].copy()
            last_observed_df_normal = last_observed_df.copy()
            last_observed_df_normal["type"] = ["Observed Values", "Final Predictions"]
            curr_final_df = pd.concat([curr_final_df_no_last_observed, last_observed_df_normal], axis=0)


            # if provided, get drug administration info
            if eval_manager is not None:
                drug_dates = get_administrations_for_patient(patient_id_to_filter, patient_sample_index_to_filter, eval_manager, starting_date)
                # Put them on the lower y axis
                y_vals = [ylims[0]] * len(drug_dates)
                drug_dates = pd.DataFrame({'relative_days': drug_dates, 'vals': y_vals, "true_group": "Administrations", "type": "Administrations"})

            else:
                drug_dates = None

            # Make observed values df
            obs_dates = curr_final_df[curr_final_df['type'] == 'Observed Values'].copy()
            obs_dates["vals"] = ylims[0]
            obs_dates["type"] = "Observed Dates"

            value_str = self.descriptive_mapping[column_to_visualize]

            # make two plots next to each other: one for true, another for predicted, return both
            p1 = ggplot(curr_final_df, aes(x='relative_days', y='vals', group='factor(true_group)', size="factor(type)", color='factor(type)', linetype='factor(type)'))   # Basics
            
            p1 = p1 + geom_line(alpha=trajectory_alpha)   # Scatter plot
            p1 = p1 + geom_point(aes(x='relative_days', y='vals'), data=obs_dates, shape='+')
            
            p1 = p1 + geom_point(data=last_observed_df, mapping=aes(x='relative_days', y='vals', size="factor(type)", color='factor(type)'), alpha=1)
            p1 = p1 + labs(title="Line: " + str(line_name) + " - Line Nr: " + str(int(line_number)) + " - " + str(curr_annotation))
            p1 = p1 + xlab("Relative days to start of LoT - Patient: " + str(p_index))
            p1 = p1 + ylab(value_str)
            p1 = p1 + coord_cartesian(xlim=xlims, ylim=ylims)
            p1 = p1 + scale_linetype_manual(values={
                "Final Predictions": "solid",
                "Observed Values": "solid",
                'Last Observed Value': "solid",  # Change this to "solid" or whatever your normal line setting is
                "Generated Predictions": "dashed",
                "Administrations" : "dashed",
                "Observed Dates" : "dashed",

            })
            p1 = p1 + scale_size_manual(values={"Final Predictions": trajectory_size,
                                                "Observed Values" : trajectory_size, 
                                                'Last Observed Value': 3,
                                                "Generated Predictions": trajectory_size / 3.0,
                                                "Administrations" : trajectory_size/2.0,
                                                "Observed Dates" : 3})
            
            p1 = p1 + scale_color_manual(values={
                "Final Predictions": "#91DB57",  # blue
                "Observed Values": "#A157DB",    # orange
                'Last Observed Value': "pink", # green
                "Generated Predictions": "skyblue", # red
                "Administrations": "gray",    # purple
                "Observed Dates": "peru"      # brown
            })

            p1 = p1 + theme(legend_title=element_blank()) 

            # Add drug dates
            if drug_dates is not None:
                # p1 = p1 + geom_point(aes(x='relative_days', y='vals'), data=drug_dates)
                p1 = p1 + geom_vline(aes(xintercept='relative_days', size="factor(type)", color='factor(type)', linetype='factor(type)'), data=drug_dates)


            return p1


        def get_worst_and_best_patients(k, predicted_df):

            meta_data["date"] = pd.to_datetime(meta_data['date'])
            meta_data["split_date"] = pd.to_datetime(meta_data['split_date'])
            meta_data["relative_days"] = (meta_data["date"] - meta_data["split_date"]).dt.days

            #: bring meta data and predictions together
            meta_data["targets"] = meta_data[column_to_visualize]
            predicted_df = predicted_df.reset_index(drop=True)
            meta_data["predictions"] = predicted_df[column_to_visualize]

            #: get mae for every trajectory
            def mean_absolute_error(group):
                return (group['targets'] - group['predictions']).abs().mean()

            # Group by 'patientid' and 'patient_sample_index' and apply the MAE function
            grouped_mae = meta_data.groupby(['patientid', 'patient_sample_index']).apply(mean_absolute_error)

            # Reset index if you want 'patientid' and 'patient_sample_index' back as columns
            grouped_mae = grouped_mae.reset_index(name='MAE')

            #: get top-k and bottom-k trajectories
            grouped_mae = grouped_mae.sort_values(by=['MAE'])
            top_k = grouped_mae.nsmallest(k, 'MAE')
            bottom_k = grouped_mae.nlargest(k, 'MAE')

            return top_k, bottom_k


        #: add selection parameters
        all_pindex = final_df["p_index"].unique().tolist()

        top_k, bottom_k = get_worst_and_best_patients(num_patients_to_show, predicted_df)
        annotation_list = []

        if selection_strategy == "random":   # "random","worst", "best"
            local_random = random.Random()
            local_random.seed(42)
            unique_value_pairs = local_random.sample(all_pindex, num_patients_to_show)
            annotation_list = ["Random"] * len(unique_value_pairs)

        elif selection_strategy == "worst":
            bottom_k_list = bottom_k["patientid"].astype(str) + "_" + bottom_k["patient_sample_index"].astype(str)
            unique_value_pairs = bottom_k_list.tolist()
            annotation_list = ("Worst patients - MAE: " + bottom_k["MAE"].round(2).astype(str)).tolist()

        elif selection_strategy == "best":
            
            top_k_list = top_k["patientid"].astype(str) + "_" + top_k["patient_sample_index"].astype(str)
            unique_value_pairs = top_k_list.tolist()
            annotation_list = ("Best patients - MAE: " + top_k["MAE"].round(2).astype(str)).tolist()
            

        #: go over all needed patients and generate plots
        plots = []

        for p_index, curr_annotation in zip(unique_value_pairs, annotation_list):
            plots.append(plot_for_patient_trajectories(p_index, curr_annotation))


        #: combine into one matplotlib plot

        def plotnine_grid(plots_list, row=None, col=1, height=None, width=None, dpi=500, ratio=None, pixels=10000,
                        figsize=(12, 8)):

            """"
            Create a grid of plotnine plots.


            Function input
            ----------
            plots_list      : a list of plotnine.ggplots
            row, col        : numerics to indicate in how many rows and columns the plots should be ordered in the grid
                        defaults: row -> length of plots_list; col -> 1
            height, width   : the height and width of the individual subplots created by plotnine
                            can be automatically determined by a combination of dpi, ratio and pixels
            dpi             : the density of pixels in the image. Default: 500. Higher numbers could lead to crisper output,
                            depending on exact situation
            ratio           : the ratio of heigth to width in the output. Standard value is 1.5 x col/row.
                            Not used if height & width are given.
            pixels          : the total number of pixels used, default 10000. Not used if height & width are given.
            figsize         : tuple containing the size of the final output after making a grid, in pixels (default: (1200,800))


            Function output
            ----------
            A matplotlib figure that can be directly saved with output.savefig().
            """

            # Assign values that have not been provided based on others. In the end, height and width should be provided.
            if row is None:
                row = len(plots_list)

            if ratio is None:
                ratio = 1.5 * col / row

            if height is None and width is not None:
                height = ratio * width

            if height is not None and width is None:
                width = height / ratio

            if height is None and width is None:
                area = pixels / dpi
                width = np.sqrt(area/ratio)
                height = ratio * width

            # Do actual subplot creation and plot output.
            i = 1
            fig = plt.figure(figsize=figsize)
            plt.autoscale(tight=True)
            
            for image_sel in plots_list:  # image_sel = plots_list[i]
                image_sel.save('image' + str(i) + '.png', height=height, width=width, dpi=500, verbose=False)
                fig.add_subplot(row, col, i)
                plt.imshow(img.imread('image' + str(i) + '.png'), aspect='auto')
                fig.tight_layout()
                fig.get_axes()[i-1].axis('off')
                i = i + 1
                os.unlink('image' + str(i - 1) + '.png')  # os.unlink is basically os.remove but in some cases quicker
            
            fig.patch.set_visible(False)
            fig.tight_layout()
            return fig

        f = plotnine_grid(plots, width=inches_horizontal_per_plot, height=inches_vertical_per_plot, figsize=(inches_horizontal_per_plot, len(plots) * inches_vertical_per_plot))

        return f


    def plot_mimic_trajectories(self, predicted_df, target_df, meta_data, column_to_visualize, ylims,
                                num_patients_to_show = 3, trajectory_alpha = 1.0, trajectory_size= 1.0):
        
        #: select random non-na patients to plot
        non_na_targets = target_df.dropna(subset=[column_to_visualize])
        all_patients_samples = non_na_targets[["patientid", "patient_sample_index"]].drop_duplicates()
        random_patients = all_patients_samples.sample(num_patients_to_show, random_state=42)

        #: get variable descriptive name
        base_path = os.path.dirname(__file__).split("/uc2_nsclc")[0] + "/uc2_nsclc/"  # Hacky way to get the base path
        r2_csv_with_names = pd.read_csv(base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/column_descriptive_name_mapping.csv")
        des_name = r2_csv_with_names[r2_csv_with_names["original_column_names"] == column_to_visualize]["descriptive_column_name"].values[0]
        
        ret_plots = []

        #: for each of them plot
        for patientid, patient_sample_index in random_patients.values.tolist():
            
            #: get the data
            curr_target = target_df[(target_df["patientid"] == patientid) & (target_df["patient_sample_index"] == patient_sample_index)]
            curr_pred = predicted_df[(predicted_df["patientid"] == patientid) & (predicted_df["patient_sample_index"] == patient_sample_index)]

            data_target = pd.DataFrame({"values" : curr_target[column_to_visualize],
                                        "date" : curr_target["date"]})
            data_target["group"] = "Observed Values"
            
            data_pred = pd.DataFrame({"values" : curr_pred[column_to_visualize],
                                        "date" : curr_pred["date"]})
            data_pred["group"] = "Predicted Values"
            
            data = pd.concat([data_target, data_pred], axis=0, ignore_index=True)

            # Interpolate over missing values
            data['values'] = data.groupby('group')['values'].transform(lambda group: group.interpolate())
            data["date"] = pd.to_datetime(data["date"])
            

            # visualize each unique index across time
            def custom_formatter(dates):
                return [date.strftime('%H') for date in pd.to_datetime(dates)]

            plot = ggplot(data, aes(x="date", y="values", color="factor(group)"))
            #plot += geom_line()
            plot += geom_path()
            plot += theme(axis_text_x=element_text(angle=90), figure_size=(10, 5))
            plot += scale_x_datetime(labels=custom_formatter, date_breaks='1 hour')
            plot += ylab(des_name)
            plot += ylim(ylims)
            plot += xlab("Hours")
            plot += labs(title="Visualization of patientid: " + str(patientid) + " - patient_sample_index: " + str(patient_sample_index))

            ret_plots.append(plot)
            
        return ret_plots


