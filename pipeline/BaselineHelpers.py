import __init__  # Do all imports
import pandas as pd
import numpy as np
import logging
import traceback




def forward_fill_median_backup(input_df, empty_target_dataframe, skip_cols, statistics_dic):

    statistics = statistics_dic
    empty_target_dataframe = empty_target_dataframe.copy()

    try: 

        #: sort by date first
        true_events_input = input_df.sort_values(by=['date'])

        #: first try and apply forward filling
        true_events_input = true_events_input.fillna(method='ffill', axis=0)
        true_events_input = true_events_input.iloc[-1:, :]
        cols_to_extract = [col for col in empty_target_dataframe.columns if col not in skip_cols]

        for col in cols_to_extract:
            if col in true_events_input.columns:
                empty_target_dataframe.loc[:, col] = true_events_input.loc[:, col].tolist()[0]

        #: then apply for each missing value its majority/median class 
        for column in empty_target_dataframe.columns:
            if column in skip_cols:
                continue

            #: any numeric column with last value mean +- 3 sigma should default to median
            if statistics[column]["type"] == "numeric":
                lower_bound = statistics[column]["mean"] - (3 * statistics[column]["std"])
                upper_bound = statistics[column]["mean"] + (3 * statistics[column]["std"])
                rows_to_select = (empty_target_dataframe[column] > upper_bound) | (empty_target_dataframe[column] < lower_bound)
                empty_target_dataframe.loc[rows_to_select, column] = np.nan
            
            # Check for empty
            if empty_target_dataframe[column].isnull().values.any():

                #: get majority or mean class here
                if statistics[column]["type"] == "numeric":
                    empty_target_dataframe[column] = statistics[column]["median"]
                else:
                    empty_target_dataframe[column] = max(statistics[column]["counts"], key=statistics[column]["counts"].get)


        #: evaluate it using eval manager
        targets_nulls = empty_target_dataframe.isnull()
        predicted_df = empty_target_dataframe.astype(str).mask(targets_nulls, np.NaN) # convert to string
        return predicted_df
    
    except Exception:
        
        # Fallback to empty DF in case of any errors
        traceback.print_exc()
        logging.info("Falling back to empty DF!")
        return empty_target_dataframe


     














