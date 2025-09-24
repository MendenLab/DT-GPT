import pandas as pd
import numpy as np
import os
import json
import time
import wandb


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



def main():

    #: go through every csv in the folder, open as csv, then note down stats for non-zero/non-na values

    folder_path = "/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/1_raw_events/csv"

    # get all files in the folder
    files = os.listdir(folder_path)

    # create a dictionary to store the stats
    stats = {}

    # Load in helper csv
    help_csv_path = "/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/0_raw_helper_files/d_items.csv"
    help_df = pd.read_csv(help_csv_path)
    help_df["itemid"] = help_df["itemid"].astype(str)

    # load in diagnosis csv
    diagnosis_csv_path = "/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/0_raw_helper_files/d_icd_diagnoses.csv"
    diagnosis_df = pd.read_csv(diagnosis_csv_path)


    time_since_last_batch = time.time()

    # go through every file
    for idx, file in enumerate(files):

        # Skip if not folder
        if not os.path.isdir(os.path.join(folder_path, file)):
            continue

        if idx % 10 == 0:
            delta_time = time.time() - time_since_last_batch
            print(f"Processing file {idx} out of {len(files)} taking {delta_time} seconds")
            time_since_last_batch = time.time()

        # open the file as a dataframe
        df = pd.read_csv(os.path.join(folder_path, file, "dynamic.csv"), skiprows=1)

        #: read also static.csv
        df_static = pd.read_csv(os.path.join(folder_path, file, "static.csv"), skiprows=1)

        # for every column, note down the number of non-zero/non-na values, the variance across time, the variance in the second half
        for column in df.columns:
            non_na = df[column].count()
            non_zero = df[column].astype(bool).sum(axis=0)
            num_non_zero_in_second_half = df[column][len(df)//2:].astype(bool).sum(axis=0)

            #: convert 0s to nans
            df[column] = df[column].replace(0, np.nan)
            non_na_or_zero = df[column].count()

            std = str(df[column].std())        # Do as string in case of nans
            std_second_half = str(df[column][len(df)//2:].std())       # Do as string in case of nans
            std_div_mean = str(df[column].std()/df[column].mean())       # Do as string in case of nans
            std_second_half_div_mean = str(df[column][len(df)//2:].std() / df[column][len(df)//2:].mean())       # Do as string in case of nans

            #: if empty initialize
            if column not in stats:
                stats[column]= {
                    "non_na": non_na,
                    "non_zero": non_zero,
                    "non_na_or_zero": non_na_or_zero,
                    "std": [std],
                    "std_second_half": [std_second_half],
                    "num_non_zero_in_second_half": num_non_zero_in_second_half,
                    "std_div_mean": [std_div_mean],
                    "std_second_half_div_mean" : [std_second_half_div_mean],
                }

                # Match on column name, and extract the correspondign value from the column "label"  - NOTE: diagnosis not included
                if help_df[help_df["itemid"] == column].shape[0] > 0:
                    stats[column]["label"] = help_df[help_df["itemid"] == column]["label"].values[0]
                    stats[column]["linksto"] = help_df[help_df["itemid"] == column]["linksto"].values[0]
                    stats[column]["category"] = help_df[help_df["itemid"] == column]["category"].values[0]
                else: 
                    stats[column]["label"] = "unknown"
                    stats[column]["linksto"] = "unknown"
                    stats[column]["category"] = "unknown"

            else:
                stats[column]["non_na"] += non_na
                stats[column]["non_zero"] += non_zero
                stats[column]["non_na_or_zero"] += non_na_or_zero
                stats[column]["std"].append(std)
                stats[column]["std_second_half"].append(std_second_half)
                stats[column]["std_div_mean"].append(std_div_mean)
                stats[column]["std_second_half_div_mean"].append(std_second_half_div_mean)
                stats[column]["num_non_zero_in_second_half"] += num_non_zero_in_second_half

        #: go through static df columns
        for column in df_static.columns:
            non_na = df_static[column].count()
            non_zero = df_static[column].astype(bool).sum(axis=0)
            non_na_or_zero = df_static[column].count()

            #: if empty initialize
            if column not in stats:
                stats[column]= {
                    "non_na": non_na,
                    "non_zero": non_zero,
                    "non_na_or_zero": non_na_or_zero,
                    "std": [],
                    "std_second_half": [],
                    "num_non_zero_in_second_half": 0,
                    "std_div_mean": [],
                    "std_second_half_div_mean" : [],
                }

                # Get diagnosis info
                label = diagnosis_df[diagnosis_df["icd_code"] == column]["long_title"].iloc[0]

                # NOTE: only diagnosis
                stats[column]["label"] = label
                stats[column]["linksto"] = "COND"
                stats[column]["category"] = "Diagnosis"
                
            else:
                stats[column]["non_na"] += non_na
                stats[column]["non_zero"] += non_zero
                stats[column]["non_na_or_zero"] += non_na_or_zero

    
    # Save dictionary
    with open('/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/2024_02_01_raw_data_stats.json', 'w') as f:
        json.dump(stats, f, cls=NpEncoder, indent=2)
    


if __name__ == "__main__":

    debug = False

    #: setup wandb
    if debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(project='UC - MIMIC-IV', group="Data Processing")

    main()





