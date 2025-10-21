import pandas as pd
import numpy as np
import os
import json
from collections import Counter
import time
import wandb



def main():

    print("\n\n\n\n\n")
    print("==========================================================================================================")
    print("Starting main filtering and processing")

    #: go through every csv in the folder, open as csv, then note down stats for non-zero/non-na values
    load_folder_path = "/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/1_raw_events/csv"
    save_folder_path = "/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/events"
    save_folder_path_constant = "/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/0_final_data"

    # get all files in the folder
    files = os.listdir(load_folder_path)

    #: open stats dic, and select which variables to keep
    with open('/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/2024_02_01_raw_data_stats.json') as f:
        stats = json.load(f)
    
    # convert everything to numpy arrays if possible
    for key in stats:
        for subkey in stats[key]:
            if isinstance(stats[key][subkey], list):
                stats[key][subkey] = np.array(stats[key][subkey]).astype(float)

    top_k_labs = 50
    top_k_inputevents = 50
    top_k_diagnoses = 100
    top_k_procedures = 50
    top_k_output_events = 50

    key_labs = "chartevents"
    key_outputevents = "outputevents"
    key_inputevents = "inputevents"
    key_procedureevents = "procedureevents"
    key_diagnoses = "COND"

    mapping_dic = {
        key_labs: top_k_labs,
        key_outputevents: top_k_output_events,
        key_inputevents: top_k_inputevents,
        key_procedureevents: top_k_procedures,
        key_diagnoses: top_k_diagnoses,
    }

    zero_to_na_mapping = [key_procedureevents, key_inputevents, key_outputevents, key_diagnoses, key_labs]

    all_cols_to_keep_dynamic = []
    all_cols_to_keep_static = []
    all_cols_to_convert_zeros_to_na = []

    for key in mapping_dic:

        all_cols_in_key = []

        for col in stats:
            if stats[col]["linksto"] == key:
                all_cols_in_key.append((col, stats[col]["non_na_or_zero"]))
        
        #: sort by second key
        all_cols_in_key = sorted(all_cols_in_key, key=lambda x: x[1], reverse=True)

        #: keep top k
        all_cols_in_key = all_cols_in_key[:mapping_dic[key]]

        #: append col names to all_cols_to_keep
        col_names = [x[0] for x in all_cols_in_key]

        if key != "COND":
            all_cols_to_keep_dynamic.extend(col_names)
        else:
            all_cols_to_keep_static.extend(col_names)

        if key in zero_to_na_mapping:
            all_cols_to_convert_zeros_to_na.extend(col_names)


    all_links_to = Counter([stats[col]["linksto"] for col in stats])
    time_since_last_batch = time.time()

    #: make empty list for constant df 
    constant_list = []

    # go through every file
    for idx, file in enumerate(files):

        # Skip if not folder
        if not os.path.isdir(os.path.join(load_folder_path, file)):
            continue

        if idx % 10 == 0:
            delta_time = time.time() - time_since_last_batch
            print(f"Processing file {idx} out of {len(files)} taking {delta_time} seconds")
            time_since_last_batch = time.time()

        # open the file as a dataframe
        df = pd.read_csv(os.path.join(load_folder_path, file, "dynamic.csv"), skiprows=1)

        #: keep only relevant columns
        df = df[all_cols_to_keep_dynamic]

        # for every column, note down the number of non-zero/non-na values, the variance across time, the variance in the second half
        for column in df.columns:
            
            # check that this is only done for correct columns (e.g. medication)
            if column in all_cols_to_convert_zeros_to_na:

                #: convert 0s to nans
                df[column] = df[column].replace(0, np.nan)

        #: add ids to df
        df.insert(0, "patientid", idx)

        #: add 48 hours to date as range of datetime objects
        df.insert(0, "date", list(pd.date_range(start='2024-01-01', periods=len(df), freq='H')))

        #: save df in folder
        df.to_csv(os.path.join(save_folder_path, f"{idx}_events.csv"))

        #: get demographic + static dfs
        df_demographics = pd.read_csv(os.path.join(load_folder_path, file, "demo.csv"))
        df_static = pd.read_csv(os.path.join(load_folder_path, file, "static.csv"), skiprows=1)

        #: select top diagnsoses
        df_static = df_static[all_cols_to_keep_static]
        df_static = df_static.replace(0, np.nan)
        df_static = df_static.replace(1.0, "diagnosed")
        
        #: combine with demographic
        df_static = pd.concat([df_demographics, df_static], axis=1)
        df_static.insert(0, "patientid", idx)

        #: save in constant list
        constant_list.append(df_static)


    #: save final constants df
    print("Saving constants df")
    df_constants = pd.concat(constant_list, axis=0, ignore_index=True)
    df_constants.to_csv(os.path.join(save_folder_path_constant, "constants.csv"))

    
    


if __name__ == "__main__":

    debug = False

    #: setup wandb
    if debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(project='UC - MIMIC-IV', group="Data Processing")

    main()






