import numpy as np

# This script generates all necessary meta data in one go
# used to be multiple scripts but merged for simplicity




################################### generate_column_mapping.py -  GENERATES column_mapping.json FILE ###################################

def generate_column_mapping():
    print("Starting with generate_column_mapping")

    import json
    import pandas as pd


    # get an example events file for the columns
    example_events_df = pd.read_csv("/home/makaron1/dt-gpt/uc2_nsclc/2_experiments/2025_02_03_adni/1_data/1_final_data/patient_events/3.csv")

    # Add info whether it is input, target or known_future_input
    target_columns = ["CDRSB", "ADAS11", "MMSE"]
    known_future_input_groups = ["date", "patientid"]

    # make mapping
    col_names = list(example_events_df.columns)

    mapping = {}

    for idx, col_name in enumerate(col_names):
        
        print(idx)

        mapping[col_name] = {}
        mapping[col_name]["col_index"] = idx
        mapping[col_name]["variable_group"] = "no_groups"

        # Set correct flags
        mapping[col_name]["input"] = True
        mapping[col_name]["known_future_input"] = col_name in known_future_input_groups
        mapping[col_name]["target"] = col_name in target_columns
        

    col_groups = list(set([mapping[col_name]["variable_group"] for col_name in mapping.keys()]))


    # save as json
    with open('/home/makaron1/dt-gpt/uc2_nsclc/2_experiments/2025_02_03_adni/1_data/1_final_data/column_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=4)


    print("Finished with generate_column_mapping")


################################### dataset_statistics_loader.py -  GENERATES dataset_statistics.json FILE ###################################


def dataset_statistics_loader():

    print("Starting with dataset_statistics_loader")

    import __init__
    import pandas as pd
    from EvaluationManager import EvaluationManager
    import numpy as np
    from pandas.api.types import is_numeric_dtype
    import time


    #: load up eval manager
    eval_manager = EvaluationManager("2025_02_03_adni", load_statistics_file=False)  # Do not load the statistics file, since we're building it now


    manual_categorical_columns = []


    #: get all training paths
    training_paths, training_patientids = eval_manager.get_paths_to_events_in_split("TRAIN")
    training_len = len(training_paths)
    skip_cols = ["patientid", "date", "patient_sample_index"]

    all_values = {}
    column_type = {}

    prev_time = time.time()

    for idx, (current_training_path, current_patient_ids) in enumerate(zip(training_paths, training_patientids)):

        # log
        if idx % 100 == 0:
            print("Currently at patient nr: " + str(idx+1) + " / " + str(training_len) + " time for last 100: " + str(time.time() - prev_time))
            prev_time = time.time()

        # load each dataframe
        patient_events_table = pd.read_csv(current_training_path)
        patient_events_table = patient_events_table.drop(['Unnamed: 0', 'X.2', 'X.1', 'X'], axis=1, errors="ignore")

        #: get count of each class in each column
        for col in patient_events_table.columns:

            if col in skip_cols:
                continue

            curr_val_counts = patient_events_table[col][~pd.isnull(patient_events_table[col])].to_list()

            #: init if needed dic
            if col not in all_values:
                all_values[col] = []
                column_type[col] = {}
            
            # Add values
            all_values[col].extend(curr_val_counts)

        # free memory from current dataframe
        del patient_events_table



    #: get min, max, mean, std, variance, IQR, 25% percentile, 75% percentile
    summarized_vals = {}

    for col in all_values.keys():
        
        #: check if numeric
        summarized_vals[col] = {}

        #: determine if numeric here, once full list is available
        is_numeric = all([(isinstance(item, int) or isinstance(item, float)) and not (isinstance(item, bool)) for item in all_values[col]]) & (col not in manual_categorical_columns)


        # override for drug
        if "drug_" == col[:len("drug_")]:

            non_administered = [float(val) for val in all_values[col] if val != "administered"]

            if len(non_administered) > 0:

                # In case we have non "administered" values, turn into numeric
                is_numeric = True
                all_values[col] = non_administered  # Remove occurences of administered


            else:
                # Else keep as categorical
                is_numeric = False


        
        if is_numeric:

            summarized_vals[col]["type"] = "numeric"

            if len(all_values[col]) == 0:
                all_values[col] = [np.nan]

            summarized_vals[col]["min"] = np.nanmin(all_values[col])
            summarized_vals[col]["max"] = np.nanmax(all_values[col])
            summarized_vals[col]["mean"] = np.nanmean(all_values[col])
            summarized_vals[col]["std"] = np.nanstd(all_values[col])
            summarized_vals[col]["variance"] = np.nanvar(all_values[col])
            summarized_vals[col]["median"] = np.nanmedian(all_values[col])
            summarized_vals[col]["25_percentile"] = np.nanpercentile(all_values[col], q=25)
            summarized_vals[col]["50_percentile"] = np.nanpercentile(all_values[col], q=50)
            summarized_vals[col]["75_percentile"] = np.nanpercentile(all_values[col], q=75)
            summarized_vals[col]["IQR"] = summarized_vals[col]["75_percentile"] - summarized_vals[col]["25_percentile"]


            #: add mean/std with values after 3 sigma filtering
            all_vals = np.asarray(all_values[col])
            lower_bound = summarized_vals[col]["mean"] - (3 * summarized_vals[col]["std"])
            upper_bound = summarized_vals[col]["mean"] + (3 * summarized_vals[col]["std"])

            all_values_in_3_sigma_to_select = (all_vals <= upper_bound) & (all_vals >= lower_bound)
            all_values_in_3_sigma = all_vals[all_values_in_3_sigma_to_select]

            summarized_vals[col]["mean_3_sigma_filtered"] = np.nanmean(all_values_in_3_sigma)
            summarized_vals[col]["std_3_sigma_filtered"] = np.nanstd(all_values_in_3_sigma)

            #: get the mean and std after double 3 sigma filtering
            
            #: first clip those values
            double_3_sigma_lower_bound = summarized_vals[col]["mean_3_sigma_filtered"] - (3 * summarized_vals[col]["std_3_sigma_filtered"])
            double_3_sigma_upper_bound = summarized_vals[col]["mean_3_sigma_filtered"] + (3 * summarized_vals[col]["std_3_sigma_filtered"])
            double_3_sigma_values = np.clip(all_values_in_3_sigma, double_3_sigma_lower_bound, double_3_sigma_upper_bound)

            summarized_vals[col]["mean_double_3_sigma_filtered"] = np.nanmean(double_3_sigma_values)
            summarized_vals[col]["std_double_3_sigma_filtered"] = np.nanstd(double_3_sigma_values)

            # Do histograms
            histo = np.histogram(all_values_in_3_sigma, bins=30)
            summarized_vals[col]["30_bucket_histogram"] = [x.tolist() for x in histo]

        
        else:

            summarized_vals[col]["type"] = "categorical"

            unique, counts = np.unique(all_values[col], return_counts=True)
            ret_dict = dict(zip(unique.astype('str').tolist(), counts.tolist()))

            summarized_vals[col]["counts"] = ret_dict


    # Free memory
    del all_values

    #: save as json
    import json


    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(object, np.generic):
                return object.item()
            if np.issubdtype(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
        

    with open("/home/makaron1/dt-gpt/uc2_nsclc/2_experiments/2025_02_03_adni/1_data/1_final_data/dataset_statistics.json", "w") as outfile: 
        json.dump(summarized_vals, outfile, cls=NpEncoder, indent=4)   



    print("Finished with dataset_statistics_loader")




################################### LAUNCHER ###################################

generate_column_mapping()
dataset_statistics_loader()
