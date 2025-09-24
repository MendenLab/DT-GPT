import numpy as np

# This script generates all necessary meta data in one go
# used to be multiple scripts but merged for simplicity




print("THIS SCRIPT HAS TO BE RUN FROM THE DATA DIRECTORY!")





################################### generate_column_mapping.py -  GENERATES column_mapping.json FILE ###################################

def generate_column_mapping():
    print("Starting with generate_column_mapping")

    import json
    import pandas as pd


    # get an example events file for the columns
    example_events_df = pd.read_csv("./patient_events/1.csv")


    # drop first columns as they are junk
    example_events_df = example_events_df.drop(example_events_df.columns[[0]],axis = 1)



    # Add info whether it is input, target or known_future_input
    target_columns = ["lab_718_7", "lab_26464_8", "lab_26478_8", "lab_26474_7", "lab_26499_4", "lab_2532_0"]   # Critical vars
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
    with open('./column_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=4)


    print("Finished with generate_column_mapping")


################################### dataset_statistics_loader.py -  GENERATES dataset_statistics.json FILE ###################################


def dataset_statistics_loader():

    print("Starting with dataset_statistics_loader")
    
    import pandas as pd
    from pipeline.EvaluationManager import EvaluationManager
    import numpy as np
    from pandas.api.types import is_numeric_dtype
    import time


    #: load up eval manager
    eval_manager = EvaluationManager("2024_02_05_critical_vars", load_statistics_file=False)  # Do not load the statistics file, since we're building it now


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
        

    with open("/home/makaron1/uc2_nsclc/2_experiments/2024_02_05_critical_vars/1_data/dataset_statistics.json", "w") as outfile: 
        json.dump(summarized_vals, outfile, cls=NpEncoder, indent=4)   



    print("Finished with dataset_statistics_loader")


################################### make_random_patient_subsets.py -  GENERATES RANDOM PATIENT SUBSETS ###################################


def make_random_patient_subsets():

    print("Starting with make_random_patient_subsets")

    import pandas as pd
    import numpy as np
    import random
    import json


    random.seed(42)


    def select_subset(split_to_select, amount_to_select, file_name):



        constants = pd.read_csv("./constant.csv")


        constants_split = constants[constants["dataset_split"] == split_to_select]


        subset_patientids = constants_split["patientid"].tolist()

        random.shuffle(subset_patientids)

        subset_patientids_selected = subset_patientids[0:amount_to_select]


        save_dic = {
            "patientids": subset_patientids_selected
        }


        # Save as json
        with open('./patient_subsets/' + file_name +'.json', 'w') as f:
            json.dump(save_dic, f, indent=4)



    select_subset("TRAIN", 1000, "2023_11_08_1k_train")
    select_subset("VALIDATION", 100, "2023_11_08_100_validation")
    select_subset("TEST", 100, "2023_11_08_100_test")



    print("Finished with make_random_patient_subsets")


################################### mapping_file_generator_nr_tokens_estimated.py -  NR OF TOKENS IN DESCRIPTIVE MAPPING ###################################


def mapping_file_generator_nr_tokens_estimated():

    print("Starting with mapping_file_generator_nr_tokens_estimated")


    from transformers import LlamaTokenizer
    import pandas as pd
    import numpy as np



    original_column_mapping = pd.read_csv("/home/makaron1/uc2_nsclc/2_experiments/2024_02_05_critical_vars/1_data/column_descriptive_name_mapping.csv")

    #: make updated column names - since they were from master.events_long originally
    if original_column_mapping["original_column_names"].tolist()[0][:len(original_column_mapping["group"].tolist()[0])] != original_column_mapping["group"].tolist()[0]:
        original_column_mapping["original_column_names"] = original_column_mapping["group"] + "_" + original_column_mapping["original_column_names"] 

    tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", truncation_side="left")


    new_column_mapping = original_column_mapping.copy()

    new_column = []

    for idx in range(new_column_mapping.shape[0]):

        curr_column_descriptive_name = new_column_mapping.iloc[idx, 3]

        tokens = tokenizer(text=curr_column_descriptive_name)["input_ids"]
        nr_tokens = len(tokens) - 1  # -1 since there is end of line
        new_column.append(nr_tokens)


    new_column_mapping["nr_tokens"] = new_column

    # Drop  bad column
    new_column_mapping = new_column_mapping.drop(['Unnamed: 0', 'X.2', 'X.1', 'X'], axis=1, errors="ignore")

    # save
    new_column_mapping.to_csv("/home/makaron1/uc2_nsclc/2_experiments/2024_02_05_critical_vars/1_data/column_descriptive_name_mapping.csv")




    print("Finished with mapping_file_generator_nr_tokens_estimated")




################################### MAKING DUPLICATED DESCRIPTIVE NAMING WITH NUMBERS ###################################





def duplicate_naming_increment():

    print("Starting  MAKING DUPLICATED DESCRIPTIVE NAMING WITH NUMBERS ")

    import pandas as pd
    import numpy as np

    original_column_mapping = pd.read_csv("/home/makaron1/uc2_nsclc/2_experiments/2024_02_05_critical_vars/1_data/column_descriptive_name_mapping.csv")

    # Create a new column that enumerates the duplicates
    counts = original_column_mapping.groupby('descriptive_column_name').cumcount() + 1

    # : skip if all counts are 1
    if all(counts == 1):
        print("Skipping since all counts are 1")
        return

    # Only append a number if there is more than one occurrence
    original_column_mapping['descriptive_column_name'] = original_column_mapping['descriptive_column_name'] + " " + counts.where(counts > 1, '').astype(str)

    #: remove edge whitespace
    original_column_mapping['descriptive_column_name'] = original_column_mapping['descriptive_column_name'].str.rstrip()

    # Drop  bad columns
    new_column_mapping = original_column_mapping.drop(['Unnamed: 0', 'X.2', 'X.1', 'X'], axis=1, errors="ignore")

    # save
    new_column_mapping.to_csv("/home/makaron1/uc2_nsclc/2_experiments/2024_02_05_critical_vars/1_data/column_descriptive_name_mapping.csv")

    print("Finished MAKING DUPLICATED DESCRIPTIVE NAMING WITH NUMBERS")





################################### LAUNCHER ###################################

generate_column_mapping()
dataset_statistics_loader()
make_random_patient_subsets()
duplicate_naming_increment()
mapping_file_generator_nr_tokens_estimated()



