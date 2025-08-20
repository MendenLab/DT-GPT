import logging
import concurrent.futures
from joblib import Parallel, delayed
from tqdm import tqdm


def process_all_tuples(list_of_data_tuples, conversion_function):

    # Setup
    list_input_strings = []
    list_target_strings = []
    list_meta_data = []
    
    for idx, data in enumerate(list_of_data_tuples):

        # log
        if idx % 10 == 0:
            logging.info("Converting DFs to Strings: " + str(idx) + " / " + str(len(list_of_data_tuples)))

        # Do actual conversion
        string_input, string_output, meta_data = conversion_function(*data)

        # Save output
        list_input_strings.append(string_input)
        list_target_strings.append(string_output)
        list_meta_data.append(meta_data)

    return list_input_strings, list_target_strings, list_meta_data



def process_all_tuples_multiprocessing(list_of_data_tuples, conversion_function, n_jobs=-1):

    # Setup
    results = Parallel(n_jobs=n_jobs)(delayed(conversion_function)(*i) for i in list_of_data_tuples)
    
    # unpack for returning
    list_input_strings, list_target_strings, list_meta_data = zip(*results)

    return list_input_strings, list_target_strings, list_meta_data




