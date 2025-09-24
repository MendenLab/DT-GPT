from pathlib import Path
from datetime import datetime
import logging
import sys
import shutil
import json
import pandas as pd
import wandb
import numpy as np
import traceback
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import GPUtil
import os
from IPython.display import display
from transformers import set_seed, DataCollatorWithPadding
from pipeline.ArchivedFunctions import OldExperiment
import warnings


class Experiment:

    def __init__(self, experiment_name, experiment_class_name=None,  nickname=None, 
                    experiment_folder_root = "/pstore/data/dt-gpt/raw_experiments/uc2_nsclc/",
                    timestamp_to_use=None):
        
        if experiment_class_name is None:
            experiment_class_name = experiment_name
        
        if nickname is None:
            nickname = experiment_name

        # Do checks
        assert experiment_folder_root is not None, "Experiment: experiment_folder_root is None! This has to be set to a valid path."
        assert experiment_class_name is not None, "Experiment: experiment_class_name is None! This has to be set to a valid path."
        assert nickname is not None, "Experiment: nickname is None! This has to be set to a valid name that helps you later remember this exact run."

        # Assign Variables
        self._experiment_folder_root = experiment_folder_root
        self._experiment_class_name = experiment_class_name
        self._experiment_name = experiment_name
        self._nickname = nickname
        self._experiment_time = None
        self.plot_folder_path = None
        self.model_path = None
        self.evaluation_meta_data_path = None

        # Create basic infrastructure
        self.experiment_folder_path = self._create_experiment_folder(timestamp_to_use=timestamp_to_use)  
        self._setup_logging()

        # Some constants
        self.base_path = os.path.dirname(__file__).split("/DT-GPT")[0] + "/DT-GPT/"  # Hacky way to get the base path
        self.model_cache_path = self.base_path + "3_cache/"



    def _create_experiment_folder(self, timestamp_to_use):
        # Generate experiment folder from pipeline.Experiment_folder_root + experiment_class_name + experiment_name + datetime + nickname
        
        if timestamp_to_use is None:
            self._experiment_time = datetime.now().strftime("%Y_%m_%d___%H_%M_%S_%f")
        else:
            self._experiment_time = timestamp_to_use


        new_dir = self._experiment_folder_root + self._experiment_class_name + "/" + self._experiment_name + "/" + self._experiment_time + "/"
        Path(new_dir).mkdir(parents=True, exist_ok=True)

        # save experiment as backup
        experiment_script_to_copy = sys.modules[self.__class__.__module__].__file__
        shutil.copy(experiment_script_to_copy, new_dir + "experiment_backup.py") 

        # Create plot folder
        self.plot_folder_path = new_dir + "plots/"
        Path(self.plot_folder_path).mkdir(parents=True, exist_ok=True)

        # Create model folder
        self.model_path = new_dir + "models/"
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        # Create model folder
        self.model_folder_path = new_dir + "models/"
        Path(self.model_folder_path).mkdir(parents=True, exist_ok=True)

        # Creat evaluation_meta_data_path
        self.evaluation_meta_data_path =  new_dir + "eval_meta_data/"
        Path(self.evaluation_meta_data_path).mkdir(parents=True, exist_ok=True)
        
        return new_dir

    def get_experiment_folder(self):
        return self.experiment_folder_path
    
    def set_to_gpu_with_most_free_memory(self):
        deviceID = GPUtil.getFirstAvailable(order = 'memory', attempts=1, interval=900, verbose=False)
        device_to_use = str(deviceID[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_to_use
        logging.info("Using GPU ID: " + str(device_to_use))


    
    def _setup_logging(self):

        # Clean logging - Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.handlers = []
        
        log_file_path = self.get_experiment_folder() + "logfile.log"

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                            handlers=[
                                logging.FileHandler(log_file_path),
                                logging.StreamHandler(sys.stdout)]
                            )

        logging.info("Logger initiated")
    
    def make_new_subfolder_in_experiment_folder(self, subfolder_name):
        new_path = self.experiment_folder_path + subfolder_name
        Path(new_path).mkdir(parents=True, exist_ok=True)
        return new_path
    
    def delete_subfolder(self, full_path):
        assert full_path != self.experiment_folder_path, "Experiment: Cannot delete experiment path!"
        assert full_path not in self.experiment_folder_path, "Experiment: Cannot delete upper experiment folder paths!"
        shutil.rmtree(full_path)
    
    def get_save_path_models(self):
        new_dir = self.get_experiment_folder() + "models/"
        if not Path.is_dir(new_dir):
            Path(new_dir).mkdir(parents=True, exist_ok=True)
        return new_dir
    
    
    def save_plotnine_image_to_wandb(self, plot, name, dpi=72):

        # If saving list of plots
        if isinstance(plot, list):
            
            for i, p in enumerate(plot):
                try:
                    
                    t_path = self.plot_folder_path + name + str(i) + ".png"
                    p.save(t_path, dpi=dpi)
                    wandb.log({name : wandb.Image(t_path)})

                except Exception as e:
                    logging.info("Could not save plot to wandb")
                    traceback.print_exc()

        else:
            
            # If saving single plot
            try:
                # plotnine
                plot.save(self.plot_folder_path + name + ".png", dpi=dpi)
            except Exception as e:
                try:
                    # matplotlib
                    plot.savefig(self.plot_folder_path + name + ".png", dpi=dpi)
                except Exception as e2:
                    logging.info("Could not save plot to local folder")
                    traceback.print_exc()
                    return
            
            try:
                wandb.log({name : wandb.Image(self.plot_folder_path + name + ".png")})
            except Exception as e:
                logging.info("Could not save plot to wandb")
                traceback.print_exc()


    def save_to_wandb_final_performances(self, resulting_performances, split_name):
        
        #: save all resulting performances to run summary
        resulting_performances = {split_name : resulting_performances}  # So that split name is added to wandb dic
        wandb.run.summary.update(resulting_performances)

        #: save resulting performance also locally
        name = split_name
        json_object = json.dumps(resulting_performances, indent=4)
        meta_path = self.evaluation_meta_data_path + name + "_resulting_performances.json"
        with open(meta_path, "w") as outfile:
            outfile.write(json_object)
        wandb.config.update({name + "_resulting_performances" : meta_path}, allow_val_change=True)

        # first flatten try the dictionary resulting_performances, then print all entries in the dictionary resulting_performances which contain "overall" in the key, as a pandas dataframe
        try:
            resulting_performances_df = pd.json_normalize(resulting_performances, sep='_').filter(regex='all.*overall', axis=1)
            resulting_performances_df = resulting_performances_df.T
            resulting_performances_df.columns = ["Value"]
            resulting_performances_df.index = resulting_performances_df.index.str.replace('_', ' ')
            logging.info("Resulting performances: ")
            display(resulting_performances_df)
        except Exception as e:
            logging.info("Could not print resulting performances")
            traceback.print_exc()
        
    
    
    def save_df_targets_predictions_locally_and_statistics_to_wandb(self, split_name, full_df_targets, full_df_prediction, meta_deta_dic=None, meta_data_df=None):

        #: save to local the dics, targets and predictions
        target_path = self.evaluation_meta_data_path + split_name + "_target_dataframe.csv"
        prediction_path = self.evaluation_meta_data_path + split_name + "_prediction_dataframe.csv"

        full_df_targets.to_csv(target_path)
        full_df_prediction.to_csv(prediction_path)

        #: save to wandb the paths
        logging.info("Saved target dataframe: " + str(target_path))
        logging.info("Saved prediction dataframe: " + str(prediction_path))
        
        wandb.config.update({
            split_name + "_save_path_targets" : target_path,
            split_name + "_save_path_predictions" : prediction_path,
        }, allow_val_change=True)


        # Save meta data DF if not none
        if meta_data_df is not None:
            meta_data_df.to_csv(self.evaluation_meta_data_path + split_name + "_meta_data_df.csv")
            wandb.config.update({
                split_name + "_save_path_meta_data_df" : self.evaluation_meta_data_path + split_name + "_meta_data_df.csv"
            }, allow_val_change=True)


        #: if not none, save meta dic locally
        if meta_deta_dic is not None:
            
            meta_path = self.evaluation_meta_data_path + split_name + "_meta_data.json"
            json_object = json.dumps(meta_deta_dic, indent=4)
 
            # Writing to sample.json
            with open(meta_path, "w") as outfile:
                outfile.write(json_object)

            #: save and log path to wandb
            wandb.config.update({
                split_name + "_save_path_meta_data": meta_path
            }, allow_val_change=True)
            logging.info("Saved meta data: " + str(meta_path))


    def setup_wandb_debug_mode(self, *args):
        wandb.init(mode="disabled")

    
    def setup_wandb(self, wandb_run_name, group, sync_tensorboard=False, project='UC2 - NSCLC'):
        wandb.init(project=project, dir=self.experiment_folder_path, group=group, sync_tensorboard=sync_tensorboard)
        # change wandb.run.name
        run_nr = wandb.run.name.split("-")[2]
        wandb.run.name = str(run_nr) + " - " + wandb_run_name


    def get_output_for_split_generic_model(self, list_of_split_dfs, eval_manager, preprocessing_and_model_and_postprocessing_function):

        # Init eval manager for streaming
        eval_manager.evaluate_split_stream_start()

        # Setup cols - using all
        inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
        target_cols_raw = target_cols.copy()
        target_cols = target_cols.copy()
        target_cols.extend(["date", "patientid", "patient_sample_index"])

        # Output saving
        output_saving = {}

        input_data = []
        input_sample_ids_patient = []
        input_sample_ids_sample = []

        # : gather all data into dataset
        for idx, (constants_row, true_events_input, true_future_events_input, target_dataframe) in enumerate(list_of_split_dfs):

            if idx % 100 == 0:
                logging.info("Generating data - current idx: " + str(idx) + " / " + str(len(list_of_split_dfs)))


            patientid = constants_row["patientid"].tolist()[0]
            patient_sample_index = true_events_input["patient_sample_index"].tolist()[0]

            #: extract columns to use
            true_events_input = true_events_input.loc[:, inputs_cols]
            true_future_events_input = true_future_events_input.loc[:, future_known_inputs_cols]
            target_dataframe = target_dataframe.loc[:, target_cols]
            target_dataframe_no_empty_rows = target_dataframe.dropna(axis=0, how='all', subset=target_dataframe.columns.difference(["patientid", "patient_sample_index", "date"]))

            # Call to model
            predicted_df = preprocessing_and_model_and_postprocessing_function(constants_row, true_events_input, true_future_events_input, target_dataframe_no_empty_rows, eval_manager)

            # Convert both to numeric to target cols
            predicted_df[target_cols_raw] = predicted_df[target_cols_raw].apply(pd.to_numeric, errors='raise')
            target_dataframe_no_empty_rows[target_cols_raw] = target_dataframe_no_empty_rows[target_cols_raw].apply(pd.to_numeric, errors='raise')
            
            # send to eval manager
            eval_manager.evaluate_split_stream_prediction(predicted_df, target_dataframe_no_empty_rows, patientid, patient_sample_index)

        #: post process predictions
        logging.info("Finished generating samples")

        #: do full eval
        full_df_targets, full_df_prediction = eval_manager.concat_eval()

        # Return
        return full_df_targets, full_df_prediction
    
    def join_meta_data_to_targets(self, targets_filtered, meta_data, generated_meta_data=None):

        eval_meta_data_df = pd.DataFrame(meta_data)
        eval_targets_filtered_with_meta_data = targets_filtered.reset_index(drop=True).merge(eval_meta_data_df.reset_index(drop=True), how='left', on=["patientid", "patient_sample_index"])

        if generated_meta_data is not None:
            eval_generated_meta_data_df = pd.DataFrame(generated_meta_data)
            eval_targets_filtered_with_meta_data = eval_targets_filtered_with_meta_data.reset_index(drop=True).merge(eval_generated_meta_data_df.reset_index(drop=True), how='left', on=["patientid", "patient_sample_index"])

        return eval_targets_filtered_with_meta_data
    
    
    
    def get_output_for_split_hf_default(self, list_of_split_dfs, eval_manager, preprocessing_function, 
                                        decoding_function, post_processing_function, max_output_length, 
                                        encoding_function, 
                                        batch_size=32, 
                                        gen_num_beams=1, gen_do_sample=False, gen_penalty_alpha=None, gen_top_k=None, gen_top_p=None,
                                        num_samples_to_generate=1, sample_merging_strategy="mean",
                                        max_new_tokens=None, pad_token_id=None,
                                        output_string_filtering_function=None,
                                        return_meta_data=False,
                                        note_down_probabilities=False,
                                        tokenizer=None,
                                        verbose=False):
        
        warnings.warn("Experiment: Using HF default evaluation - we recommend using a vLLM backend - see the ADNI example!")
        warnings.warn("This is a very slow evaluation and should only be used for small datasets or debugging!")


        ############# Backwards compatibility #################
        if tokenizer is None:
            warnings.warn("No tokenizer provided - using old version of evaluation which is much slower!")

            e = OldExperiment()
            e.model = self.model

            t = e.get_output_for_split_hf_default(list_of_split_dfs=list_of_split_dfs, eval_manager=eval_manager, preprocessing_function=preprocessing_function, encoding_function=encoding_function, 
                                                    decoding_function=decoding_function, post_processing_function=post_processing_function, max_output_length=max_output_length, batch_size=batch_size,
                                                    verbose=verbose, gen_num_beams=gen_num_beams, gen_do_sample=gen_do_sample, gen_penalty_alpha=gen_penalty_alpha, gen_top_k=gen_top_k, gen_top_p=gen_top_p,
                                                    num_samples_to_generate=num_samples_to_generate, sample_merging_strategy=sample_merging_strategy,
                                                    max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                                                    output_string_filtering_function=output_string_filtering_function,
                                                    return_meta_data=return_meta_data,
                                                    note_down_probabilities=note_down_probabilities)

            warnings.warn("No tokenizer provided - using old version of evaluation which is much slower!")
            return t

        ############# Inference set up     #############
        assert self.model is not None, "Model needs to be initialized for HF eval!"
        set_seed(42)

        # Using standard model forward - as long as the model fits into 1 gpu its fine

        # Send model to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        self.model.to(device)
        logging.info("Sending model to device: " + str(device))

        # Init eval manager for streaming
        eval_manager.evaluate_split_stream_start()
        
        # Setup cols - using all
        inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
        target_cols_orginal = target_cols.copy()
        target_cols = target_cols.copy()
        inputs_cols = inputs_cols.copy()
        target_cols.extend(["date", "patientid", "patient_sample_index"])
        inputs_cols.extend(["patient_sample_index"])

        # Output saving
        output_saving = {}
        return_meta_data_list = []

        input_data = []
        input_sample_ids_patient = []
        input_sample_ids_sample = []
        output_len = []


        ############# Gather all data into dataset #############

        for idx, (constants_row, true_events_input, true_future_events_input, target_dataframe) in enumerate(list_of_split_dfs):

            logging.info("Generating data - current idx: " + str(idx + 1) + " / " + str(len(list_of_split_dfs)))

            patientid = constants_row["patientid"].tolist()[0]
            patient_sample_index = true_events_input["patient_sample_index"].tolist()[0]

            if patientid not in output_saving:
                output_saving[patientid] = {}


            #: try, fallback to empty
            try:
                
                #: extract columns to use
                true_events_input = true_events_input.loc[:, inputs_cols]
                true_future_events_input = true_future_events_input.loc[:, future_known_inputs_cols]
                target_dataframe = target_dataframe.loc[:, target_cols]

                #: convert input to string
                str_input_raw, str_output, meta_data = preprocessing_function(constants_row, true_events_input, true_future_events_input, target_dataframe, eval_manager)

                # Apply encoding function to input string
                str_input = encoding_function(str_input_raw)

                #: loop for as many samples as needed
                for sample_idx in range(num_samples_to_generate):
                    #: save to datasets
                    input_data.append(str_input)
                    input_sample_ids_patient.append(patientid)
                    input_sample_ids_sample.append(patient_sample_index)
                    output_len.append(len(str_output))

                #: save meta data
                output_saving[patientid][patient_sample_index] = {
                    "meta_data" : meta_data,
                    "input_data" : str_input,
                    "labels" : str_output,
                    "raw_input" : str_input_raw,
                }
            
            except:
                traceback.print_exc()
                logging.info("Error in generating data!")

                #: make empty input ids
                input_data.append("Error")
                input_sample_ids_patient.append(patientid)
                input_sample_ids_sample.append(patient_sample_index)

                output_saving[patientid][patient_sample_index] = {
                    "meta_data" : None,
                    "input_data" : "Error",
                    "labels" : "Error",
                    "raw_input": "Error",
                }
            
            # Make empty fallback df
            empty_target_dataframe = target_dataframe.copy()
            empty_target_dataframe.loc[:, [col for col in empty_target_dataframe.columns if col not in ["date", "patientid", "patient_sample_index"]]] = np.nan
            output_saving[patientid][patient_sample_index]["empty_target_df"] = empty_target_dataframe

            # Make target DF, with dropping na rows
            target_df = target_dataframe.copy()
            target_df = target_df.dropna(axis=0, how='all', subset=target_df.columns.difference(["patientid", "patient_sample_index", "date"]))
            output_saving[patientid][patient_sample_index]["target_df"] = target_df

            # Make where to store the generated predictions
            output_saving[patientid][patient_sample_index]["generated_predictions"] = []
            output_saving[patientid][patient_sample_index]["generated_scores"] = []




        ############# Setup dataset #############
            
        input_sample_ids_patient = np.asarray(input_sample_ids_patient)
        input_sample_ids_sample = np.asarray(input_sample_ids_sample)
        
        curr_dataset = Dataset.from_dict({"input_text": input_data,
                                          "label_ids": list(range(len(input_data))),
                                          "output_lens": output_len})

        curr_dataset = curr_dataset.with_format("torch")

        #: make dataset ordered for similar output lengths to optimize for speed, and also to make the first sample processed the longest
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        
        
        curr_dataset = curr_dataset.sort('output_lens', reverse=True)   # Sort with longest first
        curr_dataset = curr_dataset.map(lambda examples: tokenizer(examples["input_text"], truncation=True, max_length=max_output_length), batched=True)  # Tokenize the text
        curr_dataset = curr_dataset.remove_columns("input_text")
        curr_dataset = curr_dataset.remove_columns("output_lens")

        
        dataloader = DataLoader(curr_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

        curr_index = 0

        ############# Generate samples #############
        for batch in dataloader:

            logging.info("Batch starting with index: " + str(curr_index + 1) + " / " + str(len(input_data)))
            
            #: get sample id
            patientids = input_sample_ids_patient[batch["labels"]]
            patient_sample_indices = input_sample_ids_sample[batch["labels"]]

            #: batch send to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            model_scores = None
            
            if note_down_probabilities:
                
                outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                do_sample=gen_do_sample, num_beams=gen_num_beams,
                                                penalty_alpha=gen_penalty_alpha, top_k=gen_top_k,
                                                top_p=gen_top_p,
                                                max_new_tokens=max_new_tokens, max_length=max_output_length + 92, 
                                                pad_token_id=pad_token_id,
                                                return_dict_in_generate=True, output_scores=True)
                
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )

                #: setup "predictions"
                predictions = outputs.sequences
                
                #: setup model scores
                model_scores = transition_scores
                model_scores = model_scores.cpu().numpy()
                model_scores = np.exp(model_scores)

            else:
                predictions = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                    do_sample=gen_do_sample, num_beams=gen_num_beams,
                                                    penalty_alpha=gen_penalty_alpha, top_k=gen_top_k,
                                                    top_p=gen_top_p,
                                                    max_new_tokens=max_new_tokens, max_length=max_output_length + 92, 
                                                    pad_token_id=pad_token_id)

            #: batch decode using tokenizer
            predictions_decoded = decoding_function(predictions)

            #: save in temp dic for merging for each individual sample
            for prediction_idx, prediction_str in enumerate(predictions_decoded):

                curr_patientid = patientids[prediction_idx] if not isinstance(patientids, str) else patientids
                curr_patient_sample_index = patient_sample_indices[prediction_idx] if not isinstance(patient_sample_indices, str) else patient_sample_indices

                output_saving[curr_patientid][curr_patient_sample_index]["generated_predictions"].append(prediction_str)

                if note_down_probabilities:
                    output_saving[curr_patientid][curr_patient_sample_index]["generated_scores"].append(model_scores[prediction_idx])

                if verbose and curr_index % 1000 == 0:
                    logging.info("Raw Input: " + str(output_saving[curr_patientid][curr_patient_sample_index]["raw_input"]))
                    logging.info("Input: " + str(output_saving[curr_patientid][curr_patient_sample_index]["input_data"]))
                    logging.info("Labels: " + str(output_saving[curr_patientid][curr_patient_sample_index]["labels"]))
                    logging.info("Raw Prediction: " + str(prediction_str))


                curr_index += 1

        total_samples = len(input_data)


        ####################### Conversion & Merging ##############################################
            
        logging.info("Finished generating samples! Now converting and merging")
        meta_data = {}
        curr_index = 0

        for patientid in output_saving.keys():
            for patient_sample_index in output_saving[patientid].keys():
                
                merging_list = []

                #: go over all predicted samples
                for idx, prediction_string in enumerate(output_saving[patientid][patient_sample_index]["generated_predictions"]):

                    # Apply filter function
                    good_sample = True
                    if output_string_filtering_function is not None:
                        good_sample = output_string_filtering_function(prediction_string)

                    try: 
                        #: get patientid and patient_sample_index
                        logging.info("Evaluating split - current idx: " + str((patientid, patient_sample_index)) + " " + str(curr_index + 1) + " / " + str(len(list_of_split_dfs)))

                        # Post process and convert to DF
                        predicted_df = post_processing_function(prediction_string, patientid, patient_sample_index, meta_data=output_saving[patientid][patient_sample_index]["meta_data"])
                    
                    except KeyboardInterrupt:
                        print("Keyboard interrupt!")
                        return None, None
                    
                    except Exception:
                        # Fallback to empty DF in case of any errors
                        traceback.print_exc()
                        predicted_df = output_saving[patientid][patient_sample_index]["empty_target_df"]
                        good_sample = False  # Filter out bad samples
                        logging.info("Falling back to empty DF.")

                    #: save for later meta data
                    if patientid not in meta_data:
                        meta_data[patientid] = {}
                    if patient_sample_index not in meta_data[patientid]:
                        meta_data[patientid][patient_sample_index] = {"probability_score" : []}

                    # Save to merging list
                    merging_list.append((good_sample, predicted_df))

                    if note_down_probabilities:
                        meta_data[patientid][patient_sample_index]["probability_score"].append(output_saving[curr_patientid][curr_patient_sample_index]["generated_scores"][idx])
                

                #: go over all predicted samples and merge
                final_merged_df = None

                
                #: filter out poor samples - if only poor samples then keep all of them
                interesting_samples = [x for x in merging_list if x[0]]
                
                if len(interesting_samples) != len(merging_list):
                    logging.info("Skipping bad samples for this prediction - resulting nr of samples to use: " + str(len(interesting_samples)))

                if len(interesting_samples) == 0:
                    logging.info("No good output samples for this prediction - using all bad samples for final predictions")
                    interesting_samples = [x for x in merging_list]

                #: merge interesting columns target_cols_orginal
                interesting_df_cols = [np.expand_dims(x[1][target_cols_orginal].values, axis=2) for x in interesting_samples]
                merged_np = np.concatenate(interesting_df_cols, axis=2)
                merged_np = merged_np.astype(np.float32)

                #: merge using correct strategy
                if sample_merging_strategy == "mean":
                    aggregated_np = np.nanmean(merged_np, axis=2, keepdims=False)
                elif sample_merging_strategy == "50th percentile":
                    aggregated_np = np.percentile(merged_np, 50, axis=2, keepdims=False)
                else:
                    raise Exception("Experiment: unknown sample_merging_strategy provided!")

                #: insert back into final dataframe
                final_df = merging_list[0][1].copy()
                final_df[target_cols_orginal] = aggregated_np
                final_merged_df = final_df

                #: save to meta data as list, if needed
                if return_meta_data:
                    return_meta_data_list.append({
                        "patientid": str(patientid),
                        "patient_sample_index": str(patient_sample_index),
                        "generated_trajectories": [x[1].values.tolist() for x in merging_list if x[0]],
                        "used_trajectories": [x[1].values.tolist() for x in interesting_samples],
                        "model_scores" : meta_data[patientid][patient_sample_index]["probability_score"] if note_down_probabilities else None,
                        })

                #: send merged DF to eval manager
                eval_manager.evaluate_split_stream_prediction(final_merged_df, output_saving[patientid][patient_sample_index]["target_df"], patientid, patient_sample_index)
                curr_index += 1


        ####################### post process predictions #######################
        logging.info("Finished generating samples")

        #: do full eval
        full_df_targets, full_df_prediction = eval_manager.concat_eval()

        #: check return_meta_data and return as needed
        if return_meta_data:
            return full_df_targets, full_df_prediction, return_meta_data_list
        else:
            # Return
            return full_df_targets, full_df_prediction



    








