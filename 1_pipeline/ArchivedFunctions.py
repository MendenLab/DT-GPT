from pathlib import Path
from datetime import datetime
import logging
import sys
import shutil
import json
import pandas as pd
import wandb
from PIL import Image
import numpy as np
import traceback
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import GPUtil
import os
from IPython.display import display
from transformers import set_seed, DataCollatorWithPadding
import plotnine
import matplotlib




# Used for deprecated functions but needed for backwards compatibility

class OldExperiment():

    def get_output_for_split_hf_default(self, list_of_split_dfs, eval_manager, preprocessing_function, encoding_function, 
                                        decoding_function, post_processing_function, max_output_length, batch_size=32,
                                        verbose=False, gen_num_beams=1, gen_do_sample=False, gen_penalty_alpha=None, gen_top_k=None, gen_top_p=None,
                                        num_samples_to_generate=1, sample_merging_strategy="mean",
                                        max_new_tokens=None, pad_token_id=None,
                                        output_string_filtering_function=None,
                                        return_meta_data=False,
                                        note_down_probabilities=False):
    
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

        # : gather all data into dataset
        for idx, (constants_row, true_events_input, true_future_events_input, target_dataframe) in enumerate(list_of_split_dfs):

            logging.info("Generating data - current idx: " + str(idx) + " / " + str(len(list_of_split_dfs)))


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
                str_input, str_output, meta_data = preprocessing_function(constants_row, true_events_input, true_future_events_input, target_dataframe, eval_manager)

                #: save to datasets
                input_data.append(str_input)
                input_sample_ids_patient.append(patientid)
                input_sample_ids_sample.append(patient_sample_index)

                #: save meta data
                output_saving[patientid][patient_sample_index] = {
                    "meta_data" : meta_data,
                    "input_data" : str_input,
                    "labels" : str_output
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
                    "labels" : "Error"
                }
            
            # Make empty fallback df
            empty_target_dataframe = target_dataframe.copy()
            empty_target_dataframe.loc[:, [col for col in empty_target_dataframe.columns if col not in ["date", "patientid", "patient_sample_index"]]] = np.nan
            output_saving[patientid][patient_sample_index]["empty_target_df"] = empty_target_dataframe

            # Make target DF
            output_saving[patientid][patient_sample_index]["target_df"] = target_dataframe.copy()


        #: convert to dataset
        curr_dataset = Dataset.from_dict({"input_text": input_data, "patient_ids" : input_sample_ids_patient, "patient_sample_index" : input_sample_ids_sample})
        curr_dataset = curr_dataset.with_format("torch")
    
        dataloader = DataLoader(curr_dataset, batch_size=batch_size)
        curr_index = 0

        for batch in dataloader:

            logging.info("Batch starting with index: " + str(curr_index))
            
            #: get sample id
            patientids = batch["patient_ids"]
            patient_sample_indices = batch["patient_sample_index"]

            #: batch encode
            input_text = batch["input_text"]                    
            input_ids = encoding_function(input_text)

            #: send to device
            input_ids = input_ids.to(device)

            #: use num_samples_to_generate to get multuple predictions
            predictions_decoded = []
            merging_dic = {}
            meta_data = {}

            for sample_idx in range(num_samples_to_generate):

                #: get predictions for all of them - prob need input_ids["input_ids"]
                model_scores = None
                
                if note_down_probabilities:
                    
                    outputs = self.model.generate(**input_ids, 
                                                do_sample=gen_do_sample, num_beams=gen_num_beams,
                                                penalty_alpha=gen_penalty_alpha, top_k=gen_top_k,
                                                top_p=gen_top_p,
                                                max_new_tokens=max_new_tokens, max_length=max_output_length, 
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
                    predictions = self.model.generate(**input_ids, 
                                                        do_sample=gen_do_sample, num_beams=gen_num_beams,
                                                        penalty_alpha=gen_penalty_alpha, top_k=gen_top_k,
                                                        top_p=gen_top_p,
                                                        max_new_tokens=max_new_tokens, max_length=max_output_length, 
                                                        pad_token_id=pad_token_id)

                #: batch decode using tokenizer
                predictions_decoded_curr = decoding_function(predictions)
                predictions_decoded.append(predictions_decoded_curr)


                #: go over all predicted samples
                for idx, prediction_string in enumerate(predictions_decoded[sample_idx]):
                    
                    patientid, patient_sample_index = patientids[idx], patient_sample_indices[idx]

                    # Apply filter function
                    good_sample = True
                    if output_string_filtering_function is not None:
                        good_sample = output_string_filtering_function(prediction_string)

                    try: 
                        #: get patientid and patient_sample_index
                        logging.info("Evaluating split - current idx: " + str((patientid, patient_sample_index)))

                        # Post process and convert to DF
                        predicted_df = post_processing_function(prediction_string, patientid, patient_sample_index, meta_data=output_saving[patientid][patient_sample_index]["meta_data"])

                        #: print every so often
                        if curr_index % 200 == 0 or verbose:
                            logging.info("Input: " + str(output_saving[patientid][patient_sample_index]["input_data"]))
                            logging.info("Labels: " + str(output_saving[patientid][patient_sample_index]["labels"]))
                            logging.info("Raw Prediction: " + str(prediction_string))
                            logging.info("No tokenizer provided - using old version of evaluation which is much slower!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    
                    except KeyboardInterrupt:
                        print("Keyboard interrupt!")
                        return None, None
                    
                    except Exception:
                        # Fallback to empty DF in case of any errors
                        traceback.print_exc()
                        predicted_df = output_saving[patientid][patient_sample_index]["empty_target_df"]
                        good_sample = False  # Filter out bad samples
                        logging.info("Falling back to empty DF.")

                    #: save for later merging
                    if patientid not in merging_dic:
                        merging_dic[patientid] = {}
                        meta_data[patientid] = {}
                    if patient_sample_index not in merging_dic[patientid]:
                        merging_dic[patientid][patient_sample_index] = []
                        meta_data[patientid][patient_sample_index] = {"probability_score" : []}
                    merging_dic[patientid][patient_sample_index].append((good_sample, predicted_df))

                    if note_down_probabilities:
                        meta_data[patientid][patient_sample_index]["probability_score"].append(model_scores[idx])

                    

            #: go over all predicted samples and merge
            final_merged_df = {}

            for patientid in merging_dic:
                for patient_sample_index in merging_dic[patientid]:

                    #: filter out poor samples - if only poor samples then keep all of them
                    interesting_samples = [x for x in merging_dic[patientid][patient_sample_index] if x[0]]
                    
                    if len(interesting_samples) != len(merging_dic[patientid][patient_sample_index]):
                        logging.info("Skipping bad samples for this prediction - resulting nr of samples to use: " + str(len(interesting_samples)))

                    if len(interesting_samples) == 0:
                        logging.info("No good output samples for this prediction - using all bad samples for final predictions")
                        interesting_samples = [x for x in merging_dic[patientid][patient_sample_index]]

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
                    final_df = merging_dic[patientid][patient_sample_index][0][1].copy()
                    final_df[target_cols_orginal] = aggregated_np
                    final_merged_df[patientid] = {patient_sample_index : final_df}

                    #: save to meta data as list, if needed
                    if return_meta_data:
                        return_meta_data_list.append({
                            "patientid": patientid,
                            "patient_sample_index": patient_sample_index,
                            "generated_trajectories": [x[1].values.tolist() for x in merging_dic[patientid][patient_sample_index] if x[0]],
                            "used_trajectories": [x[1].values.tolist() for x in interesting_samples],
                            "model_scores" : meta_data[patientid][patient_sample_index]["probability_score"] if note_down_probabilities else None,
                            })

            #: send merged DF to eval manager
            for patientid in final_merged_df:
                for patient_sample_index in final_merged_df[patientid]:
                    final_predicted_df = final_merged_df[patientid][patient_sample_index]
                    eval_manager.evaluate_split_stream_prediction(final_predicted_df, output_saving[patientid][patient_sample_index]["target_df"], patientid, patient_sample_index)
                    curr_index += 1

        #: post process predictions
        logging.info("Finished generating samples")

        #: do full eval
        full_df_targets, full_df_prediction = eval_manager.concat_eval()

        #: check return_meta_data and return as needed
        if return_meta_data:
            return full_df_targets, full_df_prediction, return_meta_data_list
        else:
            # Return
            return full_df_targets, full_df_prediction















