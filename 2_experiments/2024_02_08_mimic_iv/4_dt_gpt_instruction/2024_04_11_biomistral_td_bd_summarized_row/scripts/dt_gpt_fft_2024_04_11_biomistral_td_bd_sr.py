import __init__
from EvaluationManager import EvaluationManager
from Experiment import Experiment
import wandb
import pandas as pd
import logging
from DataFrameConvertTDBDMIMIC import DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC
from DFConversionHelpers import process_all_tuples_multiprocessing
from DataProcessorBiomistral import DataProcessorBiomistral
from NormalizationFilterManager import Only_Double3_sigma_Filtering
import torch
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
import gc
from NormalizationFilterManager import Only_Double3_sigma_Filtering
from MetricManager import MetricManager
from trl import SFTTrainer
import json
from Splitters import After24HSplitter
from NormalizationFilterManager import Only_Standardization



class DTGPT_mimic_biomistral_fft_ti_bd_sr:

    def run(self, debug=False, verbose=True, 
                wandb_prefix_name="DT-GPT - Meditron FFT - Completion Loss - Forecast: ", 
                wandb_group_name="DT-GPT - Meditron FFT - Completion",
                train_set= "TRAIN", validation_set = "VALIDATION", test_set = "TEST",
                learning_rate=1e-5, batch_size_training=1, batch_size_validation=1, weight_decay=0.1, gradient_accumulation=1,
                num_train_epochs=1.0, eval_interval=0.25, warmup_ratio=0.1, lr_scheduler="cosine", gradient_checkpointing=False,
                logging_steps=10,
                nr_days_forecasting=91, seq_max_len_in_tokens=4000, decimal_precision=1,
                gen_num_beams=1, gen_do_sample=False,
                eval_model_path=None,
                num_samples_to_generate=10, sample_merging_strategy="mean",
                max_new_tokens_to_generate=1200):

        
        ######################################################## SETUP EXPERIMENT ########################################################
        
        MIN_NR_DAYS_FORECAST = 24   # We want to forecast up to the first visit after this value, or until the start of the next therapy (which ever comes first) - using 91 since it is the closest multiple of 7 to 90 days - often used for meds
        SEQUENCE_MAX_LENGTH_IN_TOKENS = seq_max_len_in_tokens
        MODEL_HF_NAME = "BioMistral/BioMistral-7B-DARE"
        DECIMAL_PRECISION = decimal_precision

        # Setup hyperparameters
        LEARNING_RATE = learning_rate                # From meditron paper (pretraining setting)
        BATCH_SIZE_TRAINING = batch_size_training             # For debugging purposes
        BATCH_SIZE_VALIDATION = batch_size_validation
        WEIGHT_DECAY = weight_decay                  # From meditron paper
        GRADIENT_ACCUMULATION = gradient_accumulation

        NUM_TRAIN_EPOCHS = num_train_epochs              # 10% of training data
        EVAL_NUM_STEPS = eval_interval               # So 4 times in one training run

        WARMUP_RATIO = warmup_ratio                  # From meditron paper
        LR_SCHEDULER_TYPE = lr_scheduler        # From meditron paper


        eval_manager = EvaluationManager("2024_03_15_mimic_iv")

        experiment = Experiment("setup")

        # Uncomment for debug mode of WandB
        if debug:
            experiment.setup_wandb_debug_mode()
        else:
            experiment.setup_wandb(wandb_prefix_name + str(MIN_NR_DAYS_FORECAST) + " LR: " + str(learning_rate), wandb_group_name, project="UC - MIMIC-IV")
            
            wandb.config.update({"generation_config": { "gen_num_beams": gen_num_beams,
                                    "gen_do_sample": gen_do_sample}}, 
                                    allow_val_change=True)


        ############################################################ Load & Split Data ############################################################


        training_full_paths, training_full_patientids = eval_manager.get_paths_to_events_in_split(train_set)
        validation_full_paths, validation_full_patientids = eval_manager.get_paths_to_events_in_split(validation_set)
        test_full_paths, test_full_patientids = eval_manager.get_paths_to_events_in_split(test_set)

        # Load data
        training_full_constants, training_full_events = eval_manager.load_list_of_patient_dfs_and_constants(training_full_patientids)
        validation_full_constants, validation_full_events = eval_manager.load_list_of_patient_dfs_and_constants(validation_full_patientids)
        test_full_constants, test_full_events = eval_manager.load_list_of_patient_dfs_and_constants(test_full_patientids)

        # Setup splitter object
        splitter = After24HSplitter()
        
        if eval_model_path is None:
            training_events, training_meta_data = splitter.setup_split_indices(training_full_events, eval_manager)
        
        # Setup also validation and test
        validation_events, validation_meta = splitter.setup_split_indices(validation_full_events, eval_manager)

        test_events, test_meta = splitter.setup_split_indices(test_full_events, eval_manager)
        
        path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/dataset_statistics.json"
        with open(path_to_statistics_file) as f:
            statistics_dic = json.load(f)
        

        ######################################################## CONVERT DFS TO STRINGS ########################################################

        logging.info("Converting DFs to Strings")

        def filtering_rows_rest_budget(df, nr_tokens_budget):
            

            #: create summarizing row for every variable for its respective last observed value, and put into it first row

            # Forward fill the missing values
            df_filled = df.ffill()
            # Take the first row (which corresponds to the last non-NaN observation in the original DataFrame)
            last_valid_values = df_filled.iloc[-1]
            # Assign these values to the first row of the original DataFrame
            df.iloc[0] = last_valid_values
            # If you want to maintain the original index, you can do the following:
            df.loc[df.index[0]] = last_valid_values
            # set to nan if these values are seen in last row (this is just a simple heuristic to remove some tokens)
            first_row_label = df.index[0]
            original_basics = df.loc[first_row_label, ["date", "patientid", "patient_sample_index"]].copy()
            mask = df.iloc[0] == df.iloc[-1]
            df.loc[first_row_label, mask] = pd.NA

            #: adjust budgets of row --> 6 tokens per non na value
            budget_of_first_row = 6 * df.iloc[0].notna().sum()
            df.loc[first_row_label, "estimated_nr_tokens"] = budget_of_first_row

            # Add back in basics
            df.loc[first_row_label, ["date", "patientid", "patient_sample_index"]] = original_basics
            df.loc[first_row_label, "date"] = -1  # So that model gets it quicker


            # Add in buffer to ensure we don't go over the budget
            buffer = 300
            nr_tokens_budget = max(nr_tokens_budget - buffer, 0)

            df_reversed = df.iloc[::-1].copy()
            # Calculate the cumulative sum of the estimated_nr_tokens column
            df_reversed['cumulative_tokens'] = df_reversed['estimated_nr_tokens'].cumsum()

            #: ensure summarizing row is kept in
            df_reversed.loc[first_row_label, "cumulative_tokens"] = 0

            # Filter rows where the cumulative sum is less than or equal to nr_tokens_budget
            df_filtered = df_reversed[df_reversed['cumulative_tokens'] <= nr_tokens_budget]
            # Reverse the DataFrame back to the original order
            df_result = df_filtered.iloc[::-1]
            # drop cumulative tokens column
            df_result = df_result.drop(columns=['cumulative_tokens'])
            return df_result



        # Setup column mapping
        column_mapping = pd.read_csv(experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/column_descriptive_name_mapping.csv")

        # Setup statistics paths
        path_to_statistics_file = experiment.base_path + "2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/dataset_statistics.json"
        with open(path_to_statistics_file) as f:
            statistics_dic = json.load(f)

        # Setup conversion function
        prompt = "Please predict for the previously noted down hours the lab variables for this intensive care unit patient, in a JSON format as in the patient history, with the same ordering of variables as in the 'Variables to predict' section."


        # Setup constant column mapping {<original_colunn_name>: <descriptive_column_name>}
        constant_column_mapping = {}

        # Get random constant row
        constant_row = training_full_constants[0]
        constant_columns = constant_row.columns.tolist()
        # Get mapping for indications
        for col in constant_columns:
            match = column_mapping[column_mapping["original_column_names"] == col]
            if match.shape[0] > 0:
                constant_column_mapping[col] = match["descriptive_column_name"].values[0]

        # add basics
        constant_column_mapping["Age"] = "age"
        constant_column_mapping["gender"] = "gender"
        constant_column_mapping["ethnicity"] = "ethnicity"
        constant_column_mapping["insurance"] = "insurance"


        conversion_function = DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_df_to_strings

        if eval_model_path is None:
            # set up all tuples to be used as arguments for the conversion function
            training_events = [(column_mapping, curr_const_row, true_events_input, true_future_events_input, target_dataframe, filtering_rows_rest_budget, SEQUENCE_MAX_LENGTH_IN_TOKENS, DECIMAL_PRECISION, prompt) 
                                for curr_const_row, true_events_input, true_future_events_input, target_dataframe in training_events]


            #: apply filtering of training data to remove bad outliers
            training_norm_filter = Only_Double3_sigma_Filtering(path_to_statistics_file)

            #: use the same filtering function but apply to training data
            training_events = [(x[0], x[1], 
                                    training_norm_filter.normalize_and_filter(x[2].copy(), None, replace_nan_rows=False, replace_missing_in_prediction=False, verbose=False, specific_column_list=["lab_26499_4"])[0], 
                                    x[3], 
                                    training_norm_filter.normalize_and_filter(x[4].copy(), None, replace_nan_rows=False, replace_missing_in_prediction=False, verbose=False)[0], 
                                    x[5], x[6], x[7], x[8], constant_column_mapping) for x in training_events]

            # apply multiprocessing based DF to string conversion to speed up process            
            training_input_strings, training_target_strings, training_meta_data = process_all_tuples_multiprocessing(training_events, conversion_function)

            # Print one example
            logging.info("Example of input: " + training_input_strings[0])
            logging.info("Example of target: " + training_target_strings[0])
        
            logging.info("Example of input: " + training_input_strings[1])
            logging.info("Example of target: " + training_target_strings[1])


            # apply for validation data
            validation_events_for_tokenization = [(column_mapping, curr_const_row, true_events_input, true_future_events_input, target_dataframe, filtering_rows_rest_budget, SEQUENCE_MAX_LENGTH_IN_TOKENS, DECIMAL_PRECISION, prompt)
                                                    for curr_const_row, true_events_input, true_future_events_input, target_dataframe in validation_events]

            #: use the same filtering function but apply to training data
            validation_events_for_tokenization = [(x[0], x[1], 
                                                        training_norm_filter.normalize_and_filter(x[2].copy(), None, replace_nan_rows=False, replace_missing_in_prediction=False, verbose=False, specific_column_list=["lab_26499_4"])[0], 
                                                        x[3], 
                                                        training_norm_filter.normalize_and_filter(x[4].copy(), None, replace_nan_rows=False, replace_missing_in_prediction=False, verbose=False)[0], 
                                                        x[5], x[6], x[7], x[8], constant_column_mapping) for x in validation_events_for_tokenization]
            
            validation_input_strings, validation_target_strings, validation_meta_data = process_all_tuples_multiprocessing(validation_events_for_tokenization, conversion_function)

                
        ######################################################## SETUP DATA PROCESSOR ########################################################


        # Load data processor
        logging.info("Setting up data processor")

        dp = DataProcessorBiomistral(experiment, path_to_statistics_file, column_mapping, 
                                    model_to_use=MODEL_HF_NAME, max_total_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                                    collator_setting="completion")
        dp.set_converter(DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC)
        dp.set_for_training()

        target_cols = eval_manager.get_column_usage()[2]
        dp.setup_cols(target_cols)
        
        if eval_model_path is None:
            tokenize = True
            training_dataset = dp.preprocess_dataset(training_input_strings, training_target_strings, tokenize=tokenize)

            # Tokenize validation dataset
            validation_dataset = dp.preprocess_dataset(validation_input_strings, validation_target_strings, tokenize=tokenize)

            
            ######################################################## SETUP MODEL ########################################################

            logging.info("Setting up model")

            
            # Load in Meditron
            model = AutoModelForCausalLM.from_pretrained(MODEL_HF_NAME, 
                                                        cache_dir=experiment.model_cache_path,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map="auto", 
                                                        attn_implementation="flash_attention_2")

            logging.info("Num params in model: " + str(model.num_parameters()))

            # Load data collator
            data_collator = dp.get_collator(model)

            
            train_params = TrainingArguments(
                output_dir=experiment.model_path,
                per_device_train_batch_size=BATCH_SIZE_TRAINING,
                per_device_eval_batch_size=BATCH_SIZE_TRAINING,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                gradient_checkpointing=gradient_checkpointing,
                optim="adamw_torch",
                evaluation_strategy="steps",
                save_strategy="steps",
                save_steps=EVAL_NUM_STEPS,                         
                eval_steps=EVAL_NUM_STEPS,                         
                logging_steps=logging_steps,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                fp16=False,
                bf16=True,
                num_train_epochs=NUM_TRAIN_EPOCHS,
                warmup_ratio=WARMUP_RATIO,
                group_by_length=True,
                lr_scheduler_type=LR_SCHEDULER_TYPE,    
                lr_scheduler_kwargs={},     
                push_to_hub=False,
                save_total_limit=2,
                report_to="wandb",
                load_best_model_at_end=True,
                seed=42,
            )


            trainer = SFTTrainer(
                model=model,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                tokenizer=dp.tokenizer,
                data_collator=data_collator,
                max_seq_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                args=train_params,
                packing=False,
                dataset_text_field="concatenated_text",
            )
            
                            
            ######################################################## TRAINING ########################################################


            logging.info("Start training")
            trainer.train()

            #: save only adapter
            finetune_model_path = experiment.model_path + "fine_tuned_full"
            model.save_pretrained(finetune_model_path)


            # Clear GPU memory
            model = None
            del model
            gc.collect()
            torch.cuda.empty_cache()


            ######################################################## RELOAD MODEL ########################################################

            
            # For better predictions, we reload the model and adapter in 16 bit
            logging.info("Model reload")

            model = AutoModelForCausalLM.from_pretrained(
                finetune_model_path,
                cache_dir=experiment.model_cache_path,
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                attn_implementation="flash_attention_2")


        if eval_model_path is not None:

            logging.info("Model load from path")

            model = AutoModelForCausalLM.from_pretrained(
                eval_model_path,
                cache_dir=experiment.model_cache_path,
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                attn_implementation="flash_attention_2")


        # Setup data processing for inference
        dp.set_for_inference()
        

        ######################################################## EVALUATION SETUP ########################################################
        

        # Setup all pre and post processing functions
        logging.info("Setting up eval")

        def preprocessing_function(constants_row, true_events_input, true_future_events_input, target_dataframe, eval_manager):
            #: convert input to string
            str_input, str_output, meta_data = dp.convert_to_string_single_patient(constants_row, true_events_input, true_future_events_input, 
                                                                                    target_dataframe, input_filtering_function=filtering_rows_rest_budget,
                                                                                    prompt=prompt, decimal_precision=DECIMAL_PRECISION,
                                                                                    constant_column_mapping=constant_column_mapping)
            return str_input, str_output, meta_data



        def encoding_function(input_text):

            #: apply pre-processing
            preprocessed_input_text = dp.preprocess_inputs([input_text])[0]
            
            return preprocessed_input_text



        def decoding_function(predictions):
            return dp.tokenizer.batch_decode(predictions, skip_special_tokens=True)



        def post_processing_function(prediction_string, patientid, patient_sample_index, meta_data):

            #: post process
            prediction_post = dp.post_process_string(prediction_string)

            if verbose:
                logging.info("Post processed predictions: " + prediction_post)

            #: convert all to dataframe
            all_days = meta_data["all_days_output"]
            predicted_df = dp.convert_to_df_single_patient(prediction_post, all_days, patientid, patient_sample_index, prediction_days_column_wise=meta_data["prediction_columns"])

            #: try convert to float here
            lab_vars = predicted_df.columns.str.contains("lab_")
            predicted_df.loc[:, lab_vars] = predicted_df.loc[:, lab_vars].apply(pd.to_numeric, errors='raise')

            # Return predicted DF
            return predicted_df


        experiment.model = model
        only_standardize = Only_Standardization(path_to_statistics_file)
        metric_manager = MetricManager(path_to_statistics_file)


        def evaluate_and_record(eval_set_events, eval_set_name, eval_meta_data, num_samples_to_generate, sample_merging_strategy, batch_size=BATCH_SIZE_VALIDATION):

            #: get targets and predictions
            eval_targets, eval_prediction, return_meta_data_list = experiment.get_output_for_split_hf_default(eval_set_events, 
                                                                                        eval_manager, 
                                                                                        preprocessing_function=preprocessing_function, 
                                                                                        tokenizer=dp.tokenizer,
                                                                                        encoding_function=encoding_function, 
                                                                                        decoding_function=decoding_function, 
                                                                                        post_processing_function=post_processing_function,
                                                                                        batch_size=batch_size,
                                                                                        verbose=verbose,
                                                                                        gen_top_p=0.9,
                                                                                        gen_do_sample=True,
                                                                                        max_new_tokens=None,
                                                                                        num_samples_to_generate=num_samples_to_generate, 
                                                                                        sample_merging_strategy=sample_merging_strategy,
                                                                                        pad_token_id=dp.tokenizer.eos_token_id,
                                                                                        max_output_length=4000,                     # Setting this here due to mistral long context bug, need to check if causes errors
                                                                                        return_meta_data=True,
                                                                                        note_down_probabilities=True)
            
            # Do filtering without standardizing
            eval_targets_filtered, eval_prediction_filtered = only_standardize.normalize_and_filter(eval_targets, eval_prediction)
            
            #: set grouping by therapy
            eval_targets_filtered_with_meta_data = experiment.join_meta_data_to_targets(eval_targets_filtered, eval_meta_data, generated_meta_data=return_meta_data_list)

            # Calculate performance metrics
            eval_performance = metric_manager.calculate_metrics(eval_targets_filtered, eval_prediction_filtered, group_by=None)

            # Save tables locally and record in wandb
            experiment.save_df_targets_predictions_locally_and_statistics_to_wandb(eval_set_name, eval_targets_filtered, eval_prediction_filtered, meta_data_df=eval_targets_filtered_with_meta_data)

            # Save performance to wandb
            experiment.save_to_wandb_final_performances(eval_performance, eval_set_name)

            return eval_targets_filtered, eval_prediction_filtered, eval_targets_filtered_with_meta_data

        
        
        ######################################################## TEST EVAL ########################################################
        
        logging.info("Test set Eval")
        test_targets, test_prediction, test_meta_data = evaluate_and_record(test_events, test_set, test_meta, num_samples_to_generate=num_samples_to_generate, sample_merging_strategy=sample_merging_strategy)



        ############################################################ Finish run ############################################################
        wandb.run.finish()




