
# Set correct GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"


import __init__
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM
import os
from transformers import AutoModelForCausalLM
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
import gc
from pipeline.Experiment import Experiment
import wandb
import logging

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2000"


BATCH_SIZE_TRAINING = 1
GRADIENT_ACCUMULATION = 1
gradient_checkpointing = False
logging_steps = 100

WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.10
LR_SCHEDULER_TYPE = "cosine"
SEQUENCE_MAX_LENGTH_IN_TOKENS = 3700
#: set training
WANDB_DEBUG = False


class DataProcessorBiomistral():

    def __init__(self, model_name, max_total_length=4000): #Q: collator setting ?


        #: Define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                        force_download=False,
                                                        truncation_side="left",
                                                        padding_side="left",
                                                        add_eos_token=True,
                                                        add_bos_token=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.chat_template = ""

        # Define response template
        self.response_template = "<patient_prediction>"

        # Define collator
        self.data_collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)

        #######################################################################################################

        # Set constants
        self.max_total_length = max_total_length


        # Processed data
        self.processed_dataset = {}

    def preprocess_inputs(self, input_list):
            
        input_list = [input_prompt + ' ' + self.response_template for input_prompt in input_list]
        #: remove " since it creates unnecessary labels in the input
        input_list = [input_prompt.replace('"', '') for input_prompt in input_list]

        return input_list
        

    def preprocess_dataset(self, input_list, target_list):
        
        # : see https://discuss.huggingface.co/t/longt5-fine-tunning/22650
    
        # Preprocess inputs
        input_list = self.preprocess_inputs(input_list)
        # Setup dataset
        dataset = Dataset.from_dict({"input_prompt": input_list, "target": target_list})


        def prepare_text_for_training(input_target_dict):

            inputs = input_target_dict['input_prompt']
            targets = input_target_dict['target']

            # concatenate inputs & outputs, since we're doing causal LM  # Q: causal ML ????
            assert len(inputs) ==  len(targets), "Different lengths of input prompts and targets"
            concatenated_text_list = [str(inputs[i]) + " " + str(targets[i]) for i in range(len(inputs))]

            model_inputs = self.tokenizer(text=concatenated_text_list, max_length=self.max_total_length, truncation=True)
            model_inputs["concatenated_text"] = concatenated_text_list

            return model_inputs

        # Tokenization
        preprocessed_dataset = dataset.map(prepare_text_for_training, batched=True)

        return preprocessed_dataset

    def decode_tokenized_string(self, tokenized_string):
        output_text = self.tokenizer.decode(tokenized_string, skip_special_tokens=True)
        return output_text
    
    # MODES
    def set_for_training(self):
        self.tokenizer.add_eos_token = True
    
    def set_for_inference(self):
        self.tokenizer.add_eos_token = False

    def get_collator(self):
        return self.data_collator



    

def run_training(num_validation_samples=50,
                 lr=1e-5,
                 num_train_epochs=10,):

    
    # Setup experiment
    experiment = Experiment("adni_dt_gpt")

    # Uncomment for debug
    #if WANDB_DEBUG:
    #    experiment.setup_wandb_debug_mode()
    #else:
    #    experiment.setup_wandb("DT-GPT Training", "DT-GPT", project="UC - ADNI")
    
    # load data
    training_data = pd.read_csv("/pstore/data/dt-gpt/uc4-alzheimers-disease/data/CPAD_training_data.csv")
    training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)

    input_list = list(training_data['INPUT'].values)
    target_list = list(training_data['TARGET'].values)

    
    if num_validation_samples > 0:
        training_input_list = input_list[:-num_validation_samples]
        test_input_list = input_list[-num_validation_samples:]

        training_target_list = target_list[:-num_validation_samples]
        test_target_list = target_list[-num_validation_samples:]
    else:
        training_input_list = input_list
        training_target_list = target_list
    
    
    # Preprocess data
    model_name = "BioMistral/BioMistral-7B-DARE"
    dp = DataProcessorBiomistral(model_name, max_total_length=4000)

    dp.set_for_training()

    training_dataset = dp.preprocess_dataset(training_input_list, training_target_list)

    
    if num_validation_samples > 0:
        validation_dataset = dp.preprocess_dataset(test_input_list, test_target_list)

    # Load data collator
    data_collator = dp.get_collator()
    
    # Load model
    

    model = AutoModelForCausalLM.from_pretrained('BioMistral/BioMistral-7B-DARE', 
                                                            cache_dir=experiment.model_cache_path,
                                                            torch_dtype=torch.bfloat16,
                                                            device_map="auto")
    model.config.pretraining_tp = 1
    
    
    # Setup parameters 
    eval_num_steps = 1.0 / num_train_epochs
    
    # train model
    train_params = TrainingArguments(
        output_dir=experiment.model_path,
        per_device_train_batch_size=BATCH_SIZE_TRAINING,
        per_device_eval_batch_size=BATCH_SIZE_TRAINING if num_validation_samples > 0 else None,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch",
        evaluation_strategy="steps" if num_validation_samples > 0 else "no",
        save_strategy="steps",
        save_steps=eval_num_steps,                         
        eval_steps=eval_num_steps if num_validation_samples > 0 else None,                         
        logging_steps=logging_steps,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        fp16=False,
        bf16=True,
        num_train_epochs=num_train_epochs,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,    
        lr_scheduler_kwargs={},     
        push_to_hub=False,
        save_total_limit=2,
        report_to="wandb",
        load_best_model_at_end=True if num_validation_samples > 0 else False,
        seed=42)
    

    trainer = SFTTrainer(
        model=model,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset if num_validation_samples > 0 else None,
        tokenizer=dp.tokenizer,
        data_collator=data_collator,
        max_seq_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
        args=train_params,
        packing=False,
        dataset_text_field="concatenated_text")
    

    # Run training
    trainer.train()

    # save best model
    finetune_model_path = experiment.model_path + "cpad_final/"
    model.save_pretrained(finetune_model_path)
    dp.tokenizer.save_pretrained(finetune_model_path)
    logging.info(f"Saved fine-tuned model to {finetune_model_path}")

    # Clear GPU memory
    model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    wandb.finish()





if __name__ == "__main__":
    
    #: run, based on grid search here: https://genentech.wandb.io/nikitamakarov/UC%20-%20ADNI/runs/mo5535c0/workspace?nw=nwusernikitamakarov
    run_training(num_validation_samples=0, lr=1e-5, num_train_epochs=4)
    

