#!/pstore/data/dt-gpt/interpreter/uc1_gpu_venv2/bin/python
import __init__
from dt_gpt_fft_2024_03_21_biomistral_template_descr_bd import DTGPTCore_full_fine_tune_completion_loss_nucleus_sampling_template_description_basic_description_biomistral



# Run with exclusive 80GB gpu!

def main():
    
    import os
    # Setup for GPU usage on interactive
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2000"

    experiment = DTGPTCore_full_fine_tune_completion_loss_nucleus_sampling_template_description_basic_description_biomistral()

    eval_model_path = "/home/makaron1/scratch/uc2_experiments_lean/dt_gpt_biomistral_critical_vars/dt_gpt_biomistral_critical_vars/2024_03_21___16_59_16_333542/models/checkpoint-44028"

    experiment.run(debug=False, verbose=True, 
                    wandb_prefix_name="DT-GPT - BioMistral - 3.4k - FFT - TI - OBD - 30 Samples - EVAL 259 - Forecast: ", 
                    wandb_group_name="DT-GPT - BioMistral - FFT - Template Input - Basic Description",
                    train_set= "TRAIN", validation_set = "VALIDATION", test_set = "TEST",
                    learning_rate=1e-5, batch_size_training=1, batch_size_validation=10, weight_decay=0.1, gradient_accumulation=1,
                    num_train_epochs=5, eval_interval=1.0/5.0, warmup_ratio=0.10, lr_scheduler="cosine",
                    nr_days_forecasting=91, seq_max_len_in_tokens=3400, decimal_precision=1,
                    num_samples_to_generate=30, sample_merging_strategy="mean",
                    max_new_tokens_to_generate=1200,
                    eval_model_path=eval_model_path)



if __name__ == "__main__":
    main()










