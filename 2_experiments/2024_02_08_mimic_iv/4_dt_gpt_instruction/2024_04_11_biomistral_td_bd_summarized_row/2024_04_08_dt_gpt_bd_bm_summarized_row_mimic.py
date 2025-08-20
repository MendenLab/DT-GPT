import os
# To overcome issues with CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#!/pstore/data/dt-gpt/interpreter/uc1_gpu_venv2/bin/python
import __init__
from dt_gpt_fft_2024_04_11_biomistral_td_bd_sr import DTGPT_mimic_biomistral_fft_ti_bd_sr


# Run with exclusive 80GB gpu!

def main():
    
    import os
    # Setup for GPU usage on interactive
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

    experiment = DTGPT_mimic_biomistral_fft_ti_bd_sr()

    
    experiment.run(debug=False, verbose=True, 
                    wandb_prefix_name="DT-GPT - BioMistral - 3.4k - FFT - TI - BD - SR - 30 Samples - Forecast: ", 
                    wandb_group_name="DT-GPT - BioMistral - FFT - Template Input - Basic Description - Summarized Row",
                    train_set= "TRAIN", validation_set = "VALIDATION", test_set = "TEST",
                    learning_rate=1e-5, batch_size_training=1, batch_size_validation=10, weight_decay=0.1, gradient_accumulation=1,
                    num_train_epochs=5, eval_interval=1.0/10.0, warmup_ratio=0.10, lr_scheduler="cosine",
                    nr_days_forecasting=91, seq_max_len_in_tokens=3400, decimal_precision=1,
                    num_samples_to_generate=30, sample_merging_strategy="mean",
                    max_new_tokens_to_generate=900)



if __name__ == "__main__":
    main()




