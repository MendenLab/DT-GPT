# How to run DT-GPT ADNI

```
# 1. Run training, requires 1 GPU with >= 40 GB memory
./2025_02_03_dt_gpt_train_full.py


# 2. (Adjust scripts to model and) launch API server with model using VLLM for parallelized inference
./2025_02_04_vllm_launch_dt_gpt.sh

# 3. In parallel to step 2, (adjust scripts to model and) run python script to generate and save the results
./2025_02_04_dt_gpt_eval.py


# 4. Afterwards, transform them back into dataframes and evaluate in Jupyter Notebook
./2025_02_05_transformation_and_post_processing.ipynb


```

