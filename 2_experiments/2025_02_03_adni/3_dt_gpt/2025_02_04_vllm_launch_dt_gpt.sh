

#: set correct path
MODEL_PATH="/pstore/data/dt-gpt/raw_experiments/uc2_nsclc/adni_dt_gpt/adni_dt_gpt/2025_02_03___16_54_36_202743/models/fine_tuned_full"
PORT=21783


TOKENIZER="BioMistral/BioMistral-7B-DARE"

# ml Python/3.12.4-GCCcore-9.3.0

source /home/makaron1/dt-gpt/interpreter/venv_vllm/bin/activate


# GPU
GPU=5

echo ""
echo ""
echo "Starting VLLM API server on port $PORT with model $MODEL_PATH on GPU $GPU"
echo ""
echo ""
echo ""
echo ""

CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server --port $PORT --model $MODEL_PATH --tokenizer $TOKENIZER --enable-prefix-caching --disable-sliding-window




