
set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

size=base
YEAR=${1:-"2017"}
MODEL_TO_EVAL=data/models/atlas/${size}

port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space/P140_100.jsonl"
SAVE_DIR=data/experiments/test-pararel-compute-r-embeddings-base
EXPERIMENT_NAME=${size}-${YEAR}-${SLURM_JOB_ID}
PRECISION="fp32" # "bf16"

CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 python -m debugpy --wait-for-client --listen 5678 -m compute_retriever_embeddings \
    --name ${EXPERIMENT_NAME} \
    --gold_score_mode "ppmean" \
    --precision ${PRECISION} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 384 \
    --model_path ${MODEL_TO_EVAL} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --task "qa" \
    --qa_prompt_format "{question}" \
