set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

size=base
YEAR=${1:-"2017"}
MODEL_TO_EVAL='google/t5-base-lm-adapt' # Basline T5 model

port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas/P279.jsonl"
SAVE_DIR=data/experiments/pararel_eval_baseline_t5
EXPERIMENT_NAME=test-${RELATION_TO_EVAL}-${SLURM_JOB_ID}
PRECISION="fp32" # "bf16"

CUDA_VISIBLE_DEVICES=1 python -m debugpy --wait-for-client --listen 5678 -m evaluate_baseline \
    --name ${EXPERIMENT_NAME} \
    --generation_max_length 32 --target_maxlength 32 \
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
    --write_results \
    --qa_prompt_format "{question}" \
    --use_decoder_choices \
    --closed_book