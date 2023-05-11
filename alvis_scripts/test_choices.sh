set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

size=base
YEAR=${1:-"2017"}
MODEL_TO_EVAL='data/experiments/853398-base-templama-2017/checkpoint/step-100' # Model finetuned on tempLAMA for 100 steps

port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas/P279.jsonl"
PASSAGES="data/corpora/wiki/enwiki-dec${YEAR}-DEBUG/text-list-100-sec.jsonl data/corpora/wiki/enwiki-dec${YEAR}-DEBUG/infobox.jsonl"
SAVE_DIR=data/experiments/
EXPERIMENT_NAME=test-${size}-pararel-${YEAR}
PRECISION="fp32" # "bf16"

CUDA_VISIBLE_DEVICES=1 python -m debugpy --wait-for-client --listen 5678 -m evaluate \
    --name ${EXPERIMENT_NAME} \
    --generation_max_length 32 --target_maxlength 32 \
    --gold_score_mode "ppmean" \
    --precision ${PRECISION} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 384 \
    --model_path ${MODEL_TO_EVAL} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 2 --retriever_n_context 2 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat"  \
    --task "qa" \
    --passages ${PASSAGES}\
    --write_results \
    --qa_prompt_format "{question}" \
    --use_decoder_choices
