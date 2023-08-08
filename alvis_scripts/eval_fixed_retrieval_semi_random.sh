#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A SNIC2022-22-1040
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=A40:4
#SBATCH --job-name=eval-atlas-pararel
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/atlas/logs/pararel_eval_zero_shot_base_fixed_retrieval_semi_random_%A_%a.out
#SBATCH -t 0-06:00:00

# COMMENT: a copy of eval_fixed_retrieval_random.sh using ParaRel queries with provided random fixed retrievals, unified across passages.

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

#0-29
EVAL_RELATIONS=("P138" "P937" "P1412" "P127" "P103" "P276" "P159" "P140" "P136" "P495" "P17" "P361" "P36" "P740" "P264" "P407" "P30" "P131" "P176" "P279" "P19" "P101" "P364" "P106" "P1376" "P178" "P413" "P27" "P20" "P449")
RELATION_TO_EVAL=${EVAL_RELATIONS[$SLURM_ARRAY_TASK_ID]}

size=base
YEAR=${1:-"2017"}
MODEL_TO_EVAL=data/models/atlas/${size}

port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space_w_retrieval_semi_random/${RELATION_TO_EVAL}.jsonl"
PASSAGES="data/corpora/wiki/enwiki-dec${YEAR}/text-list-100-sec.jsonl data/corpora/wiki/enwiki-dec${YEAR}/infobox.jsonl"
SAVE_DIR=data/experiments/pararel-eval-zero-shot-base-fixed-retrieval-semi-random
EXPERIMENT_NAME=${RELATION_TO_EVAL}-${size}-${YEAR}-${SLURM_JOB_ID}
PRECISION="fp32" # "bf16"

srun python evaluate.py \
    --name ${EXPERIMENT_NAME} \
    --generation_max_length 32 --target_maxlength 32 \
    --gold_score_mode "ppmean" \
    --precision ${PRECISION} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 384 \
    --model_path ${MODEL_TO_EVAL} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 20 --retriever_n_context 20 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat"  \
    --task "qa" \
    --passages ${PASSAGES}\
    --write_results \
    --qa_prompt_format "{question}" \
    --use_decoder_choices \
    --generation_num_beams 1 \
    --choice_batch_size 128 \
    --use_file_passages
