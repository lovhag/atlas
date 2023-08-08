#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A SNIC2022-22-1040
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=A40:4
#SBATCH --job-name=get-atlas-large-embeddings
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/atlas/logs/get_atlas_large_passage_embeddings_test.out
#SBATCH -t 0-10:00:00

# COMMENT: use this file to get the passage embeddings for atlas large.

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

size=large
YEAR=${1:-"2017"}
MODEL_TO_EVAL=data/models/atlas/${size}

port=$(shuf -i 15000-16000 -n 1)
PASSAGES="data/corpora/wiki/enwiki-dec${YEAR}-DEBUG/text-list-100-sec.jsonl data/corpora/wiki/enwiki-dec${YEAR}-DEBUG/infobox.jsonl"
SAVE_DIR=data/experiments/atlas-large-get-passage-embeddings-test
EXPERIMENT_NAME=${YEAR}-${SLURM_JOB_ID}
PRECISION="fp32" # "bf16"

srun python evaluate.py \
    --name ${EXPERIMENT_NAME} \
    --generation_max_length 32 --target_maxlength 32 \
    --gold_score_mode "ppmean" \
    --precision ${PRECISION} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 384 \
    --model_path ${MODEL_TO_EVAL} \
    --per_gpu_batch_size 1 \
    --n_context 20 --retriever_n_context 20 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat"  \
    --task "qa" \
    --passages ${PASSAGES}\
    --qa_prompt_format "{question}" \
    --save_index_path "data/saved_index/test-atlas-large-wiki-${YEAR}"
