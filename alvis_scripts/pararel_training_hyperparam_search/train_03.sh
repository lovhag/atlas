#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A SNIC2022-22-1040
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=A40:4
#SBATCH --job-name=train-atlas-pararel-hyperparam-search-03
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/atlas/logs/pararel_train_hyperparam_search_03.out
#SBATCH -t 0-04:00:00

# COMMENT: this script relies on a previous passage encoding (load_index_path)

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

size=base
YEAR=${1:-"2017"}

PASSAGES="data/corpora/wiki/enwiki-dec${YEAR}/text-list-100-sec.jsonl data/corpora/wiki/enwiki-dec${YEAR}/infobox.jsonl"

EXPERIMENT_NAME=03-${SLURM_JOB_ID}

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas/P138.jsonl"
EVAL_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas/P17_100.jsonl /cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas/P101_100.jsonl /cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas/P264_100.jsonl /cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas/P449_100.jsonl"
PRETRAINED_MODEL=data/models/atlas/${size}
SAVE_DIR=data/experiments/pararel_training_hyperparam_search/
PRECISION="fp32" # "bf16"

srun python train.py \
    --shuffle \
    --train_retriever --query_side_retriever_training\
    --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever\
    --precision ${PRECISION} \
    --shard_optim --shard_grads \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index -1 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --dropout 0.1 \
    --lr 4e-05 --lr_retriever 4e-05 \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILES} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 10 --retriever_n_context 10 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 25 \
    --log_freq 4 \
    --total_steps 150 \
    --warmup_steps 20 \
    --save_freq 25 \
    --main_port $port \
    --write_results \
    --task qa \
    --index_mode flat \
    --passages ${PASSAGES}\
    --qa_prompt_format "{question}" \
    --load_index_path data/experiments/853398-base-templama-2017/saved_index  \
    --use_decoder_choices
