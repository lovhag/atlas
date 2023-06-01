#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A SNIC2022-22-1040
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=A40:4
#SBATCH --job-name=eval-baseline-t5-pararel
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/atlas/logs/pararel_eval_baseline_t5_likelihood_no_eos_with_3_%A_%a.out
#SBATCH -t 0-04:00:00

# COMMENT: this is largely a copy of eval_no_space.sh, only without retrieval and for a baseline T5

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

# 0-29
EVAL_RELATIONS=("P937" "P1412" "P103" "P276" "P159" "P140" "P136" "P495" "P17" "P361" "P36" "P740" "P264" "P407" "P30" "P131" "P176" "P279" "P19" "P101" "P364" "P106" "P1376" "P178" "P413" "P27" "P20" "P449" "P138" "P127")
RELATION_TO_EVAL=${EVAL_RELATIONS[$SLURM_ARRAY_TASK_ID]}

size=base
YEAR=${1:-"2017"}
MODEL_TO_EVAL='google/t5-base-lm-adapt' # Basline T5 model

port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space/${RELATION_TO_EVAL}.jsonl"
SAVE_DIR=data/experiments/pararel-eval-baseline-t5-no-space-likelihood-no-eos-with-3
EXPERIMENT_NAME=${RELATION_TO_EVAL}-${SLURM_JOB_ID}
PRECISION="fp32" # "bf16"

srun python evaluate_baseline.py \
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
    --choice_batch_size 256 \
    --closed_book
