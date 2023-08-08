#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -A SNIC2022-22-1040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=compute-atlas-large-pararel-r-embeddings
#SBATCH -o /mimer/NOBACKUP/groups/snic2021-23-309/project-data/atlas/logs/pararel_r_embeddings_zero_shot_large_%A_%a.out
#SBATCH -t 0-06:00:00

# COMMENT: a copy of eval.sh only for data files in which we have removed spaces preceding <extra_id_0> in the files.

set -eo pipefail

module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate

#0-29
EVAL_RELATIONS=("P138" "P937" "P1412" "P127" "P103" "P276" "P159" "P140" "P136" "P495" "P17" "P361" "P36" "P740" "P264" "P407" "P30" "P131" "P176" "P279" "P19" "P101" "P364" "P106" "P1376" "P178" "P413" "P27" "P20" "P449")
RELATION_TO_EVAL=${EVAL_RELATIONS[$SLURM_ARRAY_TASK_ID]}

size=large
YEAR=${1:-"2017"}
MODEL_TO_EVAL=data/models/atlas/${size}

port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="/cephyr/users/lovhag/Alvis/projects/pararel/data/all_n1_atlas_no_space/${RELATION_TO_EVAL}.jsonl"
SAVE_DIR=data/experiments/pararel-compute-r-embeddings-large
EXPERIMENT_NAME=${RELATION_TO_EVAL}-${YEAR}-${SLURM_JOB_ID}
PRECISION="fp32" # "bf16"

srun python compute_retriever_embeddings.py \
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
