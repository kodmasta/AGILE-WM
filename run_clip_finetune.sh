#!/bin/bash
#SBATCH --job-name=clip_ft
#SBATCH --output=logs/clip_ft_%j.out
#SBATCH --error=logs/clip_ft_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

cd ~/CODE/AGILE-WM
module load python/3.10

export SCRATCH="${SCRATCH:-/network/scratch/h/hengh}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SCRATCH/hf_cache}"

# Point this at the directory that contains caption shards such as
# shard-00000-caption-00000.tar. Update it if your caption tar files live elsewhere.
DATA_ROOT="${DATA_ROOT:-$SCRATCH/my_dataset/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/AGILE-WM/clip_finetune}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SLURM_TMPDIR:-/tmp}/clip_caption_shards}"
SHARD_GLOB="${SHARD_GLOB:-shard-*-caption-*.tar}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"

mkdir -p logs "$HF_HOME" "$TRANSFORMERS_CACHE" "$OUTPUT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "HF cache: $TRANSFORMERS_CACHE"
echo "Local shard dir: $LOCAL_DATA_DIR"

uv run python CLIP_finetune.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --hf_cache_dir "$TRANSFORMERS_CACHE" \
  --clip_model "$CLIP_MODEL" \
  --shard_glob "$SHARD_GLOB" \
  --stage_shards_to_local \
  --local_data_dir "$LOCAL_DATA_DIR" \
  --save_every 1 \
  --num_workers 2 \
  --resume
  "$@"