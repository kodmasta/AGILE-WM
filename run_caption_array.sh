#!/bin/bash
#SBATCH --job-name=qwen_cap
#SBATCH --output=logs/qwen_cap_%A_%a.out
#SBATCH --error=logs/qwen_cap_%A_%a.err
#SBATCH --array=20-60%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd ~/CODE/AGILE-WM
module load python/3.10

SHARD_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" shards.txt)
SHARD_BASENAME=$(basename "$SHARD_PATH")

MODEL_SRC="$SCRATCH/qwen3-vl-8b-instruct"

SHARD_DST="$SLURM_TMPDIR/$SHARD_BASENAME"

echo "Copying shard to local disk..."
cp "$SHARD_PATH" "$SHARD_DST"

echo "Running captioning..."
uv run python caption_shard.py \
  --shard_path "$SHARD_DST" \
  --output_dir "outputs" \
  --model_dir "$MODEL_SRC" \
  --shard_size 1000

  