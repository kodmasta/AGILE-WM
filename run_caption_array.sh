#!/bin/bash
#SBATCH --job-name=qwen_cap
#SBATCH --output=logs/qwen_cap_%A_%a.out
#SBATCH --error=logs/qwen_cap_%A_%a.err
#SBATCH --array=0-59%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

cd ~/projects/AGILE-WM/qwen_captioning
source venv/bin/activate

mkdir -p logs
mkdir -p outputs

SHARD_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" shards.txt)
SHARD_BASENAME=$(basename "$SHARD_PATH")
SHARD_NAME="${SHARD_BASENAME%.tar}"

MODEL_SRC="$SCRATCH/qwen3-vl-8b-instruct"

SHARD_DST="$SLURM_TMPDIR/$SHARD_BASENAME"

echo "Copying shard to local disk..."
cp "$SHARD_PATH" "$SHARD_DST"

echo "Running captioning..."
python caption_shard.py \
  --shard_path "$SHARD_DST" \
  --output_path "outputs/${SHARD_NAME}.jsonl" \
  --model_dir "$MODEL_SRC"