#!/bin/bash
#SBATCH --job-name=qwen_cap
#SBATCH --output=artifacts/logs/qwen_cap_%A_%a.out
#SBATCH --error=artifacts/logs/qwen_cap_%A_%a.err
#SBATCH --array=20-39%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --partition=main

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
setup_slurm_runtime

export UV_PYTHON="${UV_PYTHON:-3.10}"
OUTPUT_DIR="${OUTPUT_DIR:-$AGILE_WM_ARTIFACTS_ROOT/datasets/frame_caption_pairs}"
JOB_VENV_DIR="${SLURM_TMPDIR:-/tmp}/agile-wm-caption-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
JOB_PYTHON="$JOB_VENV_DIR/bin/python"
SHARDS_FILE="${SHARDS_FILE:-$REPO_ROOT/shards.txt}"

ensure_artifact_dirs "$OUTPUT_DIR"

CAPTION_RUNTIME_PACKAGES=(accelerate numpy pillow transformers)

prepare_caption_env() {
  echo "Preparing job-local caption environment at $JOB_VENV_DIR"
  rm -rf "$JOB_VENV_DIR"
  uv venv --python "$UV_PYTHON" "$JOB_VENV_DIR"
  uv pip install --python "$JOB_PYTHON" "${CAPTION_RUNTIME_PACKAGES[@]}"
}

SHARD_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SHARDS_FILE")
SHARD_BASENAME=$(basename "$SHARD_PATH")

MODEL_SRC="${MODEL_SRC:-$AGILE_WM_ARTIFACTS_ROOT/models/qwen3-vl-8b-instruct}"

SHARD_DST="$SLURM_TMPDIR/$SHARD_BASENAME"

echo "Copying shard to local disk..."
cp "$SHARD_PATH" "$SHARD_DST"

echo "Running captioning..."
echo "Caption outputs will be written to: $OUTPUT_DIR"
prepare_caption_env
echo "Using job-local python: $JOB_PYTHON"

GPU_CC="$(detect_gpu_compute_capability || true)"
if [[ -n "$GPU_CC" ]]; then
  echo "Detected GPU compute capability: $GPU_CC"
else
  echo "Could not determine GPU compute capability from nvidia-smi." >&2
  exit 1
fi

if ! TORCH_STACK="$(select_torch_stack "$GPU_CC")"; then
  echo "No explicit torch stack is configured for compute capability $GPU_CC." >&2
  exit 1
fi

echo "Selected torch stack: $TORCH_STACK"
install_torch_stack "$JOB_PYTHON" "$TORCH_STACK"

torch_probe_status=0
if torch_supports_active_gpu "$JOB_PYTHON"; then
  echo "Installed torch build is compatible with the active GPU."
else
  torch_probe_status=$?
  if [[ "$torch_probe_status" -eq 2 ]]; then
    echo "Installed torch stack failed to import after selecting $TORCH_STACK. Check the preceding CUDA/NCCL loader error." >&2
  else
    echo "Installed torch build is incompatible with compute capability $GPU_CC after selecting stack $TORCH_STACK." >&2
  fi
  exit 1
fi

"$JOB_PYTHON" caption_shard.py \
  --shard_path "$SHARD_DST" \
  --output_dir "$OUTPUT_DIR" \
  --model_dir "$MODEL_SRC" \
  --shard_size 1000
