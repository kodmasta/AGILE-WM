#!/bin/bash
#SBATCH --job-name=rnn_train
#SBATCH --output=artifacts/logs/rnn_train_%j.out
#SBATCH --error=artifacts/logs/rnn_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=main

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
setup_slurm_runtime

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

CONFIG_PATH="${CONFIG_PATH:-configs/carracing.config}"
EXP_NAME="${EXP_NAME:-WorldModels}"
ENV_NAME="${ENV_NAME:-CarRacing-v0}"

# Override this if your rollout directory is named differently.
DEFAULT_ROLLOUT_DIR="$AGILE_WM_ARTIFACTS_ROOT/datasets/rollouts"
ROLLOUT_DIR="${ROLLOUT_DIR:-$DEFAULT_ROLLOUT_DIR}"
ROLLOUT_GLOB="${ROLLOUT_GLOB:-episode_*.npz}"

RESULTS_ROOT="${RESULTS_ROOT:-$AGILE_WM_ARTIFACTS_ROOT/world_models/$EXP_NAME/$ENV_NAME}"
SERIES_DIR="${SERIES_DIR:-$RESULTS_ROOT/series}"
RNN_DIR="${RNN_DIR:-$RESULTS_ROOT/tf_rnn}"

RECONSTRUCTION_DIR="${RECONSTRUCTION_DIR:-$AGILE_WM_ARTIFACTS_ROOT/experiments/fusion_reconstruction/shard0_ep5_cbp544}"
FORCE_REBUILD_SERIES="${FORCE_REBUILD_SERIES:-0}"

JOB_ENV_ROOT="${SLURM_TMPDIR:-/tmp}/agile-wm-rnn-${SLURM_JOB_ID:-manual}"
SERIES_VENV_DIR="$JOB_ENV_ROOT/series-env"
SERIES_PYTHON="$SERIES_VENV_DIR/bin/python"
TRAIN_VENV_DIR="$JOB_ENV_ROOT/train-env"
TRAIN_PYTHON="$TRAIN_VENV_DIR/bin/python"

TF_SERIES_PACKAGE="${TF_SERIES_PACKAGE:-tensorflow==2.18.1}"
TF_TRAIN_PACKAGE="${TF_TRAIN_PACKAGE:-tensorflow[and-cuda]==2.18.1}"
TFP_PACKAGE="${TFP_PACKAGE:-tensorflow-probability[tf]==0.25.0}"
SERIES_RUNTIME_PACKAGES=(accelerate peft pillow transformers tqdm)

ensure_artifact_dirs "$(dirname "$JOB_ENV_ROOT")" "$RESULTS_ROOT"

choose_clip_checkpoint() {
  if [[ -n "${CLIP_CHECKPOINT:-}" ]]; then
    echo "$CLIP_CHECKPOINT"
    return 0
  fi

  local candidates=(
    "$AGILE_WM_ARTIFACTS_ROOT/models/clip_finetune/merged_final"
    "$AGILE_WM_ARTIFACTS_ROOT/models/clip_finetune/lora_final"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

ensure_cbp_state() {
  local reconstruction_dir="$1"
  local cbp_state_path="$reconstruction_dir/cbp_state.npz"
  local run_config_path="$reconstruction_dir/run_config.json"

  if [[ -f "$cbp_state_path" ]]; then
    return 0
  fi

  if [[ ! -f "$run_config_path" ]]; then
    echo "CBP state is missing and run_config.json was not found in $reconstruction_dir" >&2
    return 1
  fi

  echo "Reconstructing missing CBP state at $cbp_state_path from $run_config_path"
  python3 - "$run_config_path" "$cbp_state_path" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

run_config_path = Path(sys.argv[1])
cbp_state_path = Path(sys.argv[2])
config = json.loads(run_config_path.read_text(encoding="utf-8"))

input_dim_a = int(config["z_size"])
input_dim_b = int(config["clip_feature_dim"])
output_dim = int(config["fusion_dim"])
seed = int(config.get("seed", 42))
normalize = bool(config.get("normalize_fused_features", False))

rng = np.random.default_rng(seed)
hash_a = rng.integers(0, output_dim, size=input_dim_a, endpoint=False, dtype=np.int32)
sign_a = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=input_dim_a).astype(np.float32)
hash_b = rng.integers(0, output_dim, size=input_dim_b, endpoint=False, dtype=np.int32)
sign_b = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=input_dim_b).astype(np.float32)

cbp_state_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    cbp_state_path,
    hash_a=hash_a,
    sign_a=sign_a,
    hash_b=hash_b,
    sign_b=sign_b,
    output_dim=np.asarray([output_dim], dtype=np.int32),
    normalize=np.asarray([int(normalize)], dtype=np.int32),
)
PY
}

prepare_base_env() {
  local venv_dir="$1"
  local python_bin="$2"

  rm -rf "$venv_dir"
  uv venv --python 3.10 "$venv_dir"

  uv sync \
    --python "$python_bin" \
    --frozen \
    --no-install-package torch \
    --no-install-package torchvision \
    --no-install-package torchaudio
}

prepare_series_env() {
  prepare_base_env "$SERIES_VENV_DIR" "$SERIES_PYTHON"

  local gpu_cc
  gpu_cc="$(detect_gpu_compute_capability || true)"
  if [[ -z "$gpu_cc" ]]; then
    echo "Could not determine GPU compute capability from nvidia-smi." >&2
    exit 1
  fi

  local torch_stack
  if ! torch_stack="$(select_torch_stack "$gpu_cc")"; then
    echo "No torch stack configured for compute capability $gpu_cc." >&2
    exit 1
  fi

  echo "Detected GPU compute capability: $gpu_cc"
  echo "Installing torch stack for series preprocessing: $torch_stack"
  install_torch_stack "$SERIES_PYTHON" "$torch_stack"

  uv pip install --python "$SERIES_PYTHON" "${SERIES_RUNTIME_PACKAGES[@]}"
  uv pip install --python "$SERIES_PYTHON" "$TF_SERIES_PACKAGE" "$TFP_PACKAGE"

  "$SERIES_PYTHON" - <<'PY'
import tensorflow as tf
import tensorflow_probability as tfp
import torch

print("series_tensorflow", tf.__version__)
print("series_tensorflow_probability", tfp.__version__)
print("series_torch", torch.__version__)
print("series_torch_cuda", torch.version.cuda)
print("series_cuda_available", torch.cuda.is_available())
PY
}

prepare_train_env() {
  prepare_base_env "$TRAIN_VENV_DIR" "$TRAIN_PYTHON"

  uv pip install --python "$TRAIN_PYTHON" "$TF_TRAIN_PACKAGE" "$TFP_PACKAGE"

  "$TRAIN_PYTHON" - <<'PY'
import tensorflow as tf
import tensorflow_probability as tfp

print("tensorflow", tf.__version__)
print("tensorflow_probability", tfp.__version__)
print("tf_gpus", tf.config.list_physical_devices("GPU"))
PY
}

count_rollouts() {
  find "$ROLLOUT_DIR" -maxdepth 1 -type f -name "$ROLLOUT_GLOB" | wc -l
}

series_ready() {
  local required=(metadata.json fused_latents.npy actions.npy rewards.npy done.npy)
  local name
  for name in "${required[@]}"; do
    if [[ ! -f "$SERIES_DIR/$name" ]]; then
      return 1
    fi
  done
  return 0
}

if [[ ! -d "$ROLLOUT_DIR" ]]; then
  echo "ROLLOUT_DIR does not exist: $ROLLOUT_DIR" >&2
  exit 1
fi

NUM_ROLLOUTS="${NUM_ROLLOUTS:-$(count_rollouts)}"
if [[ "$NUM_ROLLOUTS" -le 0 ]]; then
  echo "No rollout files matching $ROLLOUT_GLOB were found in $ROLLOUT_DIR" >&2
  exit 1
fi

CLIP_CHECKPOINT="$(choose_clip_checkpoint || true)"
if [[ -z "$CLIP_CHECKPOINT" ]]; then
  echo "Could not find a CLIP checkpoint. Set CLIP_CHECKPOINT to a merged_final or lora_final directory." >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Config path: $CONFIG_PATH"
echo "Rollout dir: $ROLLOUT_DIR"
echo "Rollout glob: $ROLLOUT_GLOB"
echo "Num rollouts: $NUM_ROLLOUTS"
echo "Series dir: $SERIES_DIR"
echo "RNN dir: $RNN_DIR"
echo "Reconstruction dir: $RECONSTRUCTION_DIR"
echo "CLIP checkpoint: $CLIP_CHECKPOINT"
echo "HF cache: $TRANSFORMERS_CACHE"
echo "Series env: $SERIES_VENV_DIR"
echo "Train env: $TRAIN_VENV_DIR"

ensure_cbp_state "$RECONSTRUCTION_DIR"

if [[ "$FORCE_REBUILD_SERIES" == "1" ]]; then
  rm -rf "$SERIES_DIR"
fi

if series_ready; then
  echo "Using existing series cache at $SERIES_DIR"
else
  prepare_series_env
  echo "Building series cache at $SERIES_DIR"
  series_cmd=(
    "$SERIES_PYTHON" series.py
    --config_path "$CONFIG_PATH"
    --rollout_dir "$ROLLOUT_DIR"
    --rollout_glob "$ROLLOUT_GLOB"
    --output_dir "$SERIES_DIR"
    --num_rollouts "$NUM_ROLLOUTS"
    --reconstruction_dir "$RECONSTRUCTION_DIR"
    --clip_checkpoint "$CLIP_CHECKPOINT"
    --hf_cache_dir "$TRANSFORMERS_CACHE"
  )

  if [[ -n "${VAE_CHECKPOINT:-}" ]]; then
    series_cmd+=(--vae_checkpoint "$VAE_CHECKPOINT")
  fi
  if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
    series_cmd+=(--local_files_only)
  fi
  if [[ "${NORMALIZE_CLIP_FEATURES:-0}" == "1" ]]; then
    series_cmd+=(--normalize_clip_features)
  fi
  if [[ -n "${FRAME_BATCH_SIZE:-}" ]]; then
    series_cmd+=(--frame_batch_size "$FRAME_BATCH_SIZE")
  fi
  if [[ -n "${CHUNK_SIZE:-}" ]]; then
    series_cmd+=(--chunk_size "$CHUNK_SIZE")
  fi
  if [[ -n "${DISCARD_START_FRAMES:-}" ]]; then
    series_cmd+=(--discard_start_frames "$DISCARD_START_FRAMES")
  fi

  "${series_cmd[@]}"
fi

prepare_train_env

train_cmd=(
  "$TRAIN_PYTHON" rnn_train.py
  --config_path "$CONFIG_PATH"
  --series_dir "$SERIES_DIR"
  --output_dir "$RNN_DIR"
)

if [[ -n "${RNN_NUM_STEPS:-}" ]]; then
  train_cmd+=(--rnn_num_steps "$RNN_NUM_STEPS")
fi
if [[ -n "${RNN_BATCH_SIZE:-}" ]]; then
  train_cmd+=(--rnn_batch_size "$RNN_BATCH_SIZE")
fi
if [[ -n "${SAVE_EVERY:-}" ]]; then
  train_cmd+=(--save_every "$SAVE_EVERY")
fi
if [[ -n "${LOG_EVERY:-}" ]]; then
  train_cmd+=(--log_every "$LOG_EVERY")
fi
if [[ -n "${VAL_SPLIT:-}" ]]; then
  train_cmd+=(--val_split "$VAL_SPLIT")
fi
if [[ -n "${VAL_BATCHES:-}" ]]; then
  train_cmd+=(--val_batches "$VAL_BATCHES")
fi

echo "Starting RNN training"
"${train_cmd[@]}" "$@"
