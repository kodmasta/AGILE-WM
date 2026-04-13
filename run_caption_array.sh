#!/bin/bash
#SBATCH --job-name=qwen_cap
#SBATCH --output=logs/qwen_cap_%A_%a.out
#SBATCH --error=logs/qwen_cap_%A_%a.err
#SBATCH --array=20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd ~/CODE/AGILE-WM
module load python/3.10

export SCRATCH="${SCRATCH:-/network/scratch/h/hengh}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SCRATCH/hf_cache}"
export UV_PYTHON="${UV_PYTHON:-3.10}"

mkdir -p logs "$HF_HOME" "$TRANSFORMERS_CACHE"

LEGACY_TORCH_INDEX="https://download.pytorch.org/whl/cu124"
LEGACY_TORCH_PACKAGES=(torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0)
MODERN_TORCH_PACKAGES=(torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0)

detect_gpu_compute_capability() {
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

select_torch_stack() {
  local gpu_cc="$1"

  case "$gpu_cc" in
    7.0)
      echo "legacy-cu124"
      ;;
    7.5|8.0|8.6|8.9|9.0)
      echo "modern-default"
      ;;
    *)
      return 1
      ;;
  esac
}

install_torch_stack() {
  local stack_name="$1"

  case "$stack_name" in
    legacy-cu124)
      echo "Installing legacy CUDA 12.4 torch stack: ${LEGACY_TORCH_PACKAGES[*]}"
      uv pip install --python .venv/bin/python \
        --index-url "$LEGACY_TORCH_INDEX" \
        "${LEGACY_TORCH_PACKAGES[@]}"
      ;;
    modern-default)
      echo "Installing modern torch stack: ${MODERN_TORCH_PACKAGES[*]}"
      uv pip install --python .venv/bin/python \
        "${MODERN_TORCH_PACKAGES[@]}"
      ;;
    *)
      echo "Unknown torch stack: $stack_name" >&2
      return 1
      ;;
  esac
}

torch_supports_active_gpu() {
  uv run --python "$UV_PYTHON" python -W ignore - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    raise SystemExit(0)

device = torch.device("cuda")
cc = torch.cuda.get_device_capability(device)
arch = f"sm_{cc[0]}{cc[1]}"
supported = set(torch.cuda.get_arch_list())

if supported:
  if arch in supported:
    raise SystemExit(0)

  supported_ccs = []
  for candidate in supported:
    if not candidate.startswith("sm_"):
      continue
    suffix = candidate.removeprefix("sm_")
    if not suffix.isdigit():
      continue
    supported_ccs.append((int(suffix[:-1]), int(suffix[-1])))

  if any(major == cc[0] and minor <= cc[1] for major, minor in supported_ccs):
    raise SystemExit(0)

    print(
        f"Resolved torch build does not support {arch}; available arches: {', '.join(sorted(supported))}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

SHARD_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" shards.txt)
SHARD_BASENAME=$(basename "$SHARD_PATH")

MODEL_SRC="$SCRATCH/qwen3-vl-8b-instruct"

SHARD_DST="$SLURM_TMPDIR/$SHARD_BASENAME"

echo "Copying shard to local disk..."
cp "$SHARD_PATH" "$SHARD_DST"

echo "Running captioning..."
uv sync --python "$UV_PYTHON"

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
install_torch_stack "$TORCH_STACK"

if torch_supports_active_gpu; then
  echo "Installed torch build is compatible with the active GPU."
else
  echo "Installed torch build is incompatible with compute capability $GPU_CC after selecting stack $TORCH_STACK." >&2
  exit 1
fi

uv run --python "$UV_PYTHON" python caption_shard.py \
  --shard_path "$SHARD_DST" \
  --output_dir "outputs" \
  --model_dir "$MODEL_SRC" \
  --shard_size 1000

  