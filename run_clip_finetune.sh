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
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SCRATCH/AGILE-WM/.venvs/clip-ft-py310}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu124}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"

SHARD_GLOB="${SHARD_GLOB:-shard-*-caption-*.tar}"

# Point this at the directory that contains caption shards such as
# shard-00000-caption-00000.tar. Update it if your caption tar files live elsewhere.
DEFAULT_DATA_ROOT="$SCRATCH/my_dataset/frame-caption-pairs-temp"
if [[ ! -d "$DEFAULT_DATA_ROOT" ]] || ! compgen -G "$DEFAULT_DATA_ROOT/$SHARD_GLOB" > /dev/null; then
  DEFAULT_DATA_ROOT="$SCRATCH/my_dataset/outputs"
fi
if [[ ! -d "$DEFAULT_DATA_ROOT" ]] || ! compgen -G "$DEFAULT_DATA_ROOT/$SHARD_GLOB" > /dev/null; then
  if compgen -G "$PWD/outputs/$SHARD_GLOB" > /dev/null; then
    DEFAULT_DATA_ROOT="$PWD/outputs"
  fi
fi

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/AGILE-WM/clip_finetune}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SLURM_TMPDIR:-/tmp}/clip_caption_shards}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"

mkdir -p logs "$HF_HOME" "$TRANSFORMERS_CACHE" "$OUTPUT_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Configured DATA_ROOT does not exist: $DATA_ROOT" >&2
  if compgen -G "$PWD/outputs/$SHARD_GLOB" > /dev/null; then
    echo "Local shard fallback is available at: $PWD/outputs" >&2
  fi
  exit 1
fi

if ! compgen -G "$DATA_ROOT/$SHARD_GLOB" > /dev/null; then
  echo "No shards matching '$SHARD_GLOB' were found in: $DATA_ROOT" >&2
  if compgen -G "$PWD/outputs/$SHARD_GLOB" > /dev/null; then
    echo "Local shard fallback is available at: $PWD/outputs" >&2
  fi
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "HF cache: $TRANSFORMERS_CACHE"
echo "Local shard dir: $LOCAL_DATA_DIR"
echo "UV env: $UV_PROJECT_ENVIRONMENT"
echo "PyTorch backend: $UV_TORCH_BACKEND"
echo "PyTorch versions: torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION"

extra_args=("$@")

uv venv --python 3.10 "$UV_PROJECT_ENVIRONMENT"

uv sync \
  --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  --frozen \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package torchaudio

# Remove the CUDA 13 / NCCL packages that conflict with PyTorch cu124 wheels.
uv pip uninstall \
  --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  cuda-toolkit \
  cuda-bindings \
  cuda-pathfinder \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx || true

uv pip install \
  --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  --torch-backend "$UV_TORCH_BACKEND" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

SITE_PACKAGES="$("$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import site; print(site.getsitepackages()[0])')"

# Force PyTorch to pick its own cu124 shared libraries first.
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cusparselt/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

"$UV_PROJECT_ENVIRONMENT/bin/python" -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"

"$UV_PROJECT_ENVIRONMENT/bin/python" CLIP_finetune.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --hf_cache_dir "$TRANSFORMERS_CACHE" \
  --clip_model "$CLIP_MODEL" \
  --shard_glob "$SHARD_GLOB" \
  --stage_shards_to_local \
  --local_data_dir "$LOCAL_DATA_DIR" \
  --save_every 1 \
  --num_workers 2 \
  "${extra_args[@]}"