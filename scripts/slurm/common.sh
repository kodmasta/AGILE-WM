#!/usr/bin/env bash

# Shared helpers for Slurm-oriented entrypoints under scripts/slurm/.

LEGACY_TORCH_INDEX="https://download.pytorch.org/whl/cu124"
LEGACY_TORCH_PACKAGES=(torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0)
MODERN_TORCH_PACKAGES=(torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0)
SLURM_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

setup_slurm_runtime() {
  export PATH="$HOME/.local/bin:$PATH"

  SCRIPT_DIR="$SLURM_COMMON_DIR"
  REPO_ROOT="$(cd "$SLURM_COMMON_DIR/../.." && pwd)"
  AGILE_WM_ARTIFACTS_ROOT="${AGILE_WM_ARTIFACTS_ROOT:-$REPO_ROOT/artifacts}"
  LOG_DIR="$AGILE_WM_ARTIFACTS_ROOT/logs"

  cd "$REPO_ROOT"

  if command -v module >/dev/null 2>&1; then
    module load python/3.10
  fi

  export AGILE_WM_ARTIFACTS_ROOT
  export HF_HOME="${HF_HOME:-$AGILE_WM_ARTIFACTS_ROOT/cache/hf_home}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$AGILE_WM_ARTIFACTS_ROOT/cache/hf_cache}"
}

ensure_artifact_dirs() {
  mkdir -p "$LOG_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$@"
}

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
  local python_bin="$1"
  local stack_name="$2"

  case "$stack_name" in
    legacy-cu124)
      echo "Installing legacy CUDA 12.4 torch stack: ${LEGACY_TORCH_PACKAGES[*]}"
      uv pip install --python "$python_bin" \
        --index-url "$LEGACY_TORCH_INDEX" \
        "${LEGACY_TORCH_PACKAGES[@]}"
      ;;
    modern-default)
      echo "Installing modern torch stack: ${MODERN_TORCH_PACKAGES[*]}"
      uv pip install --python "$python_bin" \
        "${MODERN_TORCH_PACKAGES[@]}"
      ;;
    *)
      echo "Unknown torch stack: $stack_name" >&2
      return 1
      ;;
  esac
}

torch_supports_active_gpu() {
  local python_bin="$1"

  "$python_bin" -W ignore - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch after installing the selected stack: {exc}", file=sys.stderr)
    raise SystemExit(2)

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

raise SystemExit(0)
PY
}
