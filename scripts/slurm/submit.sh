#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/submit_slurm.sh <job-script> [job-arg ...]

Submit a Slurm job while routing stdout/stderr into
AGILE_WM_ARTIFACTS_ROOT/logs at submit time.

Resolution order for AGILE_WM_ARTIFACTS_ROOT:
1. Existing AGILE_WM_ARTIFACTS_ROOT
2. $SCRATCH/AGILE-WM/artifacts if SCRATCH is set
3. <repo>/artifacts

Examples:
  scripts/submit_slurm.sh scripts/slurm/run_clip_finetune.sh
  scripts/submit_slurm.sh scripts/slurm/run_caption_array.sh
  scripts/submit_slurm.sh scripts/slurm/run_rnn_train_mila.sh -- --save_every 50

Everything after <job-script> is forwarded to the job script. A literal `--`
separator is optional and will be ignored by this wrapper if present.
EOF
}

resolve_path() {
  local path="$1"
  local dir

  if [[ -f "$path" ]]; then
    dir="$(cd "$(dirname "$path")" && pwd)"
    printf '%s/%s\n' "$dir" "$(basename "$path")"
    return 0
  fi

  if [[ -f "$REPO_ROOT/$path" ]]; then
    dir="$(cd "$REPO_ROOT/$(dirname "$path")" && pwd)"
    printf '%s/%s\n' "$dir" "$(basename "$path")"
    return 0
  fi

  return 1
}

extract_job_name() {
  local script_path="$1"
  local job_name

  job_name="$(
    sed -nE \
      -e 's/^#SBATCH[[:space:]]+--job-name[[:space:]]*=?([^[:space:]]+).*$/\1/p' \
      -e 's/^#SBATCH[[:space:]]+-J[[:space:]]+([^[:space:]]+).*$/\1/p' \
      "$script_path" | head -n 1
  )"

  if [[ -n "$job_name" ]]; then
    printf '%s\n' "$job_name"
  else
    basename "$script_path" .sh
  fi
}

has_array_directive() {
  local script_path="$1"
  grep -Eq '^#SBATCH[[:space:]]+(-a|--array)([[:space:]]|=)' "$script_path"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch was not found in PATH. Run this wrapper on a Slurm-enabled cluster." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_SCRIPT="$(resolve_path "$1" || true)"
if [[ -z "$JOB_SCRIPT" ]]; then
  echo "Could not find job script: $1" >&2
  exit 1
fi
shift

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ -n "${AGILE_WM_ARTIFACTS_ROOT:-}" ]]; then
  AGILE_WM_ARTIFACTS_ROOT="${AGILE_WM_ARTIFACTS_ROOT%/}"
elif [[ -n "${SCRATCH:-}" ]]; then
  AGILE_WM_ARTIFACTS_ROOT="$SCRATCH/AGILE-WM/artifacts"
else
  AGILE_WM_ARTIFACTS_ROOT="$REPO_ROOT/artifacts"
fi

LOG_DIR="$AGILE_WM_ARTIFACTS_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_NAME="$(extract_job_name "$JOB_SCRIPT")"
JOB_ID_PATTERN="%j"
if has_array_directive "$JOB_SCRIPT"; then
  JOB_ID_PATTERN="%A_%a"
fi

echo "Submitting $JOB_SCRIPT"
echo "Artifacts root: $AGILE_WM_ARTIFACTS_ROOT"
echo "Log dir: $LOG_DIR"

sbatch \
  --export="ALL,AGILE_WM_ARTIFACTS_ROOT=$AGILE_WM_ARTIFACTS_ROOT" \
  --output="$LOG_DIR/${JOB_NAME}_${JOB_ID_PATTERN}.out" \
  --error="$LOG_DIR/${JOB_NAME}_${JOB_ID_PATTERN}.err" \
  "$JOB_SCRIPT" "$@"
