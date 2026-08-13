#!/bin/bash
#SBATCH --job-name=pld-qwen36-sglang
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/sashraf/projects/prompt-distill/slurm-%x-%j.out

# Prompt-Level Distillation full run against SGLang serving
# Qwen/Qwen3.6-35B-A3B-FP8 on one H100 — container-free path per the
# sglang-snellius-bare skill (Apptainer broken cluster-wide; vLLM JIT
# unbuildable on pip CUDA wheels).
#
# PREREQUISITES (once, on the login node — compute nodes have no
# package-fetch internet):
#   bash slurm/prepare_sglang_native.sh
#   UV_PROJECT_ENVIRONMENT=$HOME/projects/prompt-distill/venvs/pipeline-venv \
#       uv sync --frozen
#   HF caches pre-warmed (contract-nli + all-MiniLM-L6-v2) into
#   $HOME/.cache/huggingface
#
# Submit from the repo root: sbatch slurm/pld_qwen36_sglang.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export NATIVE_ROOT="${HOME}/projects/prompt-distill/venvs"
PIPELINE_VENV="${NATIVE_ROOT}/pipeline-venv"

# wandb key first: a missing key must abort at second zero, not hour 12
source "$HOME/.pld_env"
[ -n "${WANDB_API_KEY:-}" ] || { echo "[job] ERROR: WANDB_API_KEY missing from ~/.pld_env" >&2; exit 1; }

echo "============================================================"
echo "PLD full run: Qwen3.6-35B-A3B-FP8 (teacher+student) via SGLang"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    $(hostname)"
echo "  Started: $(date)"
echo "============================================================"

module purge 2>/dev/null || true
module load 2024 2>/dev/null || module load 2023 2>/dev/null || true
module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
    || module load Python 2>/dev/null || true

# preflight the pipeline venv BEFORE the server burns GPU minutes
"${PIPELINE_VENV}/bin/python" -c \
    "import litellm, sentence_transformers, wandb, sklearn, datasets, jinja2" \
    || { echo "[job] ERROR: pipeline venv failed import preflight — run 'UV_PROJECT_ENVIRONMENT=${PIPELINE_VENV} uv sync --frozen' on the login node." >&2; exit 1; }

export SGLANG_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
export SGLANG_TP_SIZE=1

# starts the server, waits for /health, exports SGLANG_PID + VLLM_BASE_URL
source "${PROJECT_DIR}/slurm/_sglang_native_common.sh"

cleanup() {
    if [ -n "${SGLANG_PID:-}" ]; then
        echo "[job] Stopping SGLang server (pid ${SGLANG_PID})..."
        kill "$SGLANG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[job] SGLang endpoint ready at ${VLLM_BASE_URL}"

# HF strictly offline (caches pre-warmed on the login node); wandb online
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

"${PIPELINE_VENV}/bin/python" main.py -c configs/snellius.yaml -v
STATUS=$?
echo "[job] Pipeline exited with status ${STATUS} at $(date)"
exit $STATUS
