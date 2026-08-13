#!/bin/bash
#SBATCH --job-name=pld-e2b-student-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/sashraf/projects/prompt-distill/slurm-%x-%j.out

# Paper-faithful student evaluation: serve google/gemma-4-E2B-it via the
# gemma sglang venv and run ONLY phase 4 with the distilled instructions
# produced by the Qwen3.6-35B teacher run (job 25550458).
#
# PREREQUISITES (once, login node):
#   bash slurm/prepare_sglang_gemma_native.sh
#
# Submit from the repo root: sbatch slurm/pld_e2b_student_eval.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export NATIVE_ROOT="${HOME}/projects/prompt-distill/venvs"
PIPELINE_VENV="${NATIVE_ROOT}/pipeline-venv"

source "$HOME/.pld_env"
[ -n "${WANDB_API_KEY:-}" ] || { echo "[job] ERROR: WANDB_API_KEY missing from ~/.pld_env" >&2; exit 1; }

echo "============================================================"
echo "PLD student eval: gemma-4-E2B-it with the distilled prompt"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    $(hostname)"
echo "  Started: $(date)"
echo "============================================================"

module purge 2>/dev/null || true
module load 2024 2>/dev/null || module load 2023 2>/dev/null || true
module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
    || module load Python 2>/dev/null || true

"${PIPELINE_VENV}/bin/python" -c \
    "import litellm, sentence_transformers, wandb, sklearn, datasets, jinja2" \
    || { echo "[job] ERROR: pipeline venv failed import preflight" >&2; exit 1; }

# stage the teacher run's distilled instructions into this eval's output dir
STUDENT_OUT="/home/sashraf/projects/prompt-distill/output_student"
mkdir -p "$STUDENT_OUT"
cp /home/sashraf/projects/prompt-distill/output/conflict_resolution.json "$STUDENT_OUT/" \
    || { echo "[job] ERROR: teacher run's conflict_resolution.json not found" >&2; exit 1; }

export SGLANG_MODEL="google/gemma-4-E2B-it"
export SGLANG_TP_SIZE=1
export SGLANG_VENV="${NATIVE_ROOT}/sglang-gemma-venv"
export SGLANG_REASONING_PARSER="gemma4"
export SGLANG_TOOL_CALL_PARSER="gemma4"
export SGLANG_PORT=18472

if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" > /dev/null 2>&1; then
    echo "[job] ERROR: port ${SGLANG_PORT} already serving on this node" >&2
    exit 1
fi

source "${PROJECT_DIR}/slurm/_sglang_native_common.sh"

cleanup() {
    if [ -n "${SGLANG_PID:-}" ]; then
        echo "[job] Stopping SGLang server (pid ${SGLANG_PID})..."
        kill "$SGLANG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

kill -0 "$SGLANG_PID" 2>/dev/null || { echo "[job] ERROR: our SGLang process died" >&2; exit 1; }
echo "[job] SGLang endpoint ready at ${VLLM_BASE_URL} (pid ${SGLANG_PID})"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

"${PIPELINE_VENV}/bin/python" main.py -c configs/snellius_student_eval.yaml
STATUS=$?
echo "[job] Student eval exited with status ${STATUS} at $(date)"
exit $STATUS
