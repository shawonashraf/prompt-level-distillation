#!/usr/bin/env bash
# Run ONCE on the Snellius LOGIN node. SGLang counterpart of
# prepare_vllm_native.sh: builds a SEPARATE venv (never share a venv
# between serving stacks -- a mid-swap package once killed a running job
# on this project) and reuses the same persistent HF weights cache.
#
# All additive: the vLLM venv, scripts and configs stay untouched.
#
# Usage:
#   bash slurm/prepare_sglang_native.sh                         # qwen3.6 default
#   DRY_RUN=1 bash slurm/prepare_sglang_native.sh
#
# sglang>=0.5.10 is the Qwen-documented floor for Qwen3.6 (reasoning
# parser, qwen3_coder tool parser, GDN scheduling). Resolved versions are
# recorded to ${NATIVE_ROOT}/sglang-manifest.txt.

set -euo pipefail

DRY_RUN="${DRY_RUN:-}"
run() { if [ -n "$DRY_RUN" ]; then echo "[dry-run] $*"; else "$@"; fi; }

SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"

# Same directory rules as the vLLM path (project convention, 2026-08-10).
if [ -z "${NATIVE_ROOT:-}" ]; then
    # /projects, not /scratch-shared: scratch deletes hours-old files
    # platform-side (2026-08-12); must match _sglang_native_common.sh.
    NATIVE_ROOT="/projects/0/prjs2013/users/${USER}/pdistill_venv"
    if ! mkdir -p "$NATIVE_ROOT" 2>/dev/null || [ ! -w "$NATIVE_ROOT" ]; then
        echo "[prep-sglang] ERROR: cannot write ${NATIVE_ROOT}. Override: NATIVE_ROOT=... bash $0" >&2
        exit 1
    fi
fi
if [ -z "${HF_CACHE_DIR:-}" ]; then
    HF_CACHE_DIR="/projects/0/prjs2013/cache/huggingface"
    if ! mkdir -p "$HF_CACHE_DIR" 2>/dev/null || [ ! -w "$HF_CACHE_DIR" ]; then
        echo "[prep-sglang] ERROR: cannot write ${HF_CACHE_DIR}. Override: HF_CACHE_DIR=... bash $0" >&2
        exit 1
    fi
fi

echo "[prep-sglang] NATIVE_ROOT  = ${NATIVE_ROOT}"
echo "[prep-sglang] HF_CACHE_DIR = ${HF_CACHE_DIR}"
echo "[prep-sglang] SGLANG_MODEL = ${SGLANG_MODEL}"

if [ -z "$DRY_RUN" ]; then
    module purge 2>/dev/null || true
    module load 2024 2>/dev/null || module load 2023 2>/dev/null || true
    module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
        || module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
        || module load Python 2>/dev/null || true
    command -v python3 >/dev/null 2>&1 || { echo "[prep-sglang] ERROR: no python3 after module load ('module spider Python')." >&2; exit 1; }
    echo "[prep-sglang] python: $(command -v python3) ($(python3 --version 2>&1))"
fi

VENV="${NATIVE_ROOT}/sglang-venv"
# uv, not python -m venv + pip: uv's installer doesn't leave half-installed
# packages behind on interruption (three separate Lustre venv corruptions
# on 2026-08-12 traced to interrupted pip/uv-reinstall operations).
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "[prep-sglang] ERROR: uv not found." >&2; exit 1; }
if [ ! -x "${VENV}/bin/python" ]; then
    run uv venv --python python3 "$VENV"
fi
run uv pip install --python "${VENV}/bin/python" "sglang[all]>=0.5.10"
run uv pip check --python "${VENV}/bin/python"
if [ -z "$DRY_RUN" ]; then
    uv pip freeze --python "${VENV}/bin/python" > "${NATIVE_ROOT}/sglang-manifest.txt"
    echo "[prep-sglang] resolved: $(grep -iE '^sglang==' "${NATIVE_ROOT}/sglang-manifest.txt" || echo 'sglang (see sglang-manifest.txt)')"
    # SGLang's JIT kernels (tvm_ffi) link with -L${CUDA_HOME}/lib64 -lcudart,
    # but the pip cu13 wheel ships lib/ (no lib64/) and only the versioned
    # libcudart.so.13 (no dev symlink). Job 25490351: nvcc compile passed,
    # link failed "cannot find -lcudart". Two symlinks fix it, no rebuild.
    for CUDIR in "${VENV}"/lib/python*/site-packages/nvidia/cu*; do
        [ -d "${CUDIR}/lib" ] || continue
        [ -e "${CUDIR}/lib64" ] || ln -sfn lib "${CUDIR}/lib64"
        # || true: under pipefail, ls's exit 2 on no-match propagates
        # through the pipe and a failing assignment kills a set -e script
        RT="$(ls "${CUDIR}"/lib/libcudart.so.* 2>/dev/null | head -1)" || true
        # if, not a && chain: under set -e a false && leg kills the script
        if [ -n "$RT" ] && [ ! -e "${CUDIR}/lib/libcudart.so" ]; then
            ln -sf "$(basename "$RT")" "${CUDIR}/lib/libcudart.so"
        fi
    done
fi

# Weights: same shared cache the vLLM path populated -- for the default
# qwen model this is a cache-hit no-op. Token stays in $HOME (HF_TOKEN
# env), never on scratch.
if [ -z "$DRY_RUN" ]; then
    export HF_HOME="$HF_CACHE_DIR"
    if [ -z "${HF_TOKEN:-}" ] && [ -s "${HOME}/.cache/huggingface/token" ]; then
        HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"; export HF_TOKEN
    fi
    uv pip install -q --python "${VENV}/bin/python" "huggingface_hub[cli]"
    # python -m, not bin/hf: entry-point shebangs bake in absolute paths
    # and break when a venv is relocated (as this one was, scratch->projects).
    "${VENV}/bin/python" -c "from huggingface_hub.cli.hf import main; main()" download "$SGLANG_MODEL"
else
    echo "[dry-run] HF_HOME=${HF_CACHE_DIR} hf download ${SGLANG_MODEL}"
fi

echo "[prep-sglang] Done. Submit with:  sbatch slurm/native_qwen3_6_sglang_debug.sh"
