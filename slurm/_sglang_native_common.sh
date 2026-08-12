#!/usr/bin/env bash
# SGLang counterpart of _vllm_native_common.sh: starts an OpenAI-compatible
# SGLang server from the venv prepare_sglang_native.sh built. Sourced, not
# executed. Exports SGLANG_PID and VLLM_BASE_URL (the same variable name
# the job scripts already consume -- our stack only needs an
# OpenAI-compatible endpoint and does not care who serves it).
#
# Carries the two hard-won native-path fixes from the vLLM bootstrap:
# bundled nvidia lib dirs on LD_LIBRARY_PATH (mixed cu12/cu13 wheel sets)
# and CUDA_HOME pointed at the wheel-bundled toolkit (SGLang uses
# FlashInfer as its default attention backend, which JIT-compiles with
# nvcc -- the exact failure that cost the vLLM path two job starts).
#
# PREREQUISITE (once, login node): bash slurm/prepare_sglang_native.sh

set -euo pipefail

SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
# Qwen-documented parsers for Qwen3.6 (sglang>=0.5.10). Both are plain
# launch flags. Known upstream wart (sglang#9184): tool-call tags can
# leak into content -- harmless here, the agents parse JSON from content.
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
# pytorch, not flashinfer: FlashInfer JIT-compiles its sampling kernel at
# the FIRST real request (startup+warmup don't touch it), and that compile
# dies on pip-wheel nvcc vs bundled-CCCL header clash (job 25495380, 15 min
# in). Torch-native sampling needs no JIT.
SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
SGLANG_HEALTH_TIMEOUT_SECS="${SGLANG_HEALTH_TIMEOUT_SECS:-2400}"

# /projects, not /scratch-shared: scratch was observed deleting hours-old
# venv files platform-side (2026-08-12, see the SURF ticket evidence) --
# venvs live on project storage now, alongside the containers and HF cache.
NATIVE_ROOT="${NATIVE_ROOT:-/projects/0/prjs2013/users/${USER}/pdistill_venv}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/projects/0/prjs2013/cache/huggingface}"
# Overridable: model families with conflicting pins get their own venv
# (gemma-4 needs sglang@main + a pinned transformers commit; qwen runs on
# the release). Never share or mutate a venv another job may be using.
VENV="${SGLANG_VENV:-${NATIVE_ROOT}/sglang-venv}"

if [ ! -x "${VENV}/bin/python" ] || [ ! -d "$HF_CACHE_DIR" ] || [ -z "$(ls -A "$HF_CACHE_DIR" 2>/dev/null)" ]; then
    echo "[sglang-native] ERROR: venv (${VENV}) or model cache (${HF_CACHE_DIR}) missing/empty."
    echo "[sglang-native] Run ONCE on the login node first: bash slurm/prepare_sglang_native.sh"
    exit 1
fi

echo "[sglang-native] venv: ${VENV} ($(grep -iE '^sglang==' "${NATIVE_ROOT}/sglang-manifest.txt" 2>/dev/null || echo 'version unrecorded'))"
echo "[sglang-native] Starting: ${SGLANG_MODEL} (TP=${SGLANG_TP_SIZE}, mem_fraction=${SGLANG_MEM_FRACTION}, port=${SGLANG_PORT})"

SGLANG_LOG="${NATIVE_ROOT}/sglang_server_${SLURM_JOB_ID}.log"

NVIDIA_LIB_DIRS="$(find "${VENV}/lib" -type d -name lib -path "*/nvidia/*" 2>/dev/null | paste -sd: -)"
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[sglang-native] LD_LIBRARY_PATH includes $(echo "$NVIDIA_LIB_DIRS" | tr ":" "\n" | wc -l | tr -d " ") bundled nvidia lib dirs"
fi
BUNDLED_NVCC="$(find "${VENV}/lib" -type f -name nvcc -path "*/nvidia/*" 2>/dev/null | head -1)"
if [ -n "$BUNDLED_NVCC" ] && [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME="$(dirname "$(dirname "$BUNDLED_NVCC")")"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    # pip splits CUDA into per-library wheels, so headers like curand.h
    # live outside the toolkit dir nvcc -I's by default (job 25488462:
    # "fatal error: curand.h: No such file or directory"). nvcc honors
    # CPATH for includes and LIBRARY_PATH for the link stage.
    NVIDIA_INC_DIRS="$(find "${VENV}/lib" -type d -name include -path "*/nvidia/*" 2>/dev/null | paste -sd: -)"
    [ -n "$NVIDIA_INC_DIRS" ] && export CPATH="${NVIDIA_INC_DIRS}${CPATH:+:$CPATH}"
    [ -n "${NVIDIA_LIB_DIRS:-}" ] && export LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    echo "[sglang-native] CUDA_HOME=${CUDA_HOME} (bundled toolkit, for JIT compilers)"
fi

EXTRA_ARGS=()
if [ -n "$SGLANG_EXTRA_ARGS" ]; then
    read -ra EXTRA_ARGS <<< "$SGLANG_EXTRA_ARGS"
fi

# DeepGEMM's own JIT (separate from tvm_ffi) invokes nvcc in a way the
# pip-wheel toolkit can't satisfy (job 25491667: "NVCC compilation
# failed"). Disable it -- SGLang's FP8 path has a documented Triton
# fallback (deepgemm_w8a8_block_fp8_linear_with_fallback) needing no nvcc.
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

HF_HOME="$HF_CACHE_DIR" HF_HUB_OFFLINE=1 \
    "${VENV}/bin/python" -m sglang.launch_server \
        --model-path "$SGLANG_MODEL" \
        --tp-size "$SGLANG_TP_SIZE" \
        --mem-fraction-static "$SGLANG_MEM_FRACTION" \
        --reasoning-parser "$SGLANG_REASONING_PARSER" \
        --tool-call-parser "$SGLANG_TOOL_CALL_PARSER" \
        --sampling-backend "$SGLANG_SAMPLING_BACKEND" \
        "${EXTRA_ARGS[@]}" \
        --host 0.0.0.0 \
        --port "$SGLANG_PORT" \
        > "$SGLANG_LOG" 2>&1 &

SGLANG_PID=$!
echo "[sglang-native] Server starting (pid ${SGLANG_PID}), logging to ${SGLANG_LOG}"

echo "[sglang-native] Waiting for health (timeout: ${SGLANG_HEALTH_TIMEOUT_SECS}s)..."
elapsed=0
poll_interval=10
while true; do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "[sglang-native] ERROR: server exited before becoming healthy. Last 100 log lines:"
        tail -n 100 "$SGLANG_LOG"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        echo "[sglang-native] Healthy after ${elapsed}s"
        break
    fi
    if [ "$elapsed" -ge "$SGLANG_HEALTH_TIMEOUT_SECS" ]; then
        echo "[sglang-native] ERROR: not healthy within ${SGLANG_HEALTH_TIMEOUT_SECS}s. Last 100 log lines:"
        tail -n 100 "$SGLANG_LOG"
        kill "$SGLANG_PID" 2>/dev/null || true
        exit 1
    fi
    sleep "$poll_interval"
    elapsed=$((elapsed + poll_interval))
done

export SGLANG_PID
export VLLM_BASE_URL="http://127.0.0.1:${SGLANG_PORT}/v1"
