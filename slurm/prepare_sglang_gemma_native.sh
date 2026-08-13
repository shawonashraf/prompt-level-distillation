#!/usr/bin/env bash
# Run ONCE on the Snellius LOGIN node. Gemma-4 variant of
# prepare_sglang_native.sh: builds a SEPARATE venv, because per the SGLang
# cookbook (docs.sglang.io/cookbook/autoregressive/Google/Gemma4) gemma-4
# needs sglang from the main branch plus a PINNED transformers commit
# ("encoder-free unified family" support) -- pins that would break the
# accepted qwen venv. Never share or mutate a venv another job may use.
#
# Usage:
#   bash slurm/prepare_sglang_gemma_native.sh
#   DRY_RUN=1 bash slurm/prepare_sglang_gemma_native.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-}"
run() { if [ -n "$DRY_RUN" ]; then echo "[dry-run] $*"; else "$@"; fi; }

SGLANG_MODEL="${SGLANG_MODEL:-google/gemma-4-E2B-it}"
# Cookbook pins (2026-08-12). Both are pure-python installs -- no CUDA
# compiles. Update TRANSFORMERS_PIN only from the cookbook page.
SGLANG_GIT_SPEC="${SGLANG_GIT_SPEC:-git+https://github.com/sgl-project/sglang.git#subdirectory=python}"
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-git+https://github.com/huggingface/transformers.git@1423d22f7a3b62e8c70ad67b58ec25cd9b675897}"

# Venvs on /projects, never scratch (platform deletions, 2026-08-12).
if [ -z "${NATIVE_ROOT:-}" ]; then
    NATIVE_ROOT="${HOME}/projects/prompt-distill/venvs"
    if ! mkdir -p "$NATIVE_ROOT" 2>/dev/null || [ ! -w "$NATIVE_ROOT" ]; then
        echo "[prep-gemma] ERROR: cannot write ${NATIVE_ROOT}. Override: NATIVE_ROOT=... bash $0" >&2
        exit 1
    fi
fi
if [ -z "${HF_CACHE_DIR:-}" ]; then
    HF_CACHE_DIR="${HOME}/.cache/huggingface"
    if ! mkdir -p "$HF_CACHE_DIR" 2>/dev/null || [ ! -w "$HF_CACHE_DIR" ]; then
        echo "[prep-gemma] ERROR: cannot write ${HF_CACHE_DIR}. Override: HF_CACHE_DIR=... bash $0" >&2
        exit 1
    fi
fi

echo "[prep-gemma] NATIVE_ROOT  = ${NATIVE_ROOT}"
echo "[prep-gemma] HF_CACHE_DIR = ${HF_CACHE_DIR}"
echo "[prep-gemma] SGLANG_MODEL = ${SGLANG_MODEL}"

if [ -z "$DRY_RUN" ]; then
    module purge 2>/dev/null || true
    module load 2024 2>/dev/null || module load 2023 2>/dev/null || true
    module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
        || module load Python 2>/dev/null || true
    command -v python3 >/dev/null 2>&1 || { echo "[prep-gemma] ERROR: no python3 after module load." >&2; exit 1; }
fi

VENV="${NATIVE_ROOT}/sglang-gemma-venv"
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "[prep-gemma] ERROR: uv not found." >&2; exit 1; }
if [ ! -x "${VENV}/bin/python" ]; then
    run uv venv --python python3 "$VENV"
fi
# sglang[all] from main pulls the prebuilt kernel wheels (sgl-kernel etc.);
# transformers pin applied AFTER so it wins the resolution.
# SGLANG_BUILD_RUST_EXTS=none: the main-branch source build otherwise
# demands cargo for optional Rust extensions (no Rust toolchain on the
# login node; the escape hatch is named in sglang's own build error).
export SGLANG_BUILD_RUST_EXTS=none
run uv pip install --python "${VENV}/bin/python" "sglang[all] @ ${SGLANG_GIT_SPEC}"
run uv pip install --python "${VENV}/bin/python" "transformers @ ${TRANSFORMERS_PIN}"
run uv pip check --python "${VENV}/bin/python" || echo "[prep-gemma] WARNING: uv pip check flagged conflicts (expected: the transformers pin overrides sglang's spec)." >&2
if [ -z "$DRY_RUN" ]; then
    uv pip freeze --python "${VENV}/bin/python" > "${NATIVE_ROOT}/sglang-gemma-manifest.txt"
    echo "[prep-gemma] manifest: ${NATIVE_ROOT}/sglang-gemma-manifest.txt"
    # Same cudart fix as prepare_sglang_native.sh: tvm_ffi JIT links
    # -L${CUDA_HOME}/lib64 -lcudart; pip wheels ship lib/ + .so.13 only.
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

# Weights: gemma-4-31B-it is already in the shared cache (59GB) -- this is
# a cache-hit no-op unless the snapshot is incomplete. Token stays in $HOME.
if [ -z "$DRY_RUN" ]; then
    export HF_HOME="$HF_CACHE_DIR"
    if [ -z "${HF_TOKEN:-}" ] && [ -s "${HOME}/.cache/huggingface/token" ]; then
        HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"; export HF_TOKEN
    fi
    uv pip install -q --python "${VENV}/bin/python" "huggingface_hub[cli]"
    # python -c, not bin/hf: entry-point shebangs bake absolute paths.
    "${VENV}/bin/python" -c "from huggingface_hub.cli.hf import main; main()" download "$SGLANG_MODEL"
else
    echo "[dry-run] HF_HOME=${HF_CACHE_DIR} hf download ${SGLANG_MODEL}"
fi

echo "[prep-gemma] Done. Submit with:  sbatch slurm/pld_e2b_student_eval.sh"
