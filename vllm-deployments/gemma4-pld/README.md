# Gemma-4 vLLM deployment for Prompt-Level Distillation (Snellius)

Serves **two** models on a **single H100 94GB** (`gpu_h100` partition) and runs the
distillation pipeline against them in the same job:

| Role | Model | Port | GPU mem | Why it fits |
|---|---|---|---|---|
| Teacher | `google/gemma-4-26B-A4B-it` (MoE, 4B active) | 8000 | 0.68 × 94G ≈ 64G | ~53G bf16 weights + KV |
| Student | `google/gemma-4-E2B-it` (5.1B) | 8001 | 0.18 × 94G ≈ 17G | ~10G bf16 weights + KV |

Total ≈ 0.86 GPU memory utilization — both fit on one H100, so tensor parallelism
is not needed (`--tensor-parallel-size 1`, the recipes' own recommendation).

Config source: official recipes at
[recipes.vllm.ai/Google/gemma-4-26B-A4B-it](https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it)
and [recipes.vllm.ai/Google/gemma-4-E2B-it](https://recipes.vllm.ai/Google/gemma-4-E2B-it)
(both verified on H100). Deviations from the recipe defaults:

- No tool-call parser / tool chat template — the pipeline uses plain chat completions.
- `--reasoning-parser gemma4` kept so reasoning tokens are separated and counted.
- `--limit-mm-per-prompt '{"image": 0, "audio": 0}'` — text-only workload.
- `--gpu-memory-utilization` split 0.68 / 0.18 so both servers share one GPU;
  the teacher starts first so vLLM's memory profiling sees a free device.
- `--max-model-len 65536` — the models allow 256K/128K, but a full-length KV
  cache is not guaranteed to fit beside the second model on a shared GPU. The
  pipeline sends `max_tokens: null`, so each request may generate up to the
  full remaining 64K window.

## Prerequisites (once)

- `~/.pld_env` on Snellius with `export WANDB_API_KEY=...` and `export HF_TOKEN=...` (chmod 600)
- Repo cloned/synced at `/scratch-shared/sashraf1/prompt-d/prompt-level-distillation`
  with its `.venv` built (`uv sync`)
- `HF_HOME` lives at `/scratch-shared/sashraf1/prompt-d/hf`

## Deploy

```bash
sbatch job.sh
squeue -u $USER                     # wait for R
tail -f /scratch-shared/sashraf1/prompt-d/slurm-<jobid>.out
```

The job pulls `vllm/vllm-openai:latest` on first run (cached as
`/scratch-shared/sashraf1/prompt-d/vllm-openai.sif`), starts both servers, waits for
health, runs `main.py -c configs/snellius.yaml`, and exits — servers are killed on
job exit, so no idle GPU burn.

## Smoke test (from the compute node)

```bash
./smoke_test.sh localhost:8000                            # teacher
./smoke_test.sh localhost:8001 google/gemma-4-E2B-it      # student
```

Cost note: `gpu_h100` is 192 SBU/GPU-hour; a full pipeline run is roughly 2–3 h ≈ 400–600 SBU.
