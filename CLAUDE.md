# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Implementation of "Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning" (Badhe & Shah, arXiv:2602.21103). A teacher LLM extracts reasoning rules from labeled examples; the rules are clustered, consolidated, refined against student errors, and injected into a student model's system prompt — no fine-tuning.

## Commands

```bash
uv sync                                            # install deps (Python >= 3.12)
uv run python main.py -c configs/example.yaml      # quick end-to-end run (5 samples)
uv run python main.py -c configs/config.yaml       # full run (50 samples)
uv run python main.py -c <cfg> --phase cluster     # run one phase (extract|cluster|conflict|evaluate)
uv run python main.py -c <cfg> -v                  # debug logging
```

- All LLM calls go to OpenAI-compatible endpoints (LM Studio at `http://localhost:1234/v1` by default). Verify a server is up with `curl -s http://localhost:1234/v1/models` and that the model named in the config's `teacher`/`student` sections is loaded.
- Runs log to Weights & Biases (`WANDB_API_KEY` env var). Use `WANDB_MODE=offline` to run without an account — always do this for test runs.
- There is no test suite or linter; verification is running the pipeline with `configs/example.yaml` against LM Studio.

## Architecture

`main.py` orchestrates four phases. Each phase writes a JSON artifact to `output_dir`, and `--phase <name>` re-runs a single phase by loading the previous phase's artifact — so keep those file schemas in sync with the dataclasses in `src/models.py` (they are loaded back via `ExtractionResult(**r)` / `ClusterResult(**c)`).

| Phase | Module | Reads | Writes |
|---|---|---|---|
| 1 extract | `src/phase1_extract.py` | train split | `extracted_instructions.json` |
| 2 cluster | `src/phase2_cluster.py` | phase 1 output | `clusters.json` |
| 3 conflict | `src/phase3_conflict.py` | phase 2 output + train split | `conflict_resolution.json` |
| 4 evaluate | `src/phase4_inference.py` + `src/evaluate.py` | phase 3 output + `eval_split` | `final_instructions.txt` |

Cross-cutting pieces:

- **`src/utils.py`** — single `chat_completion()` chokepoint for all LLM calls (LiteLLM, `openai/<model>` prefix), plus a global `LLMStats` token/request counter logged to wandb at the end. `parse_json_response()` tolerates markdown-fenced or prose-wrapped JSON; all teacher prompts demand JSON output.
- **`src/data.py`** — dataset registry. Datasets load from the HF `refs/convert/parquet` branch with `data_dir` as the config name (`contract-nli` → `kiddothe2b/contract-nli`/`contractnli_a`; `stereoset` → `McGill-NLP/stereoset`/`intersentence`). `DatasetExample._get_input_text/_get_hypothesis/_get_gold_label` are the only per-dataset field accessors — go through them, never raw fields, in phase code. StereoSet rows are flattened at load time (`bias_type` → `category`, stereotype/anti-stereotype sentences picked by `sentences.gold_label` 1/0) and only have a `validation` split.
- **`prompts/*.j2`** — Jinja2 templates, one per phase; phase 1 has one per dataset (`extract_<dataset>.j2`). Adding a dataset means: entry in `DATASET_MAP`/`SUBSET_MAP`/`LABEL_MAPS`, accessor cases in `DatasetExample`, and a new extract template selected in `phase1_extract.py`.
- **Evaluation** — `normalize_label()` maps free-text student output to a label by leftmost substring match; `compute_macro_f1()` keeps failed/empty predictions so they count as errors. Phase 4 evaluates on `dataset.eval_split` (held out); if unset it falls back to the train split and logs a leakage warning.

## Deviations from the paper

Deliberate, for local-scale runs — don't "fix" toward the paper without measuring:

- Teacher/synthesizer/student are all local LM Studio models (paper: Gemini 3 Flash/Pro teacher, Gemma-3 4B / Mistral Small students).
- Embeddings: `all-MiniLM-L6-v2` (paper: Gemini Embedding 768-dim). DBSCAN `eps=0.5, min_samples=2` works only at ~50-sample scale; at full scale (6,819 rules) MiniLM embeddings form one dense continuum and DBSCAN chains everything into a single cluster at every workable eps — use `clustering.algorithm: kmeans` with `n_clusters: 17` (measured; matches the paper's contract-nli cluster count) as in `configs/snellius.yaml`.
- Full-scale serving on Snellius is bare-metal SGLang (`slurm/`), not vLLM/Apptainer — see `slurm/pld_qwen36_sglang.sh` for the H100 job that produced the 0.76 test F1 run.
- Phase 3 convergence is measured on the train split (paper monitors a validation split).
- LogiQA (third task in the paper) is not implemented.
