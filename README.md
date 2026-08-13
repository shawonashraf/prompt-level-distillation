# Prompt-Level Distillation

This project is an implementation of Sanket Badhe and Deep Shah, "Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning," arXiv:2602.21103, http://arxiv.org/abs/2602.21103. It is a four-phase pipeline that extracts reasoning patterns from a teacher LLM and distills them into a structured instruction list for a student model's system prompt. It runs entirely against local LLM endpoints (e.g., LM Studio) via LiteLLM and tracks token consumption, request counts, and evaluation metrics through Weights & Biases.

## Run

```bash
# Install dependencies
uv sync

# Set your W&B API key
export WANDB_API_KEY=your-key-here

# Run the full pipeline with the default config
uv run python main.py -c configs/config.yaml

# Or use the small example config for a quick local test
uv run python main.py -c configs/example.yaml
```

## Track experiments

All runs are set up to be logged to Weights & Biases.

### Completed Runs

Report on the completed run using Qwen 3.6 35B A3B on Snellius: https://api.wandb.ai/links/shawonashraf/yn47mza1


