# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package manager

This project uses `uv`. Install deps and register the local package with:

```bash
uv sync
uv pip install -e .
```

Import check after any structural change:

```bash
python -c "from nemotron_grpo import GRPOExperimentConfig, run_experiment; print('ok')"
python -c "from nemotron_grpo.rewards import REGISTRY; print(list(REGISTRY))"
```

## Architecture

`nemotron_grpo/` is a flat Python package for GRPO fine-tuning of NVIDIA Nemotron. The entry points are:

- **`run_experiment(config)`** in `trainer.py` — end-to-end orchestrator. Calls setup_wandb → load_csv_dataset → load_model_and_tokenizer → apply_lora → build_grpo_trainer → train → save.
- **`build_grpo_trainer(...)`** in `trainer.py` — assembles `GRPOConfig` from `GRPOExperimentConfig`, wires reward functions and callbacks, returns a `GRPOTrainer`.

### Config (`config.py`)

`GRPOExperimentConfig` is the single source of truth for all knobs. Fields split into two categories:

- **Always-forwarded** to `GRPOConfig`: `learning_rate`, `per_device_train_batch_size`, `num_generations`, `gradient_accumulation_steps`, `max_completion_length`, `temperature`, `epsilon`, `max_steps`, `logging_steps`.
- **Optional (`None` by default)**: `beta`, `max_grad_norm`, `warmup_steps`, `weight_decay`, `lr_scheduler_type`, `optim`, `gradient_checkpointing`. These are only forwarded to `GRPOConfig` when explicitly set — otherwise TRL's own defaults apply. This is enforced in `trainer.py::_nonnull_kwargs`.

### Rewards (`rewards.py`)

`REGISTRY` maps string names to factory lambdas `(config) -> callable`. To add a new reward: implement a function with signature `(completions, ground_truth, **kwargs) -> list[float]` and add one entry to `REGISTRY`. The user selects rewards via `config.reward_functions` (list of string names); `resolve_rewards(config)` resolves them at trainer build time.

Available rewards: `accuracy_reward` (TRL built-in, imported lazily), `cosine_reward` (length-scaled accuracy, factory over `max_completion_length`), `format_reward` (binary `\boxed{}` check), `length_reward` (length penalty, factory over `max_completion_length`).

### W&B (`wandb_utils.py`)

`setup_wandb(config)` is offline-first: sets `WANDB_MODE` from config before any `wandb.init` call, so no network access occurs by default. API key is only fetched from Kaggle secrets when `wandb_mode="online"`. `run_experiment` wraps training in `try/finally` to always call `wandb.finish()`.

## Notebook

`notebooks/nvidia-nemotron-training.ipynb` is the Kaggle runner. It:
1. Sets up the Triton `ptxas-blackwell` binary (Kaggle-specific).
2. Installs TRL and W&B from offline Kaggle datasets.
3. Adds `/kaggle/input/nemotron-rl-approach` to `sys.path` so `nemotron_grpo` is importable without `pip install`.
4. Instantiates `GRPOExperimentConfig` and calls `run_experiment(config)`.

All Kaggle input paths, dataset names, and the full runbook for offline W&B sync are documented in `README.md`.
