from __future__ import annotations

import os

from trl import GRPOConfig, GRPOTrainer

from .callbacks import PrintLossCallback, SaveAdapterCallback
from .config import GRPOExperimentConfig
from .data import load_csv_dataset
from .model import apply_lora, load_model_and_tokenizer
from .rewards import resolve_rewards
from .wandb_utils import setup_wandb

_OPTIONAL_GRPO_KEYS = [
    "beta",
    "max_grad_norm",
    "warmup_steps",
    "weight_decay",
    "lr_scheduler_type",
    "optim",
    "gradient_checkpointing",
]


def _nonnull_kwargs(config: GRPOExperimentConfig, keys: list[str]) -> dict:
    return {k: getattr(config, k) for k in keys if getattr(config, k) is not None}


def build_grpo_trainer(model, tokenizer, dataset, config: GRPOExperimentConfig) -> GRPOTrainer:
    os.makedirs(config.output_dir, exist_ok=True)

    grpo_kwargs = dict(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        epsilon=config.epsilon,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.use_wandb else "none",
        remove_unused_columns=False,
    )
    grpo_kwargs.update(_nonnull_kwargs(config, _OPTIONAL_GRPO_KEYS))

    grpo_config = GRPOConfig(**grpo_kwargs)

    tokenizer.padding_side = "left"

    return GRPOTrainer(
        model=model,
        reward_funcs=resolve_rewards(config),
        train_dataset=dataset,
        processing_class=tokenizer,
        args=grpo_config,
        callbacks=[
            PrintLossCallback(config.log_phase_name),
            SaveAdapterCallback(config.output_dir, config.save_every_n_steps),
        ],
    )


def run_experiment(config: GRPOExperimentConfig):
    """End-to-end: load data → load model → apply LoRA → train → save adapter."""
    run = setup_wandb(config)

    try:
        dataset = load_csv_dataset(config)
        model, tokenizer = load_model_and_tokenizer(config)
        model = apply_lora(model, config)
        trainer = build_grpo_trainer(model, tokenizer, dataset, config)
        trainer.train()
        model.save_pretrained(config.output_dir)
        print(f"Adapter saved to {config.output_dir}")
    finally:
        if run is not None:
            import wandb
            wandb.finish()

    return model
