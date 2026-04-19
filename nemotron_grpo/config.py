from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRPOExperimentConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    model_path: str
    train_csv: str = "inputs/train.csv"
    output_dir: str = "outputs/grpo_run"

    # ── Data ───────────────────────────────────────────────────────────────
    num_train_samples: Optional[int] = None
    prompt_column: str = "prompt"
    answer_column: str = "answer"
    boxed_instruction: str = (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

    # ── GRPO core (notebook 1 defaults) ────────────────────────────────────
    learning_rate: float = 3e-6
    per_device_train_batch_size: int = 16
    num_generations: int = 4
    gradient_accumulation_steps: int = 1
    max_completion_length: int = 7680
    temperature: float = 1.0
    epsilon: float = 10.0
    max_steps: int = 10
    logging_steps: int = 1

    # ── GRPO extras — None means TRL's own default applies ─────────────────
    beta: Optional[float] = None
    max_grad_norm: Optional[float] = None
    warmup_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    optim: Optional[str] = None
    gradient_checkpointing: Optional[bool] = None

    # ── Rewards ────────────────────────────────────────────────────────────
    reward_functions: list[str] = field(default_factory=lambda: [
        "cosine_reward", "format_reward", "length_reward",
    ])

    # ── Callbacks ──────────────────────────────────────────────────────────
    save_every_n_steps: int = 50
    log_phase_name: str = "GRPO"

    # ── Weights & Biases — offline by default ──────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "nemotron-grpo"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: str = "offline"
    wandb_dir: str = "/kaggle/working/wandb"
