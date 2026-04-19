from __future__ import annotations

import os
from dataclasses import asdict


def setup_wandb(config):
    """Configure and init W&B. Offline by default — no Kaggle internet required.

    Returns the wandb run object, or None when use_wandb=False.
    """
    if not config.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    os.environ["WANDB_MODE"] = config.wandb_mode
    os.environ["WANDB_DIR"] = config.wandb_dir
    os.makedirs(config.wandb_dir, exist_ok=True)

    if config.wandb_mode == "online" and "WANDB_API_KEY" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            pass  # local dev: assume `wandb login` already done

    import wandb  # imported late so offline runs don't require network on import

    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        tags=config.wandb_tags,
        config=asdict(config),
    )
