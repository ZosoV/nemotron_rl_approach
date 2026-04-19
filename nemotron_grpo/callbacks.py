from __future__ import annotations

import os
import shutil
import time
import zipfile

from transformers import TrainerCallback


class PrintLossCallback(TrainerCallback):
    """Logs step / loss / lr / GRPO-specific metrics with elapsed time."""

    def __init__(self, phase: str = "GRPO"):
        self.phase = phase
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{self.phase}] Training started — {state.max_steps} steps")
        print(f"{'=' * 60}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        elapsed = time.time() - self.start_time if self.start_time else 0
        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")
        parts = [f"[{self.phase}] step {state.global_step}/{state.max_steps}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        for key in [
            "reward", "reward_mean", "rewards/mean", "completion_length",
            "completions/mean_length", "completions/clipped_ratio", "kl",
        ]:
            if key in logs:
                parts.append(f"{key}={logs[key]:.4f}")
        parts.append(f"elapsed={elapsed / 60:.1f}min")
        print(" | ".join(parts), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(
            f"\n[{self.phase}] Training complete — "
            f"{state.global_step} steps in {elapsed / 60:.1f}min",
            flush=True,
        )


class SaveAdapterCallback(TrainerCallback):
    """Saves the LoRA adapter and zips it every N steps."""

    def __init__(self, output_dir: str, save_every: int = 50):
        self.output_dir = output_dir
        self.save_every = save_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.save_every == 0 and state.global_step > 0:
            model.save_pretrained(self.output_dir)
            zip_path = f"submission_step{state.global_step}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(self.output_dir):
                    zf.write(os.path.join(self.output_dir, fname), fname)
            shutil.copy2(zip_path, "submission.zip")
            print(
                f"[CHECKPOINT] Saved {zip_path} at step {state.global_step}",
                flush=True,
            )
