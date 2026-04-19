from __future__ import annotations

import math
import re
from typing import Callable


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    s = s.strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return s


def _extract_boxed(content: str):
    match = re.search(r'\\boxed\{([^}]*)\}', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'boxed\{([^}]*)\}', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _get_content(completion) -> str:
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


# ── Reward factories ───────────────────────────────────────────────────────────

_debug_counter: dict[str, int] = {"calls": 0}


def build_cosine_reward(max_completion_length: int) -> Callable:
    """Cosine-scaled accuracy reward (from Light-R1).

    Correct answers: reward decays 1.0 → 0.1 as length approaches max.
    Incorrect answers: reward decays -0.1 → -1.0 as length approaches max.
    No \\boxed{}: penalized proportional to length.
    """
    def cosine_reward(completions, ground_truth, **kwargs):
        _debug_counter["calls"] += 1
        show_debug = _debug_counter["calls"] <= 2
        rewards = []

        for i, (completion, gt) in enumerate(zip(completions, ground_truth)):
            content = _get_content(completion)
            extracted = _extract_boxed(content)
            clen = len(content)
            progress = min(clen / max(max_completion_length, 1), 1.0)
            cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

            if extracted is not None and _normalize_answer(extracted) == _normalize_answer(gt):
                reward = 0.1 + 0.9 * cos_scale
            elif extracted is not None:
                reward = -0.1 - 0.9 * (1.0 - cos_scale)
            else:
                reward = -0.5 * progress

            rewards.append(reward)

            if show_debug and i < 2:
                tail = content[-120:] if len(content) > 120 else content
                print(
                    f"  [COSINE REWARD batch={_debug_counter['calls']}] "
                    f"extracted={extracted!r}, gt={str(gt).strip()!r}, "
                    f"reward={reward:.3f}, len={clen}, tail={tail!r}",
                    flush=True,
                )

        return rewards

    return cosine_reward


def build_length_reward(max_completion_length: int) -> Callable:
    """Length penalty: 0.0 for empty → -1.0 at max_completion_length."""
    def length_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            content = _get_content(completion)
            progress = min(len(content) / max(max_completion_length, 1), 1.0)
            rewards.append(-progress)
        return rewards

    return length_reward


def format_reward(completions, **kwargs):
    """Binary format reward: 1.0 if \\boxed{} present, 0.0 otherwise."""
    return [
        1.0 if _extract_boxed(_get_content(c)) is not None else 0.0
        for c in completions
    ]


# ── Registry ───────────────────────────────────────────────────────────────────

def _get_accuracy_reward():
    from trl.rewards import accuracy_reward
    return accuracy_reward


REGISTRY: dict[str, Callable] = {
    "accuracy_reward": lambda cfg: _get_accuracy_reward(),
    "cosine_reward": lambda cfg: build_cosine_reward(cfg.max_completion_length),
    "format_reward": lambda cfg: format_reward,
    "length_reward": lambda cfg: build_length_reward(cfg.max_completion_length),
}


def resolve_rewards(config) -> list[Callable]:
    missing = [name for name in config.reward_functions if name not in REGISTRY]
    if missing:
        raise ValueError(f"Unknown reward function(s): {missing}. Available: {list(REGISTRY)}")
    return [REGISTRY[name](config) for name in config.reward_functions]
