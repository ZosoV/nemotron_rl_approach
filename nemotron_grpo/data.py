from __future__ import annotations

import polars as pl
from datasets import Dataset

from .config import GRPOExperimentConfig


def load_csv_dataset(config: GRPOExperimentConfig) -> Dataset:
    df = pl.read_csv(config.train_csv)

    if config.num_train_samples is not None:
        df = df.sample(n=min(config.num_train_samples, len(df)), seed=42)

    records = {
        "prompt": df[config.prompt_column].to_list(),
        "ground_truth": df[config.answer_column].to_list(),
    }
    dataset = Dataset.from_dict(records)
    dataset = dataset.map(_format_prompt(config.boxed_instruction))
    return dataset


def _format_prompt(boxed_instruction: str):
    def _inner(example):
        example["prompt"] = [
            {"role": "user", "content": example["prompt"] + boxed_instruction}
        ]
        return example
    return _inner
