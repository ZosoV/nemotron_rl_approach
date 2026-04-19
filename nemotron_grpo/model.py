from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GRPOExperimentConfig


def load_model_and_tokenizer(config: GRPOExperimentConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )
    print("Model loaded successfully.")
    model.print_trainable_parameters()
    return model, tokenizer


def apply_lora(model, config: GRPOExperimentConfig):
    print(f"Initializing LoRA adapter with rank={config.lora_rank}...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    print("Apply Lora successfully.")
    model.print_trainable_parameters()
    return model

def foo():
    print("testing githubg actions")