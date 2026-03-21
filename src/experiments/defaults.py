from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SEED = 17


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    eval_ratio: float = 0.1


@dataclass(frozen=True)
class LoraDefaults:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"


@dataclass(frozen=True)
class SFTDefaults:
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 100
    max_seq_length: int = 2048


@dataclass(frozen=True)
class GRPODefaults:
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    logging_steps: int = 5
    save_steps: int = 50
    max_prompt_length: int = 1536
    max_completion_length: int = 256
    num_generations: int = 4
    temperature: float = 0.8
    top_p: float = 0.95


@dataclass(frozen=True)
class EvalDefaults:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    best_of_n: int = 4
