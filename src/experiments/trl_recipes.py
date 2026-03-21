from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .formatting import chat_example, sft_target_text
from .loaders import load_dataset
from .rewards import reward_for_dataset
from .tasks import DATASET_SPECS, DatasetName, Method


@dataclass(frozen=True)
class TrainingRecipe:
    dataset: str
    method: str
    supported: bool
    reason: str


def training_recipe(dataset: str, method: str) -> TrainingRecipe:
    dataset_name = DatasetName(dataset)
    method_name = Method(method)
    spec = DATASET_SPECS[dataset_name]
    supported = method_name in spec.supported_methods
    if supported:
        return TrainingRecipe(dataset, method, True, spec.notes)
    if method_name is Method.SFT and dataset_name is DatasetName.IFEVAL:
        reason = "SFT unsupported: local IFEval file has prompts but no reference completions."
    elif method_name is Method.SFT and dataset_name is DatasetName.JAILBREAKBENCH:
        reason = "SFT unsupported: local JailbreakBench targets are harmful exemplars, not safe refusals."
    else:
        reason = f"{method} unsupported for {dataset}."
    return TrainingRecipe(dataset, method, False, reason)


def build_sft_examples(dataset: str, limit: int | None = None) -> list[dict[str, Any]]:
    recipe = training_recipe(dataset, Method.SFT.value)
    if not recipe.supported:
        raise ValueError(recipe.reason)
    rows = load_dataset(dataset)
    if limit is not None:
        rows = rows[:limit]
    examples = []
    for row in rows:
        prompt_messages = chat_example(row)
        examples.append(
            {
                "messages": prompt_messages,
                "completion": sft_target_text(row),
                "row_id": row.row_id,
            }
        )
    return examples


def build_grpo_examples(dataset: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_dataset(dataset)
    if limit is not None:
        rows = rows[:limit]
    return [
        {
            "messages": chat_example(row),
            "sample": row.payload,
            "row_id": row.row_id,
        }
        for row in rows
    ]


def grpo_reward_fn(dataset: str):
    def _reward_fn(completions, prompts=None, sample=None, samples=None, **_: Any):
        active_samples = samples if samples is not None else sample
        if active_samples is None:
            raise ValueError("GRPO reward function requires sample metadata.")
        rewards = []
        for completion, one_sample in zip(completions, active_samples, strict=False):
            text = completion
            if isinstance(completion, list) and completion:
                text = completion[0].get("content", "")
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            rewards.append(reward_for_dataset(dataset, str(text), one_sample))
        return rewards

    return _reward_fn


def build_trl_sft_trainer(*, model: str, dataset: str, output_dir: str, max_examples: int | None = None, **trainer_kwargs: Any):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    examples = build_sft_examples(dataset, limit=max_examples)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_dataset = Dataset.from_list(examples)
    training_args = SFTConfig(output_dir=output_dir, **trainer_kwargs)
    model_obj = AutoModelForCausalLM.from_pretrained(model)
    return SFTTrainer(
        model=model_obj,
        processing_class=tokenizer,
        train_dataset=hf_dataset,
        args=training_args,
    )


def build_trl_grpo_trainer(*, model: str, dataset: str, output_dir: str, max_examples: int | None = None, **trainer_kwargs: Any):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    examples = build_grpo_examples(dataset, limit=max_examples)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_dataset = Dataset.from_list(examples)
    training_args = GRPOConfig(output_dir=output_dir, **trainer_kwargs)
    model_obj = AutoModelForCausalLM.from_pretrained(model)
    return GRPOTrainer(
        model=model_obj,
        reward_funcs=grpo_reward_fn(dataset),
        args=training_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )
