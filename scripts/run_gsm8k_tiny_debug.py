from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from experiments.defaults import DEFAULT_MODEL, DEFAULT_SEED, EvalDefaults, GRPODefaults, LoraDefaults, SFTDefaults
from experiments.formatting import prompt_text, sft_target_text
from experiments.loaders import DatasetRecord, load_dataset
from experiments.rewards import evaluate_completion, trl_reward_fn
from experiments.runtime import _apply_chat_template, _bf16_enabled, _build_model, _build_tokenizer, _generate_texts, _load_adapter_if_needed
from experiments.tasks import DatasetName


DATASET = DatasetName.GSM8K.value
SUBSET_SIZE = 10
RESULTS_ROOT = Path("results/gsm8k-10shot")

# Debug-specific defaults so the trainable methods can actually move on 10 rows.
SFT_DEBUG = SFTDefaults(
    num_train_epochs=16.0,
    gradient_accumulation_steps=1,
    logging_steps=1,
    save_steps=10,
)
GRPO_DEBUG = GRPODefaults(
    num_train_epochs=8.0,
    learning_rate=5e-6,
    gradient_accumulation_steps=1,
    logging_steps=1,
    save_steps=10,
)


def _tiny_subset() -> list[DatasetRecord]:
    records = load_dataset(DATASET)
    if len(records) < SUBSET_SIZE:
        raise ValueError(f"Need at least {SUBSET_SIZE} GSM8K rows, found {len(records)}")
    return records[:SUBSET_SIZE]


def _prepare_sft_dataset(tokenizer, records: list[DatasetRecord]) -> Dataset:
    rows = []
    for record in records:
        rendered = _apply_chat_template(tokenizer, DATASET, prompt_text(record))
        rows.append(
            {
                "text": rendered + sft_target_text(record) + tokenizer.eos_token,
                "row_id": record.row_id,
            }
        )
    return Dataset.from_list(rows)


def _prepare_grpo_dataset(tokenizer, records: list[DatasetRecord]) -> Dataset:
    rows = []
    for record in records:
        rows.append(
            {
                "row_id": record.row_id,
                **record.payload,
                "prompt": _apply_chat_template(tokenizer, DATASET, prompt_text(record)),
            }
        )
    return Dataset.from_list(rows)


def _evaluate_records(
    *,
    method: str,
    records: list[DatasetRecord],
    base_model: str,
    adapter_path: str | None = None,
    output_path: Path,
    eval_defaults: EvalDefaults | None = None,
) -> dict[str, object]:
    defaults = eval_defaults or EvalDefaults()
    tokenizer = _build_tokenizer(base_model)
    model = _load_adapter_if_needed(base_model, adapter_path)
    prompts = [_apply_chat_template(tokenizer, DATASET, prompt_text(record)) for record in records]

    if method == "best_of_n":
        generated = _generate_texts(
            model,
            tokenizer,
            prompts,
            max_new_tokens=defaults.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=defaults.best_of_n,
        )
    else:
        generated = _generate_texts(
            model,
            tokenizer,
            prompts,
            max_new_tokens=defaults.max_new_tokens,
            temperature=defaults.temperature,
            top_p=defaults.top_p,
            num_return_sequences=1,
        )

    rows = []
    for record, candidates in zip(records, generated, strict=False):
        scored = []
        for candidate in candidates:
            reward, metadata = evaluate_completion(DATASET, candidate, record.payload)
            scored.append((candidate, reward, metadata))
        best_completion, best_reward, best_metadata = max(scored, key=lambda item: item[1])
        rows.append(
            {
                "row_id": record.row_id,
                "dataset": DATASET,
                "method": method,
                "reward": best_reward,
                "best_completion": best_completion,
                **best_metadata,
                "candidates": [
                    {"text": text, "reward": reward, **metadata}
                    for text, reward, metadata in scored
                ],
            }
        )

    summary = {
        "dataset": DATASET,
        "method": method,
        "split": "tiny_debug_same10",
        "base_model": base_model,
        "adapter_path": adapter_path,
        "num_examples": len(rows),
        "subset_row_ids": [record.row_id for record in records],
        "average_reward": (sum(row["reward"] for row in rows) / len(rows)) if rows else 0.0,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_sft(records: list[DatasetRecord], base_model: str) -> tuple[str, dict[str, object]]:
    tokenizer = _build_tokenizer(base_model)
    train_dataset = _prepare_sft_dataset(tokenizer, records)
    model = _build_model(base_model)
    config = SFTConfig(
        output_dir=str(RESULTS_ROOT / "sft"),
        num_train_epochs=SFT_DEBUG.num_train_epochs,
        learning_rate=SFT_DEBUG.learning_rate,
        per_device_train_batch_size=SFT_DEBUG.per_device_train_batch_size,
        gradient_accumulation_steps=SFT_DEBUG.gradient_accumulation_steps,
        logging_steps=SFT_DEBUG.logging_steps,
        save_steps=SFT_DEBUG.save_steps,
        eval_strategy="no",
        save_strategy="steps",
        max_length=SFT_DEBUG.max_seq_length,
        bf16=_bf16_enabled(),
        report_to=[],
        seed=DEFAULT_SEED,
    )
    peft_config = LoraConfig(
        r=LoraDefaults().r,
        lora_alpha=LoraDefaults().alpha,
        lora_dropout=LoraDefaults().dropout,
        target_modules=LoraDefaults().target_modules,
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=lambda sample: sample["text"],
    )
    trainer.train()
    output_dir = str(RESULTS_ROOT / "sft")
    trainer.save_model(output_dir)
    summary = _evaluate_records(
        method="sft",
        records=records,
        base_model=base_model,
        adapter_path=output_dir,
        output_path=RESULTS_ROOT / "gsm8k-10shot-sft-test.json",
    )
    return output_dir, summary


def _run_grpo(records: list[DatasetRecord], base_model: str) -> tuple[str, dict[str, object]]:
    tokenizer = _build_tokenizer(base_model)
    train_dataset = _prepare_grpo_dataset(tokenizer, records)
    model = _build_model(base_model)
    config = GRPOConfig(
        output_dir=str(RESULTS_ROOT / "grpo"),
        num_train_epochs=GRPO_DEBUG.num_train_epochs,
        learning_rate=GRPO_DEBUG.learning_rate,
        per_device_train_batch_size=GRPO_DEBUG.per_device_train_batch_size,
        gradient_accumulation_steps=GRPO_DEBUG.gradient_accumulation_steps,
        logging_steps=GRPO_DEBUG.logging_steps,
        save_steps=GRPO_DEBUG.save_steps,
        eval_strategy="no",
        save_strategy="steps",
        max_completion_length=GRPO_DEBUG.max_completion_length,
        num_generations=GRPO_DEBUG.num_generations,
        generation_batch_size=max(GRPO_DEBUG.per_device_train_batch_size, GRPO_DEBUG.num_generations),
        temperature=GRPO_DEBUG.temperature,
        top_p=GRPO_DEBUG.top_p,
        bf16=_bf16_enabled(),
        report_to=[],
        seed=DEFAULT_SEED,
    )
    peft_config = LoraConfig(
        r=LoraDefaults().r,
        lora_alpha=LoraDefaults().alpha,
        lora_dropout=LoraDefaults().dropout,
        target_modules=LoraDefaults().target_modules,
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=trl_reward_fn(DATASET),
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    output_dir = str(RESULTS_ROOT / "grpo")
    trainer.save_model(output_dir)
    summary = _evaluate_records(
        method="grpo",
        records=records,
        base_model=base_model,
        adapter_path=output_dir,
        output_path=RESULTS_ROOT / "gsm8k-10shot-grpo-test.json",
    )
    return output_dir, summary


def main() -> int:
    records = _tiny_subset()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    prompt_summary = _evaluate_records(
        method="prompt",
        records=records,
        base_model=DEFAULT_MODEL,
        output_path=RESULTS_ROOT / "gsm8k-10shot-prompt-test.json",
    )
    best_of_n_summary = _evaluate_records(
        method="best_of_n",
        records=records,
        base_model=DEFAULT_MODEL,
        output_path=RESULTS_ROOT / "gsm8k-10shot-best_of_n-test.json",
    )
    sft_adapter_path, sft_summary = _run_sft(records, DEFAULT_MODEL)
    grpo_adapter_path, grpo_summary = _run_grpo(records, DEFAULT_MODEL)

    final_summary = {
        "dataset": DATASET,
        "subset_source": "first_10_overall",
        "subset_size": len(records),
        "subset_row_ids": [record.row_id for record in records],
        "base_model": DEFAULT_MODEL,
        "methods": [
            {
                "method": "prompt",
                "average_reward": prompt_summary["average_reward"],
                "result_path": str(RESULTS_ROOT / "gsm8k-10shot-prompt-test.json"),
            },
            {
                "method": "best_of_n",
                "average_reward": best_of_n_summary["average_reward"],
                "result_path": str(RESULTS_ROOT / "gsm8k-10shot-best_of_n-test.json"),
            },
            {
                "method": "sft",
                "average_reward": sft_summary["average_reward"],
                "adapter_path": sft_adapter_path,
                "result_path": str(RESULTS_ROOT / "gsm8k-10shot-sft-test.json"),
            },
            {
                "method": "grpo",
                "average_reward": grpo_summary["average_reward"],
                "adapter_path": grpo_adapter_path,
                "result_path": str(RESULTS_ROOT / "gsm8k-10shot-grpo-test.json"),
            },
        ],
    }
    summary_path = RESULTS_ROOT / "gsm8k-10shot-summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
