from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Method(StrEnum):
    PROMPT = "prompt"
    SFT = "sft"
    BEST_OF_N = "best_of_n"
    GRPO = "grpo"


class DatasetName(StrEnum):
    BFCL = "bfcl"
    IFEVAL = "ifeval"
    JAILBREAKBENCH = "jailbreakbench"
    SAMSUM = "samsum"
    GSM8K = "gsm8k"


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    path: Path
    required_fields: tuple[str, ...]
    primary_input_field: str
    notes: str
    supported_methods: tuple[Method, ...]


@dataclass(frozen=True)
class MethodSpec:
    name: Method
    objective: str
    uses_updates: bool
    uses_reward: bool


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_OSS_ROOT = Path(os.environ.get("GRPO_OSS_BENCH_ROOT", "/home/nikhil/bench/OSS-datasets"))
LOCAL_DATA_ROOT = Path(os.environ.get("GRPO_OSS_DATA_ROOT", str(REPO_ROOT / "data")))


DATASET_SPECS: dict[DatasetName, DatasetSpec] = {
    DatasetName.BFCL: DatasetSpec(
        name=DatasetName.BFCL,
        path=BENCH_OSS_ROOT / "bfcl_3000.jsonl",
        required_fields=("function", "ground_truth"),
        primary_input_field="question",
        notes="Keep only rows with non-empty function schemas and ground-truth calls.",
        supported_methods=(
            Method.PROMPT,
            Method.SFT,
            Method.BEST_OF_N,
            Method.GRPO,
        ),
    ),
    DatasetName.IFEVAL: DatasetSpec(
        name=DatasetName.IFEVAL,
        path=BENCH_OSS_ROOT / "ifeval_samples.jsonl",
        required_fields=("prompt",),
        primary_input_field="prompt",
        notes="Keep only rows with non-empty prompts. SFT is unsupported from this local file because there are no reference completions.",
        supported_methods=(
            Method.PROMPT,
            Method.BEST_OF_N,
            Method.GRPO,
        ),
    ),
    DatasetName.JAILBREAKBENCH: DatasetSpec(
        name=DatasetName.JAILBREAKBENCH,
        path=BENCH_OSS_ROOT / "jailbreakbench_samples.jsonl",
        required_fields=("goal", "target"),
        primary_input_field="goal",
        notes="Keep only rows with non-empty attack goals and unsafe target strings. SFT is unsupported because the provided targets are harmful exemplars, not desired safe responses.",
        supported_methods=(
            Method.PROMPT,
            Method.BEST_OF_N,
            Method.GRPO,
        ),
    ),
    DatasetName.SAMSUM: DatasetSpec(
        name=DatasetName.SAMSUM,
        path=BENCH_OSS_ROOT / "samsum_3000.jsonl",
        required_fields=("dialogue", "summary"),
        primary_input_field="dialogue",
        notes="Keep only rows with non-empty dialogues and reference summaries.",
        supported_methods=(
            Method.PROMPT,
            Method.SFT,
            Method.BEST_OF_N,
            Method.GRPO,
        ),
    ),
    DatasetName.GSM8K: DatasetSpec(
        name=DatasetName.GSM8K,
        path=LOCAL_DATA_ROOT / "gsm8k_main.jsonl",
        required_fields=("question", "answer"),
        primary_input_field="question",
        notes="Keep only rows with non-empty questions and reference answers. Scoring uses strict extracted final-answer exact match.",
        supported_methods=(
            Method.PROMPT,
            Method.SFT,
            Method.BEST_OF_N,
            Method.GRPO,
        ),
    ),
}


METHOD_SPECS: dict[Method, MethodSpec] = {
    Method.PROMPT: MethodSpec(
        name=Method.PROMPT,
        objective="Prompt-only baseline with no parameter updates.",
        uses_updates=False,
        uses_reward=False,
    ),
    Method.SFT: MethodSpec(
        name=Method.SFT,
        objective="Supervised fine-tuning on filtered benchmark rows.",
        uses_updates=True,
        uses_reward=False,
    ),
    Method.BEST_OF_N: MethodSpec(
        name=Method.BEST_OF_N,
        objective="Inference-time sampling with verifier- or reward-based reranking.",
        uses_updates=False,
        uses_reward=True,
    ),
    Method.GRPO: MethodSpec(
        name=Method.GRPO,
        objective="Group Relative Policy Optimization with verifier- or reward-scored samples.",
        uses_updates=True,
        uses_reward=True,
    ),
}
