# GRPO OSS Experiments

Small TRL-based experiment repo for comparing four methods on verifier-heavy benchmarks:

- `prompt`
- `sft`
- `best_of_n`
- `grpo`

Current focus:

- `ifeval`
- `gsm8k`

## Overview

This repo currently provides:

- dataset adapters for the current benchmark set plus prior OSS experiments
- strict filtering of rows with missing or empty task-critical fields
- a generated experiment matrix for all `dataset x method` combinations
- TRL-facing training and evaluation entrypoints
- runnable CLI commands for training and evaluation with a Qwen3 4B default
- a small CLI to inspect dataset health, emit the matrix manifest, and summarize completed runs

It does not include a full distributed training or serving stack. The goal is to keep the training and evaluation flow lightweight and easy to modify.

## Data

The current code expects:

- local OSS benchmark JSONL files from a companion `bench` workspace for legacy datasets
- a local `gsm8k_main.jsonl` file under `data/` for GSM8K

Dataset roots are configured in [`src/experiments/tasks.py`](src/experiments/tasks.py) and can be overridden with:

- `GRPO_OSS_BENCH_ROOT` for local OSS benchmark files
- `GRPO_OSS_DATA_ROOT` for repo-local prepared data such as `gsm8k_main.jsonl`

## Dataset filtering policy

Any row with missing or empty task-critical fields is dropped at load time.

- `ifeval`: requires non-empty `prompt`
- `gsm8k`: requires non-empty `question` and `answer`

With the local files currently referenced by the repo, the usable counts are:

- `ifeval`: `541 / 541`
- `gsm8k`: created locally after running the prep script

## Quick Start

```bash
cd grpo-oss-experiments
pip install -e .
python -m experiments.cli inspect-datasets
python -m experiments.cli inspect-support
python -m experiments.cli show-defaults
python -m experiments.cli build-manifest --out configs/experiment_manifest.json
```

Prepare `GSM8K`:

```bash
cd grpo-oss-experiments
python scripts/prepare_gsm8k.py
```

Train `GRPO` on `IFEval`:

```bash
cd grpo-oss-experiments
python -m experiments.cli train \
  --method grpo \
  --dataset ifeval \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/ifeval-grpo
```

Train `SFT` on `GSM8K`:

```bash
cd grpo-oss-experiments
python -m experiments.cli train \
  --method sft \
  --dataset gsm8k \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/gsm8k-sft
```

Evaluate `best_of_n` on `GSM8K`:

```bash
cd grpo-oss-experiments
python -m experiments.cli eval \
  --dataset gsm8k \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/gsm8k-best_of_n-test.json
```

Run the focused benchmark suites:

```bash
cd grpo-oss-experiments
bash scripts/run_ifeval_suite.sh
bash scripts/run_gsm8k_suite.sh
```

## Methods

- `prompt`: zero-shot or prompted baseline with no parameter updates
- `sft`: supervised fine-tuning baseline on each benchmark's train split
- `best_of_n`: sample `n` candidates at inference time and pick the best using the same verifier or reward used for RL
- `grpo`: policy optimization with group-relative ranking on verifier- or reward-scored outputs

## Comparison Notes

- use the same base model across all four methods
- use the same prompt template family where possible
- use the same verifier or reward logic for `best_of_n` and `grpo`
- keep held-out eval splits fixed per dataset
- report dataset-level metrics and macro averages separately

## Dataset Support

- `ifeval`: supports `prompt`, `best_of_n`, `grpo`
- `gsm8k`: supports `prompt`, `sft`, `best_of_n`, `grpo`

Why `SFT` is limited:

- `IFEval` has prompts and constraint metadata but no reference completions.

## Runtime Entry Points

The main runtime entrypoints are in [`src/experiments/runtime.py`](src/experiments/runtime.py):

- `train_method(...)`
- `evaluate_method(...)`
- `build_sft_trainer(...)`
- `build_grpo_trainer(...)`

Lower-level helper functions also exist in [`src/experiments/trl_recipes.py`](src/experiments/trl_recipes.py):

- `build_trl_sft_trainer(...)`
- `build_trl_grpo_trainer(...)`

The current reward functions are dataset-specific:

- `IFEval`: deterministic constraint checks over the 25 instruction types present in the local file, with per-example support coverage
- `GSM8K`: strict extracted final-answer exact match

These reward functions are enough to start experiments, but not enough for a publication-quality benchmark without stronger verifiers.

## Default Setup

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- adaptation: LoRA via `peft`
- dataset splits: deterministic hash split, `80/10/10`
- `SFT`: used only on datasets with reference targets (`GSM8K`)
- `GRPO`: online policy optimization with dataset-specific reward functions

## Limitations

- The current setup assumes local benchmark files already exist; `GSM8K` still requires a one-time preparation step.
- `IFEval` verification is deterministic and auditable, but it is still a repo-local implementation rather than an official benchmark evaluator.
- `GSM8K` scoring is strict final-answer match and does not separately reward reasoning quality.
- The repo has been smoke-tested for imports and data flow, but it has not yet been validated with a full end-to-end training run in this workspace.
