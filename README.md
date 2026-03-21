# GRPO OSS Experiments

Small TRL-based experiment repo for comparing four methods on four open benchmark subsets:

- `prompt`
- `sft`
- `best_of_n`
- `grpo`

Benchmarks included:

- `bfcl`
- `ifeval`
- `jailbreakbench`
- `samsum`

## Overview

This repo currently provides:

- dataset adapters for four OSS benchmarks
- strict filtering of rows with missing or empty task-critical fields
- a generated experiment matrix for all `dataset x method` combinations
- TRL-facing training and evaluation entrypoints
- runnable CLI commands for training and evaluation with a Qwen3 4B default
- a small CLI to inspect dataset health and emit the matrix manifest

It does not include a full distributed training or serving stack. The goal is to keep the training and evaluation flow lightweight and easy to modify.

## Data

The current code expects local JSONL files from a companion `bench` workspace. The source paths are defined in [`src/experiments/tasks.py`](src/experiments/tasks.py). If your benchmark files live somewhere else, update those paths before running training or evaluation.

## Dataset filtering policy

Any row with missing or empty task-critical fields is dropped at load time.

- `bfcl`: requires non-empty `function`
- `ifeval`: requires non-empty `prompt`
- `jailbreakbench`: requires non-empty `goal` and `target`
- `samsum`: requires non-empty `dialogue` and `summary`

With the local files currently referenced by the repo, the usable counts are:

- `bfcl`: `240 / 3000` for rows with both non-empty `function` and non-empty `ground_truth`
- `ifeval`: `541 / 541`
- `jailbreakbench`: `200 / 200`
- `samsum`: `3000 / 3000`

## Quick Start

```bash
cd grpo-oss-experiments
pip install -e .
python -m experiments.cli inspect-datasets
python -m experiments.cli inspect-support
python -m experiments.cli show-defaults
python -m experiments.cli build-manifest --out configs/experiment_manifest.json
```

Train `SFT` on `SAMSum`:

```bash
cd grpo-oss-experiments
python -m experiments.cli train \
  --method sft \
  --dataset samsum \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/samsum-sft
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

Evaluate `best_of_n` on `BFCL`:

```bash
cd grpo-oss-experiments
python -m experiments.cli eval \
  --dataset bfcl \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/bfcl-best-of-n-test.json
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

- `bfcl`: supports `prompt`, `sft`, `best_of_n`, `grpo`
- `ifeval`: supports `prompt`, `best_of_n`, `grpo`
- `jailbreakbench`: supports `prompt`, `best_of_n`, `grpo`
- `samsum`: supports `prompt`, `sft`, `best_of_n`, `grpo`

Why `SFT` is limited:

- `IFEval` has prompts and constraint metadata but no reference completions.
- `JailbreakBench` has harmful target completions, which are not suitable as safe SFT targets.

## Runtime Entry Points

The main runtime entrypoints are in [`src/experiments/runtime.py`](src/experiments/runtime.py):

- `train_method(...)`
- `evaluate_method(...)`
- `build_sft_trainer(...)`
- `build_grpo_trainer(...)`

Lower-level helper functions also exist in [`src/experiments/trl_recipes.py`](src/experiments/trl_recipes.py):

- `build_trl_sft_trainer(...)`
- `build_trl_grpo_trainer(...)`

The GRPO reward functions are heuristic and dataset-specific:

- `BFCL`: exact or near-exact match against `ground_truth`
- `IFEval`: constraint checks for a supported subset of instruction types
- `JailbreakBench`: refusal-oriented heuristic reward
- `SAMSum`: overlap-style reward against the reference summary

These reward functions are enough to start experiments, but not enough for a publication-quality benchmark without stronger verifiers.

## Default Setup

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- adaptation: LoRA via `peft`
- dataset splits: deterministic hash split, `80/10/10`
- `SFT`: used only on datasets with reference targets (`BFCL`, `SAMSum`)
- `GRPO`: online policy optimization with dataset-specific reward functions

## Limitations

- The benchmark file locations are currently hard-coded in [`src/experiments/tasks.py`](src/experiments/tasks.py).
- `BFCL` is small under the current strict filter because many local rows are missing `ground_truth`.
- `IFEval`, `JailbreakBench`, and `SAMSum` use heuristic rewards rather than official benchmark evaluators.
- The repo has been smoke-tested for imports and data flow, but it has not yet been validated with a full end-to-end training run in this workspace.
