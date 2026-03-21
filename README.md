# GRPO OSS Experiments

Separate experiment repo for comparing four methods on three real-world benchmark subsets from `/home/nikhil/bench/OSS-datasets`:

- `prompt`
- `sft`
- `best_of_n`
- `grpo`

Benchmarks included:

- `bfcl`
- `ifeval`
- `jailbreakbench`
- `samsum`

## Current scope

This repo currently provides:

- dataset adapters for the three OSS benchmarks
- strict filtering of rows with missing or empty task-critical fields
- a generated experiment matrix for all `dataset x method` combinations
- TRL-facing recipe builders for `SFTTrainer` and `GRPOTrainer`
- runnable CLI commands for training and evaluation with a Qwen3 4B default
- a small CLI to inspect dataset health and emit the matrix manifest

It does not yet include a full distributed training or model serving stack. The current code is centered on TRL and leaves runtime orchestration lightweight on purpose.

## Dataset filtering policy

Any row with missing or empty task-critical fields is dropped at load time.

- `bfcl`: requires non-empty `function`
- `ifeval`: requires non-empty `prompt`
- `jailbreakbench`: requires non-empty `goal` and `target`
- `samsum`: requires non-empty `dialogue` and `summary`

With the local copies in `bench`, the current usable counts are:

- `bfcl`: `240 / 3000` for rows with both non-empty `function` and non-empty `ground_truth`
- `ifeval`: `541 / 541`
- `jailbreakbench`: `200 / 200`
- `samsum`: `3000 / 3000`

## Quick start

```bash
export PYTHONPATH=/tmp/grpo-oss-experiments/src
python -m experiments.cli inspect-datasets
python -m experiments.cli inspect-support
python -m experiments.cli show-defaults
python -m experiments.cli build-manifest --out /tmp/grpo-oss-experiments/configs/experiment_manifest.json
```

Train `SFT` on `SAMSum`:

```bash
python -m experiments.cli train \
  --method sft \
  --dataset samsum \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir /tmp/grpo-oss-experiments/results/samsum-sft
```

Train `GRPO` on `IFEval`:

```bash
python -m experiments.cli train \
  --method grpo \
  --dataset ifeval \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir /tmp/grpo-oss-experiments/results/ifeval-grpo
```

Evaluate `best_of_n` on `BFCL`:

```bash
python -m experiments.cli eval \
  --dataset bfcl \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path /tmp/grpo-oss-experiments/results/bfcl-best-of-n-test.json
```

## Method intent

- `prompt`: zero-shot or prompted baseline with no parameter updates
- `sft`: supervised fine-tuning baseline on each benchmark's train split
- `best_of_n`: sample `n` candidates at inference time and pick the best using the same verifier or reward used for RL
- `grpo`: policy optimization with group-relative ranking on verifier- or reward-scored outputs

## Recommended comparison rules

- use the same base model across all four methods
- use the same prompt template family where possible
- use the same verifier or reward logic for `best_of_n` and `grpo`
- keep held-out eval splits fixed per dataset
- report dataset-level metrics and macro averages separately

## Local-data support matrix

- `bfcl`: supports `prompt`, `sft`, `best_of_n`, `grpo`
- `ifeval`: supports `prompt`, `best_of_n`, `grpo`
- `jailbreakbench`: supports `prompt`, `best_of_n`, `grpo`
- `samsum`: supports `prompt`, `sft`, `best_of_n`, `grpo`

Why `SFT` is limited:

- `IFEval` has prompts and constraint metadata but no reference completions.
- `JailbreakBench` has harmful target completions, which are not suitable as safe SFT targets.

## TRL entrypoints

The repo exposes builder functions in [trl_recipes.py](/tmp/grpo-oss-experiments/src/experiments/trl_recipes.py):

- `build_trl_sft_trainer(...)`
- `build_trl_grpo_trainer(...)`

The GRPO reward functions are heuristic and dataset-specific:

- `BFCL`: exact or near-exact match against `ground_truth`
- `IFEval`: constraint checks for a supported subset of instruction types
- `JailbreakBench`: refusal-oriented heuristic reward
- `SAMSum`: overlap-style reward against the reference summary

These are enough to start experiments, but not enough for a publication-quality benchmark without refining the verifiers.

## Default runtime setup

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- adaptation: LoRA via `peft`
- dataset splits: deterministic hash split, `80/10/10`
- `SFT`: used only on datasets with reference targets (`BFCL`, `SAMSum`)
- `GRPO`: online policy optimization with dataset-specific reward functions
