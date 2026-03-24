#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/nikhil/bench/.bench/bin/python"

mkdir -p "${ROOT}/results/logs"

cd "${ROOT}"

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli eval \
  --dataset jailbreakbench \
  --method prompt \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/jailbreakbench-prompt-test.json | tee results/logs/jailbreakbench-prompt.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli eval \
  --dataset jailbreakbench \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/jailbreakbench-best_of_n-test.json | tee results/logs/jailbreakbench-best_of_n.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli train \
  --method grpo \
  --dataset jailbreakbench \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/jailbreakbench-grpo | tee results/logs/jailbreakbench-grpo-train.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli eval \
  --dataset jailbreakbench \
  --method grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/jailbreakbench-grpo \
  --split test \
  --output-path results/jailbreakbench-grpo-test.json | tee results/logs/jailbreakbench-grpo-eval.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli eval \
  --dataset samsum \
  --method prompt \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/samsum-prompt-test.json | tee results/logs/samsum-prompt.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli eval \
  --dataset samsum \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/samsum-best_of_n-test.json | tee results/logs/samsum-best_of_n.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli train \
  --method sft \
  --dataset samsum \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/samsum-sft | tee results/logs/samsum-sft-train.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli eval \
  --dataset samsum \
  --method sft \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/samsum-sft \
  --split test \
  --output-path results/samsum-sft-test.json | tee results/logs/samsum-sft-eval.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli train \
  --method grpo \
  --dataset samsum \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/samsum-grpo | tee results/logs/samsum-grpo-train.log

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m experiments.cli eval \
  --dataset samsum \
  --method grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/samsum-grpo \
  --split test \
  --output-path results/samsum-grpo-test.json | tee results/logs/samsum-grpo-eval.log
