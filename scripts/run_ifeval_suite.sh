#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/nikhil/bench/.bench/bin/python"

mkdir -p "${ROOT}/results/logs"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" -m experiments.cli eval \
  --dataset ifeval \
  --method prompt \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/ifeval-prompt-test.json | tee results/logs/ifeval-prompt.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" -m experiments.cli eval \
  --dataset ifeval \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/ifeval-best_of_n-test.json | tee results/logs/ifeval-best_of_n.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" -m experiments.cli train \
  --method grpo \
  --dataset ifeval \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/ifeval-grpo | tee results/logs/ifeval-grpo-train.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" -m experiments.cli eval \
  --dataset ifeval \
  --method grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/ifeval-grpo \
  --split test \
  --output-path results/ifeval-grpo-test.json | tee results/logs/ifeval-grpo-eval.log
