#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/nikhil/bench/.bench/bin/python"

mkdir -p "${ROOT}/results/logs"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" -m experiments.cli eval \
  --dataset gsm8k \
  --method prompt \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/gsm8k-prompt-test.json | tee results/logs/gsm8k-prompt.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" -m experiments.cli eval \
  --dataset gsm8k \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/gsm8k-best_of_n-test.json | tee results/logs/gsm8k-best_of_n.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" -m experiments.cli train \
  --method sft \
  --dataset gsm8k \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/gsm8k-sft | tee results/logs/gsm8k-sft-train.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" -m experiments.cli eval \
  --dataset gsm8k \
  --method sft \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/gsm8k-sft \
  --split test \
  --output-path results/gsm8k-sft-test.json | tee results/logs/gsm8k-sft-eval.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" -m experiments.cli train \
  --method grpo \
  --dataset gsm8k \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/gsm8k-grpo | tee results/logs/gsm8k-grpo-train.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" -m experiments.cli eval \
  --dataset gsm8k \
  --method grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/gsm8k-grpo \
  --split test \
  --output-path results/gsm8k-grpo-test.json | tee results/logs/gsm8k-grpo-eval.log
