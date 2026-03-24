#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/nikhil/bench/.bench/bin/python"

mkdir -p "${ROOT}/results/logs"

cd "${ROOT}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli eval \
  --dataset bfcl \
  --method prompt \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/bfcl-prompt-test.json | tee results/logs/bfcl-prompt.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli eval \
  --dataset bfcl \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/bfcl-best_of_n-test.json | tee results/logs/bfcl-best_of_n.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli train \
  --method sft \
  --dataset bfcl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/bfcl-sft | tee results/logs/bfcl-sft-train.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli eval \
  --dataset bfcl \
  --method sft \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/bfcl-sft \
  --split test \
  --output-path results/bfcl-sft-test.json | tee results/logs/bfcl-sft-eval.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli train \
  --method grpo \
  --dataset bfcl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/bfcl-grpo | tee results/logs/bfcl-grpo-train.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli eval \
  --dataset bfcl \
  --method grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/bfcl-grpo \
  --split test \
  --output-path results/bfcl-grpo-test.json | tee results/logs/bfcl-grpo-eval.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli eval \
  --dataset ifeval \
  --method prompt \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/ifeval-prompt-test.json | tee results/logs/ifeval-prompt.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli eval \
  --dataset ifeval \
  --method best_of_n \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --split test \
  --output-path results/ifeval-best_of_n-test.json | tee results/logs/ifeval-best_of_n.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli train \
  --method grpo \
  --dataset ifeval \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir results/ifeval-grpo | tee results/logs/ifeval-grpo-train.log

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.cli eval \
  --dataset ifeval \
  --method grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-path results/ifeval-grpo \
  --split test \
  --output-path results/ifeval-grpo-test.json | tee results/logs/ifeval-grpo-eval.log
