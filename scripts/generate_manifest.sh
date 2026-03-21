#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src"

python -m experiments.cli build-manifest --out "${ROOT}/configs/experiment_manifest.json"
