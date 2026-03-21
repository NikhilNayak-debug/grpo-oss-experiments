from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .loaders import dataset_report
from .manifest import build_experiment_manifest
from .runtime import evaluate_method, support_summary, train_method
from .tasks import DatasetName, Method
from .trl_recipes import training_recipe


def inspect_datasets_cmd() -> int:
    reports = [asdict(dataset_report(dataset)) for dataset in DatasetName]
    print(json.dumps({"datasets": reports}, indent=2))
    return 0


def inspect_support_cmd() -> int:
    rows = []
    for dataset in DatasetName:
        for method in Method:
            rows.append(asdict(training_recipe(dataset.value, method.value)))
    print(json.dumps({"support": rows}, indent=2))
    return 0


def build_manifest_cmd(out: str) -> int:
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_experiment_manifest(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))
    return 0


def train_cmd(method: str, dataset: str, output_dir: str, model_name_or_path: str) -> int:
    final_path = train_method(
        method,
        dataset=dataset,
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
    )
    print(final_path)
    return 0


def eval_cmd(
    dataset: str,
    method: str,
    base_model: str,
    adapter_path: str | None,
    split: str,
    limit: int | None,
    output_path: str | None,
) -> int:
    summary = evaluate_method(
        dataset=dataset,
        method=method,
        base_model=base_model,
        adapter_path=adapter_path,
        split=split,
        limit=limit,
        output_path=output_path,
    )
    print(json.dumps(summary, indent=2))
    return 0


def defaults_cmd() -> int:
    print(json.dumps(support_summary(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OSS benchmark experiment helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect-datasets", help="Show dataset health and usable counts.")
    subparsers.add_parser("inspect-support", help="Show which methods are supported by local data.")
    subparsers.add_parser("show-defaults", help="Show the default model and trainer settings.")

    build_manifest = subparsers.add_parser(
        "build-manifest",
        help="Emit the dataset x method experiment matrix as JSON.",
    )
    build_manifest.add_argument("--out", required=True, help="Output JSON path.")

    train_parser = subparsers.add_parser("train", help="Run SFT or GRPO training.")
    train_parser.add_argument("--method", choices=["sft", "grpo"], required=True)
    train_parser.add_argument("--dataset", choices=[dataset.value for dataset in DatasetName], required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")

    eval_parser = subparsers.add_parser("eval", help="Run prompt, best-of-n, SFT, or GRPO evaluation.")
    eval_parser.add_argument("--dataset", choices=[dataset.value for dataset in DatasetName], required=True)
    eval_parser.add_argument("--method", choices=[method.value for method in Method], required=True)
    eval_parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    eval_parser.add_argument("--adapter-path")
    eval_parser.add_argument("--split", default="test", choices=["train", "eval", "test"])
    eval_parser.add_argument("--limit", type=int)
    eval_parser.add_argument("--output-path")

    args = parser.parse_args()
    if args.command == "inspect-datasets":
        return inspect_datasets_cmd()
    if args.command == "inspect-support":
        return inspect_support_cmd()
    if args.command == "show-defaults":
        return defaults_cmd()
    if args.command == "train":
        return train_cmd(args.method, args.dataset, args.output_dir, args.model)
    if args.command == "eval":
        return eval_cmd(
            args.dataset,
            args.method,
            args.base_model,
            args.adapter_path,
            args.split,
            args.limit,
            args.output_path,
        )
    return build_manifest_cmd(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
