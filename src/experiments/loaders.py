from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import DATASET_SPECS, DatasetName


@dataclass(frozen=True)
class DatasetRecord:
    dataset: str
    row_id: str
    prompt: Any
    payload: dict[str, Any]


@dataclass(frozen=True)
class DatasetReport:
    dataset: str
    path: str
    exists: bool
    total_rows: int
    kept_rows: int
    dropped_rows: int
    required_fields: tuple[str, ...]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _row_id(name: DatasetName, index: int, obj: dict[str, Any]) -> str:
    if name is DatasetName.IFEVAL:
        return f"{name.value}:{obj.get('key', index)}"
    if name is DatasetName.GSM8K:
        return f"{name.value}:{obj.get('id', index)}"
    return f"{name.value}:{index}"


def _normalize_prompt(name: DatasetName, obj: dict[str, Any], input_field: str) -> Any:
    raw = obj[input_field]
    if name is DatasetName.BFCL and isinstance(raw, list):
        return raw
    return raw


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "If you are adding GSM8K, prepare it first with scripts/prepare_gsm8k.py."
        )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def load_dataset(name: str | DatasetName) -> list[DatasetRecord]:
    dataset_name = DatasetName(name)
    spec = DATASET_SPECS[dataset_name]
    records: list[DatasetRecord] = []

    for index, obj in enumerate(_iter_jsonl(spec.path), start=1):
        if any(_is_empty(obj.get(field)) for field in spec.required_fields):
            continue
        records.append(
            DatasetRecord(
                dataset=dataset_name.value,
                row_id=_row_id(dataset_name, index, obj),
                prompt=_normalize_prompt(dataset_name, obj, spec.primary_input_field),
                payload=obj,
            )
        )
    return records


def dataset_report(name: str | DatasetName) -> DatasetReport:
    dataset_name = DatasetName(name)
    spec = DATASET_SPECS[dataset_name]
    if not spec.path.exists():
        return DatasetReport(
            dataset=dataset_name.value,
            path=str(spec.path),
            exists=False,
            total_rows=0,
            kept_rows=0,
            dropped_rows=0,
            required_fields=spec.required_fields,
        )
    total_rows = 0
    kept_rows = 0
    for obj in _iter_jsonl(spec.path):
        total_rows += 1
        if any(_is_empty(obj.get(field)) for field in spec.required_fields):
            continue
        kept_rows += 1
    return DatasetReport(
        dataset=dataset_name.value,
        path=str(spec.path),
        exists=True,
        total_rows=total_rows,
        kept_rows=kept_rows,
        dropped_rows=total_rows - kept_rows,
        required_fields=spec.required_fields,
    )
