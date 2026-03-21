from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .defaults import SplitConfig
from .loaders import DatasetRecord


def _bucket_from_id(row_id: str) -> float:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_records(records: Iterable[DatasetRecord], config: SplitConfig | None = None) -> dict[str, list[DatasetRecord]]:
    split_config = config or SplitConfig()
    train_cutoff = split_config.train_ratio
    eval_cutoff = split_config.train_ratio + split_config.eval_ratio
    splits = {"train": [], "eval": [], "test": []}

    for record in records:
        bucket = _bucket_from_id(record.row_id)
        if bucket < train_cutoff:
            splits["train"].append(record)
        elif bucket < eval_cutoff:
            splits["eval"].append(record)
        else:
            splits["test"].append(record)
    return splits
