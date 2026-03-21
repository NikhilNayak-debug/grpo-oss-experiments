from __future__ import annotations

from dataclasses import asdict, dataclass

from .loaders import dataset_report
from .tasks import DATASET_SPECS, METHOD_SPECS, DatasetName, Method


@dataclass(frozen=True)
class ExperimentEntry:
    experiment_id: str
    dataset: str
    method: str
    supported: bool
    source_path: str
    total_rows: int
    usable_rows: int
    dropped_rows: int
    required_fields: tuple[str, ...]
    notes: str
    objective: str
    uses_updates: bool
    uses_reward: bool


def build_experiment_manifest() -> dict[str, list[dict[str, object]]]:
    entries: list[dict[str, object]] = []
    for dataset_name in DatasetName:
        report = dataset_report(dataset_name)
        spec = DATASET_SPECS[dataset_name]
        for method in Method:
            method_spec = METHOD_SPECS[method]
            entry = ExperimentEntry(
                experiment_id=f"{dataset_name.value}__{method.value}",
                dataset=dataset_name.value,
                method=method.value,
                supported=method in spec.supported_methods,
                source_path=report.path,
                total_rows=report.total_rows,
                usable_rows=report.kept_rows,
                dropped_rows=report.dropped_rows,
                required_fields=report.required_fields,
                notes=spec.notes,
                objective=method_spec.objective,
                uses_updates=method_spec.uses_updates,
                uses_reward=method_spec.uses_reward,
            )
            entries.append(asdict(entry))
    return {"experiments": entries}
