"""Synthetic diagram cases, deterministic held-out splits, and JSON artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any, TextIO

from .schema import DiagramSchema

JsonObject = dict[str, Any]


def _write_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, record: Any) -> None:
    def write(stream: TextIO) -> None:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    _write_atomic(path, write)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    def write(stream: TextIO) -> None:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")

    _write_atomic(path, write)


def read_json(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> Iterator[tuple[int, JsonObject]]:
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_no}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, value


@dataclass(frozen=True, slots=True)
class DiagramTask:
    id: str
    brief: str
    target: Mapping[str, str]
    draft: Mapping[str, str]

    def __post_init__(self) -> None:
        if not all(isinstance(text, str) and text.strip() for text in (self.id, self.brief)):
            raise ValueError("A task needs an id and a brief")
        target, draft = dict(self.target), dict(self.draft)
        if any(target.get(key) != value for key, value in draft.items()):
            raise ValueError("Draft evidence must agree with the target")
        object.__setattr__(self, "target", MappingProxyType(target))
        object.__setattr__(self, "draft", MappingProxyType(draft))

    def validate(self, schema: DiagramSchema) -> None:
        schema.validate(self.target, complete=True)
        schema.validate(self.draft)

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "brief": self.brief,
            "target": dict(self.target),
            "draft": dict(self.draft),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DiagramTask:
        return cls(raw["id"], raw["brief"], raw["target"], raw.get("draft", {}))


def load_tasks(path: Path, schema: DiagramSchema) -> tuple[DiagramTask, ...]:
    tasks = []
    seen = set()
    for line_no, raw in read_jsonl(path):
        try:
            task = DiagramTask.from_dict(raw)
            task.validate(schema)
            if task.id in seen:
                raise ValueError(f"Duplicate task id: {task.id}")
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}:{line_no}: {error}") from error
        tasks.append(task)
        seen.add(task.id)
    if not tasks:
        raise ValueError(f"{path}: expected at least one diagram task")
    return tuple(tasks)


@dataclass(frozen=True, slots=True)
class Scenario:
    topic: str
    preferred: Mapping[str, str]


_SCENARIOS = (
    Scenario(
        "the stages of the water cycle",
        {
            "diagram.goal": "explain",
            "diagram.layout": "radial",
            "components.grouping": "by_stage",
            "relations.structure": "sequence",
        },
    ),
    Scenario(
        "how a telescope collects and focuses light",
        {
            "diagram.goal": "illustrate",
            "diagram.layout": "horizontal",
            "components.primary": "image",
            "relations.arrows": "one_way",
        },
    ),
    Scenario(
        "a comparison of three approaches to storing solar energy",
        {
            "diagram.goal": "compare",
            "diagram.panel_count": "three",
            "diagram.layout": "grid",
            "relations.structure": "comparison",
        },
    ),
    Scenario(
        "the organization of an observatory's data catalog",
        {
            "diagram.audience": "researchers",
            "diagram.layout": "vertical",
            "components.grouping": "by_category",
            "relations.structure": "hierarchy",
        },
    ),
    Scenario(
        "how different wind speeds affect turbine output",
        {
            "diagram.goal": "compare",
            "components.primary": "plot",
            "style.legend": "right",
            "style.labeling": "detailed",
        },
    ),
    Scenario(
        "energy transfer across a simple food web",
        {
            "diagram.goal": "explain",
            "diagram.audience": "students",
            "relations.structure": "network",
            "relations.arrows": "mixed",
        },
    ),
)

_COMMON = {
    "style.background": "white",
    "style.labeling": "standard",
    "style.palette": "colorblind_safe",
    "style.legend": "bottom",
}


def synthetic_corpus(schema: DiagramSchema, size: int, *, seed: int) -> tuple[DiagramTask, ...]:
    """Build varied cases with shared biases and disjoint, reproducible task IDs."""
    if size < 1:
        raise ValueError("Corpus size must be positive")
    rng = Random(seed)
    cases = []
    for index in range(size):
        scenario = rng.choice(_SCENARIOS)
        preferred = {**_COMMON, **scenario.preferred}
        target = {}
        for attr in schema.attributes:
            favored = preferred.get(attr.key)
            if favored in attr.values and rng.random() < 0.65:
                target[attr.key] = favored
            else:
                target[attr.key] = rng.choice(attr.values)
        draft_keys = set(rng.sample(schema.keys, min(len(schema.keys), rng.randint(2, 5))))
        draft = {key: value for key, value in target.items() if key in draft_keys}
        cases.append(
            DiagramTask(
                f"synthetic-{seed}-{index:05d}",
                f"I need a scientific diagram about {scenario.topic}.",
                target,
                draft,
            )
        )
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    train: tuple[DiagramTask, ...]
    test: tuple[DiagramTask, ...]


def split_corpus(
    tasks: Iterable[DiagramTask], *, seed: int, test_fraction: float = 0.25
) -> CorpusSplit:
    ordered = sorted(tasks, key=lambda task: task.id)
    if len(ordered) < 2 or not 0 < test_fraction < 1:
        raise ValueError("Need at least two tasks and a test fraction in (0, 1)")
    if len({task.id for task in ordered}) != len(ordered):
        raise ValueError("Task IDs must be unique before splitting")
    Random(seed).shuffle(ordered)
    count = max(1, min(len(ordered) - 1, round(len(ordered) * test_fraction)))
    return CorpusSplit(tuple(ordered[count:]), tuple(ordered[:count]))
