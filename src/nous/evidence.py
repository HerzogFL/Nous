"""Strict parsing for templated simulation and explicit interactive answers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .schema import DiagramSchema

_ASSIGNMENT = re.compile(r"\s*([a-z][a-z0-9_.]*)\s*=\s*([a-z0-9_]+)\s*\Z")


class EvidenceParser(Protocol):
    def parse(self, answer: str, schema: DiagramSchema, targets: Sequence[str]) -> Evidence: ...


@dataclass(frozen=True, slots=True)
class Evidence:
    facts: Mapping[str, str]
    ignored: tuple[str, ...] = ()


def _unique_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


class StructuredEvidenceParser:
    """Parse JSON objects or key=value clauses, ignoring prose in simulation replies.

    Values outside the schema are omitted with diagnostics. Ambiguous assignments
    for the same key are dropped as a unit. A single bare domain value is supported
    for interactive answers to one-target questions.
    """

    def parse(self, answer: str, schema: DiagramSchema, targets: Sequence[str]) -> Evidence:
        raw = answer.strip()
        if not raw:
            return Evidence({})
        if raw.startswith("{"):
            pairs, ignored = self._json_pairs(raw)
        else:
            pairs, ignored = self._assignment_pairs(raw, schema, targets)

        accepted: dict[str, str] = {}
        conflicting = set()
        known = schema.index
        for key, value in pairs:
            if key not in known or not isinstance(value, str) or value not in known[key].values:
                ignored.append(f"Unknown assignment: {key}={value!r}")
                continue
            if key in accepted and accepted[key] != value:
                conflicting.add(key)
            accepted[key] = value
        for key in sorted(conflicting):
            del accepted[key]
            ignored.append(f"Conflicting assignments for {key}")
        return Evidence(schema.validate(accepted), tuple(ignored))

    @staticmethod
    def _json_pairs(raw: str) -> tuple[list[tuple[str, object]], list[str]]:
        try:
            values = json.loads(raw, object_pairs_hook=_unique_json_fields)
        except ValueError as error:
            return [], [f"Malformed evidence: {error}"]
        return list(values.items()), []

    @staticmethod
    def _assignment_pairs(
        raw: str, schema: DiagramSchema, targets: Sequence[str]
    ) -> tuple[list[tuple[str, object]], list[str]]:
        attribute = schema.index.get(targets[0]) if len(targets) == 1 else None
        if attribute and raw in attribute.values:
            return [(attribute.key, raw)], []
        pairs = []
        ignored = []
        for clause in re.split(r"[;\n]+", raw):
            clean = clause.strip()
            match = _ASSIGNMENT.fullmatch(clean)
            if match:
                pairs.append((match[1], match[2]))
            elif clean:
                ignored.append(clean)
        return pairs, ignored


def format_evidence(values: Mapping[str, str]) -> str:
    return "; ".join(f"{key}={value}" for key, value in values.items())
