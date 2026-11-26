"""Finite ordered domains for structured scientific-diagram specifications."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*\Z")
_VALUE = re.compile(r"[a-z0-9][a-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class Attribute:
    key: str
    label: str
    values: tuple[str, ...]
    section: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _KEY.fullmatch(self.key):
            raise ValueError(f"Invalid attribute key: {self.key!r}")
        if not all(isinstance(text, str) and text.strip() for text in (self.label, self.section)):
            raise ValueError(f"Expected a label and section for {self.key}")
        values = tuple(self.values)
        if not values or any(
            not isinstance(value, str) or not _VALUE.fullmatch(value) for value in values
        ):
            raise ValueError(f"Invalid domain for {self.key}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate values in domain for {self.key}")
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "values": list(self.values),
            "section": self.section,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Attribute:
        values = raw.get("values")
        if not isinstance(values, list):
            raise ValueError("Attribute values must be a JSON array")
        return cls(raw["key"], raw["label"], tuple(values), raw["section"])


@dataclass(frozen=True, slots=True)
class DiagramSchema:
    attributes: tuple[Attribute, ...]

    def __post_init__(self) -> None:
        attributes = tuple(self.attributes)
        if not attributes:
            raise ValueError("A diagram schema must have at least one attribute")
        if len({attribute.key for attribute in attributes}) != len(attributes):
            raise ValueError("Diagram schema contains duplicate keys")
        object.__setattr__(self, "attributes", attributes)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(attribute.key for attribute in self.attributes)

    @property
    def index(self) -> Mapping[str, Attribute]:
        return MappingProxyType({attribute.key: attribute for attribute in self.attributes})

    def validate(self, values: Mapping[str, str], *, complete: bool = False) -> dict[str, str]:
        if not isinstance(values, Mapping):
            raise ValueError("Expected an attribute-to-value object")
        unknown = set(values).difference(self.keys)
        if unknown:
            raise ValueError(f"Unknown attributes: {', '.join(sorted(map(str, unknown)))}")
        if complete:
            missing = set(self.keys).difference(values)
            if missing:
                raise ValueError(f"Missing attributes: {', '.join(sorted(missing))}")
        canonical = {}
        for attribute in self.attributes:
            if attribute.key in values:
                value = values[attribute.key]
                if value not in attribute.values:
                    raise ValueError(f"{value!r} is outside the domain for {attribute.key}")
                canonical[attribute.key] = value
        return canonical

    def sections(self) -> dict[str, tuple[Attribute, ...]]:
        sections: dict[str, list[Attribute]] = {}
        for attribute in self.attributes:
            sections.setdefault(attribute.section, []).append(attribute)
        return {section: tuple(attributes) for section, attributes in sections.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"attributes": [attribute.to_dict() for attribute in self.attributes]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DiagramSchema:
        return cls(tuple(Attribute.from_dict(item) for item in raw["attributes"]))


def _attributes(section: str, definitions: Iterable[tuple[str, str, str]]) -> Iterable[Attribute]:
    for key, label, values in definitions:
        yield Attribute(key, label, tuple(values.split()), section)


def scientific_diagram_schema() -> DiagramSchema:
    """A compact educational vocabulary spanning layout, style, content and relations."""
    attributes = []
    attributes.extend(
        _attributes(
            "Overview",
            (
                ("diagram.goal", "primary purpose", "explain compare summarize illustrate"),
                ("diagram.audience", "target audience", "students researchers general_public"),
                ("diagram.layout", "overall layout", "horizontal vertical radial grid"),
                ("diagram.panel_count", "number of panels", "one two three four"),
            ),
        )
    )
    attributes.extend(
        _attributes(
            "Style",
            (
                ("style.palette", "color palette", "monochrome muted saturated colorblind_safe"),
                ("style.background", "background", "white dark transparent"),
                ("style.labeling", "label density", "minimal standard detailed"),
                ("style.legend", "legend placement", "none bottom right integrated"),
            ),
        )
    )
    attributes.extend(
        _attributes(
            "Components",
            (
                ("components.count", "number of main components", "two three four five"),
                ("components.primary", "primary component shape", "box circle image plot"),
                ("components.secondary", "secondary component shape", "box circle image plot"),
                ("components.grouping", "component grouping", "none by_stage by_category"),
            ),
        )
    )
    attributes.extend(
        _attributes(
            "Relationships",
            (
                (
                    "relations.structure",
                    "relationship structure",
                    "sequence network hierarchy comparison",
                ),
                ("relations.arrows", "connector direction", "none one_way two_way mixed"),
                ("relations.labeling", "connector labels", "none short detailed"),
            ),
        )
    )
    return DiagramSchema(tuple(attributes))
