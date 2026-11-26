"""Seeded answer simulation with controlled omissions and irrelevant noise."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from random import Random
from types import MappingProxyType

from .evidence import format_evidence
from .questions import Question
from .schema import DiagramSchema


@dataclass(frozen=True, slots=True)
class OracleProfile:
    name: str
    open_response: float
    choice_response: float
    extra_target_response: float
    noise_rate: float

    def __post_init__(self) -> None:
        for rate in (
            self.open_response,
            self.choice_response,
            self.extra_target_response,
            self.noise_rate,
        ):
            if not isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("Oracle rates must be finite and in [0, 1]")


PROFILES = MappingProxyType(
    {
        "template": OracleProfile("template", 1.0, 1.0, 1.0, 0.0),
        "vague": OracleProfile("vague", 0.67, 0.91, 0.38, 0.0),
        "noisy": OracleProfile("noisy", 1.0, 1.0, 1.0, 0.82),
        "novice": OracleProfile("novice", 0.51, 0.83, 0.27, 0.35),
        "expert": OracleProfile("expert", 0.96, 0.98, 0.87, 0.08),
    }
)

_NOISE = (
    "I may share the diagram in a presentation later",
    "The final deadline is still being discussed",
    "I have a spreadsheet that may be useful later",
    "Please keep the editable source handy",
)


def stable_seed(*parts: object) -> int:
    serialized = repr(parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(serialized, digest_size=8).digest(), "big")


class SimulatedOracle:
    def __init__(
        self,
        schema: DiagramSchema,
        target: Mapping[str, str],
        profile: OracleProfile,
        seed: int,
    ) -> None:
        self.target = MappingProxyType(schema.validate(target, complete=True))
        self.profile = profile
        self.seed = seed

    def answer(self, question: Question, *, turn: int) -> str:
        """Use a per-question RNG so candidate evaluation order cannot change labels."""
        rng = Random(stable_seed(self.seed, turn, question.id))
        first_rate = (
            self.profile.choice_response
            if question.style == "choice"
            else self.profile.open_response
        )
        found = {}
        for index, key in enumerate(question.targets):
            rate = first_rate if index == 0 else self.profile.extra_target_response
            if rng.random() < rate:
                found[key] = self.target[key]
        clauses = [format_evidence(found) or "I am unsure about that detail"]
        if rng.random() < self.profile.noise_rate:
            clauses.append(rng.choice(_NOISE))
        return "; ".join(clauses)
