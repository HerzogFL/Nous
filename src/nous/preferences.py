"""Construct static preference groups from counterfactual questions per turn."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from random import Random
from typing import Any

from .belief import Belief
from .corpus import DiagramTask, read_jsonl
from .dialogue import DialogueLimits, apply_answer, initial_context, stop_reason
from .evidence import StructuredEvidenceParser
from .oracle import PROFILES, SimulatedOracle, stable_seed
from .questions import Context, Question, candidate_questions


@dataclass(frozen=True, slots=True)
class Candidate:
    question: Question
    answer: str
    gain: float
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not isfinite(self.gain) or self.gain < 0:
            raise ValueError("Candidate needs an answer and a finite nonnegative gain")
        if len(self.features) != len(FEATURE_NAMES) or not all(map(isfinite, self.features)):
            raise ValueError("Candidate feature vector has the wrong size or nonfinite values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "answer": self.answer,
            "gain": self.gain,
            "features": list(self.features),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Candidate:
        return cls(
            Question.from_dict(raw["question"]),
            raw["answer"],
            float(raw["gain"]),
            tuple(float(value) for value in raw["features"]),
        )


@dataclass(frozen=True, slots=True)
class PreferenceGroup:
    task_id: str
    turn: int
    profile: str
    prompt: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.task_id, self.profile, self.prompt)
        ):
            raise ValueError("Preference group needs a task id, profile, and prompt")
        if type(self.turn) is not int or self.turn < 0 or len(self.candidates) < 2:
            raise ValueError("Preference group needs a valid turn and two or more candidates")
        if len({item.question.id for item in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate IDs within a preference group must be distinct")

    def rewards(self, kind: str = "entropy") -> tuple[float, ...]:
        if kind == "entropy":
            return tuple(candidate.gain for candidate in self.candidates)
        if kind == "targets":
            return tuple(float(len(candidate.question.targets)) for candidate in self.candidates)
        raise ValueError(f"Unknown reward: {kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turn": self.turn,
            "profile": self.profile,
            "prompt": self.prompt,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreferenceGroup:
        return cls(
            raw["task_id"],
            raw["turn"],
            raw["profile"],
            raw["prompt"],
            tuple(Candidate.from_dict(candidate) for candidate in raw["candidates"]),
        )


FEATURE_NAMES = (
    "first_target_entropy",
    "additional_target_entropy",
    "target_count",
    "choice",
    "pair",
    "section",
    "max_marginal_entropy",
    "previous_target_fraction",
)


def question_features(context: Context, question: Question) -> tuple[float, ...]:
    """Use only features available before receiving the oracle answer."""
    entropies = [context.belief.marginal_entropy(key) / 2.0 for key in question.targets]
    previous = {key for exchange in context.history for key in exchange.question.targets}
    return (
        entropies[0],
        sum(entropies[1:]) / 2.0,
        len(entropies) / 4.0,
        float(question.style == "choice"),
        float(question.style == "pair"),
        float(question.style == "section"),
        max(entropies),
        sum(key in previous for key in question.targets) / len(question.targets),
    )


def balanced_sample(questions: Sequence[Question], size: int, rng: Random) -> tuple[Question, ...]:
    """Include different question styles before sampling the remaining pool."""
    if size < 2:
        raise ValueError("A preference group needs at least two candidates")
    if len(questions) <= size:
        return tuple(questions)

    by_style: dict[str, list[Question]] = {}
    for question in questions:
        by_style.setdefault(question.style, []).append(question)
    styles = list(by_style)
    rng.shuffle(styles)
    selected = [rng.choice(by_style[style]) for style in styles[:size]]
    seen = {question.id for question in selected}
    remainder = [question for question in questions if question.id not in seen]
    selected.extend(rng.sample(remainder, size - len(selected)))
    rng.shuffle(selected)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    group_size: int = 8
    profiles: tuple[str, ...] = ("template", "vague", "noisy")
    draft_fraction: float = 0.5
    exploration: float = 0.35
    limits: DialogueLimits = DialogueLimits()

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("Group size must be at least two")
        if not self.profiles or any(profile not in PROFILES for profile in self.profiles):
            raise ValueError("Choose at least one known oracle profile")
        if not 0 <= self.draft_fraction <= 1 or not 0 <= self.exploration <= 1:
            raise ValueError("Draft fraction and exploration must lie in [0, 1]")


def collect_groups(
    tasks: Iterable[DiagramTask], prior: Belief, config: CollectionConfig, *, seed: int
) -> Iterable[PreferenceGroup]:
    parser = StructuredEvidenceParser()
    for task in tasks:
        task_seed = stable_seed(seed, task.id)
        rng = Random(task_seed)
        profile = rng.choice(config.profiles)
        oracle = SimulatedOracle(prior.schema, task.target, PROFILES[profile], task_seed)
        context = initial_context(prior, task, use_draft=rng.random() < config.draft_fraction)
        while stop_reason(context, config.limits) is None:
            candidates = balanced_sample(candidate_questions(context), config.group_size, rng)
            if not candidates:
                break
            turn = len(context.history)
            alternatives = [
                apply_answer(context, question, oracle.answer(question, turn=turn), parser)
                for question in candidates
            ]
            if len(alternatives) >= 2:
                yield PreferenceGroup(
                    task.id,
                    turn,
                    profile,
                    context.prompt(),
                    tuple(
                        Candidate(
                            result.question,
                            result.answer,
                            result.gain,
                            question_features(context, result.question),
                        )
                        for result in alternatives
                    ),
                )
            result = (
                rng.choice(alternatives)
                if rng.random() < config.exploration
                else max(alternatives, key=lambda item: item.gain)
            )
            context = result.context


def load_groups(path: Path) -> tuple[PreferenceGroup, ...]:
    groups = []
    identities = set()
    for line_no, raw in read_jsonl(path):
        try:
            group = PreferenceGroup.from_dict(raw)
            identity = (group.task_id, group.turn, group.profile)
            if identity in identities:
                raise ValueError(f"Duplicate preference group: {identity}")
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}:{line_no}: {error}") from error
        groups.append(group)
        identities.add(identity)
    if not groups:
        raise ValueError(f"{path}: expected at least one preference group")
    return tuple(groups)
