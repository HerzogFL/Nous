"""One-turn belief updates, dialogue rollouts, and final brief synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from random import Random
from typing import Protocol

from .belief import Belief
from .corpus import DiagramTask
from .evidence import EvidenceParser, StructuredEvidenceParser
from .oracle import SimulatedOracle
from .questions import Context, Exchange, Question, candidate_questions


@dataclass(frozen=True, slots=True)
class DialogueLimits:
    max_turns: int = 15
    entropy_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("Turn limit must be positive")
        if not isfinite(self.entropy_threshold) or self.entropy_threshold < 0:
            raise ValueError("Entropy threshold must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class TurnResult:
    context: Context
    question: Question
    answer: str
    newly_resolved: tuple[str, ...]
    gain: float
    ignored: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DialogueResult:
    initial: Context
    final: Context
    turns: tuple[TurnResult, ...]
    stop_reason: str

    @property
    def gain(self) -> float:
        return fsum(turn.gain for turn in self.turns)


class InquiryPolicy(Protocol):
    def select(self, context: Context, candidates: Sequence[Question], rng: Random) -> Question: ...


class RandomPolicy:
    def select(self, context: Context, candidates: Sequence[Question], rng: Random) -> Question:
        return rng.choice(candidates)


class EntropyPolicy:
    """Greedy upper bound assuming the user answers every targeted attribute."""

    def select(self, context: Context, candidates: Sequence[Question], rng: Random) -> Question:
        return max(
            candidates,
            key=lambda question: (
                context.belief.total_possible_gain(question.targets),
                -len(question.targets),
                question.id,
            ),
        )


class SingleQuestionPolicy:
    """A one-attribute-at-a-time baseline that presents available choices."""

    def select(self, context: Context, candidates: Sequence[Question], rng: Random) -> Question:
        singles = [candidate for candidate in candidates if len(candidate.targets) == 1]
        return max(
            singles or candidates,
            key=lambda question: (
                context.belief.total_possible_gain(question.targets),
                question.style == "choice",
                question.id,
            ),
        )


def initial_context(prior: Belief, task: DiagramTask, *, use_draft: bool) -> Context:
    task.validate(prior.schema)
    belief = prior.condition(task.draft).posterior if use_draft else prior
    return Context(task.brief, belief)


def apply_answer(
    context: Context, question: Question, answer: str, parser: EvidenceParser
) -> TurnResult:
    known = context.belief.schema.index
    if any(key not in known for key in question.targets):
        raise ValueError(f"Question {question.id} contains unknown targets")
    evidence = parser.parse(answer, context.belief.schema, question.targets)
    facts = {}
    ignored = list(evidence.ignored)
    for key, value in evidence.facts.items():
        if key in context.belief.resolved and value != context.belief.resolved[key]:
            ignored.append(f"Compared with earlier evidence, {key} conflicts and needs review")
        else:
            facts[key] = value
    update = context.belief.condition(facts)
    next_context = Context(
        context.brief, update.posterior, (*context.history, Exchange(question, answer))
    )
    return TurnResult(
        next_context, question, answer, update.newly_resolved, update.gain, tuple(ignored)
    )


def stop_reason(context: Context, limits: DialogueLimits) -> str | None:
    if not context.belief.unresolved:
        return "resolved"
    if context.belief.uncertainty <= limits.entropy_threshold:
        return "entropy_threshold"
    if len(context.history) >= limits.max_turns:
        return "turn_limit"
    return None


def simulate(
    initial: Context,
    oracle: SimulatedOracle,
    policy: InquiryPolicy,
    limits: DialogueLimits,
    *,
    seed: int,
    parser: EvidenceParser | None = None,
) -> DialogueResult:
    evidence_parser = parser if parser is not None else StructuredEvidenceParser()
    rng = Random(seed)
    context = initial
    turns = []
    while (reason := stop_reason(context, limits)) is None:
        candidates = candidate_questions(context)
        if not candidates:
            reason = "no_questions"
            break
        question = policy.select(context, candidates, rng)
        if question not in candidates:
            raise ValueError("Policy returned a question outside the provided candidate set")
        answer = oracle.answer(question, turn=len(context.history))
        result = apply_answer(context, question, answer, evidence_parser)
        turns.append(result)
        context = result.context
    return DialogueResult(initial, context, tuple(turns), reason)


def render_brief(context: Context, *, include_estimates: bool = False) -> str:
    """Produce a grouped generation brief and explicitly mark unresolved fields."""
    values = context.belief.infer(complete=include_estimates)
    lines = [context.brief, ""]
    for section, attributes in context.belief.schema.sections().items():
        lines.append(section)
        for attr in attributes:
            if attr.key not in values:
                lines.append(f"- {attr.label}: unspecified")
                continue
            label = values[attr.key].replace("_", " ")
            if attr.key not in context.belief.resolved:
                label += " (prior estimate)"
            lines.append(f"- {attr.label}: {label}")
        lines.append("")
    return "\n".join(lines).rstrip()
