"""Question candidates and historical context visible to the inquiry policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .belief import Belief


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    text: str
    targets: tuple[str, ...]
    style: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip() for value in (self.id, self.text, self.style)
        ):
            raise ValueError("A question needs an id, text, and style")
        if (
            not self.targets
            or len(set(self.targets)) != len(self.targets)
            or any(not isinstance(target, str) or not target for target in self.targets)
        ):
            raise ValueError("A question needs distinct nonempty target keys")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "targets": list(self.targets),
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Question:
        targets = record["targets"]
        if not isinstance(targets, list):
            raise ValueError("Question targets must be a JSON array")
        return cls(record["id"], record["text"], tuple(targets), record["style"])


@dataclass(frozen=True, slots=True)
class Exchange:
    question: Question
    answer: str


@dataclass(frozen=True, slots=True)
class Context:
    brief: str
    belief: Belief
    history: tuple[Exchange, ...] = ()

    @property
    def asked_ids(self) -> set[str]:
        return {turn.question.id for turn in self.history}

    def prompt(self) -> str:
        """Represent the same decision state for every response in a preference group."""
        lines = [
            "You help a user specify a scientific diagram through Socratic questioning.",
            "Ask one focused clarifying question about the unresolved specifications.",
            f"User request: {self.brief}",
            "",
            "Confirmed specifications:",
        ]
        lines.extend(
            [f"- {key}: {value}" for key, value in self.belief.resolved.items()] or ["- (none)"]
        )
        lines.append("Unresolved specifications:")
        attributes = self.belief.schema.index
        lines.extend(
            f"- {key}: {', '.join(attributes[key].values)}" for key in self.belief.unresolved
        )
        if self.history:
            lines.append("Dialogue:")
            for turn in self.history:
                lines.append(f"Assistant: {turn.question.text}")
                lines.append(f"User: {turn.answer}")
        lines.append("Next assistant question:")
        return "\n".join(lines) + "\n"


def candidate_questions(context: Context) -> tuple[Question, ...]:
    """Generate single-attribute, choice-based, and grouped template questions."""
    questions = []
    schema = context.belief.schema
    unresolved = set(context.belief.unresolved)
    for attr in schema.attributes:
        if attr.key not in unresolved:
            continue
        choices = ", ".join(value.replace("_", " ") for value in attr.values)
        questions.append(
            Question(
                f"open:{attr.key}",
                f"What should the {attr.label} be?",
                (attr.key,),
                "open",
            )
        )
        questions.append(
            Question(
                f"choice:{attr.key}",
                f"Which {attr.label} should we use: {choices}?",
                (attr.key,),
                "choice",
            )
        )
    for section, attributes in schema.sections().items():
        pending = [attr for attr in attributes if attr.key in unresolved]
        for first, second in zip(pending, pending[1:]):
            questions.append(
                Question(
                    f"pair:{first.key}+{second.key}",
                    f"How should we set the {first.label} and the {second.label}?",
                    (first.key, second.key),
                    "pair",
                )
            )
        if len(pending) > 2:
            keys = tuple(attr.key for attr in pending)
            labels = ", ".join(attr.label for attr in pending)
            questions.append(
                Question(
                    f"section:{section}:" + "+".join(keys),
                    f"Could you specify the {section.lower()} details: {labels}?",
                    keys,
                    "section",
                )
            )
    asked = context.asked_ids
    return tuple(question for question in questions if question.id not in asked)
