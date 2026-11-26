"""Independent attribute marginals, empirical priors, and hard evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite, log2
from types import MappingProxyType

from .schema import DiagramSchema


def entropy(probabilities: Iterable[float]) -> float:
    """Shannon entropy in bits, with 0 log 0 defined by continuity."""
    return -fsum(value * log2(value) for value in probabilities if value > 0)


@dataclass(frozen=True, slots=True)
class Belief:
    schema: DiagramSchema
    marginals: Mapping[str, tuple[float, ...]]
    resolved: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.marginals) != set(self.schema.keys):
            raise ValueError("Marginal keys must match the diagram schema")
        resolved = self.schema.validate(self.resolved)
        canonical = {}
        for attribute in self.schema.attributes:
            probabilities = tuple(float(value) for value in self.marginals[attribute.key])
            if (
                len(probabilities) != len(attribute.values)
                or any(not isfinite(value) or value < 0 for value in probabilities)
                or not isclose(fsum(probabilities), 1.0, abs_tol=1e-12)
            ):
                raise ValueError(f"Invalid probability distribution for {attribute.key}")
            if attribute.key in resolved:
                expected = tuple(
                    float(option == resolved[attribute.key]) for option in attribute.values
                )
                if probabilities != expected:
                    raise ValueError(f"Resolved attribute {attribute.key} needs a point mass")
            canonical[attribute.key] = probabilities
        object.__setattr__(self, "marginals", MappingProxyType(canonical))
        object.__setattr__(self, "resolved", MappingProxyType(resolved))

    @classmethod
    def uniform(cls, schema: DiagramSchema) -> Belief:
        marginals = {
            attr.key: (1.0 / len(attr.values),) * len(attr.values) for attr in schema.attributes
        }
        return cls(schema, marginals, {})

    @classmethod
    def empirical(
        cls, schema: DiagramSchema, specs: Iterable[Mapping[str, str]], *, smoothing: float = 0.0
    ) -> Belief:
        """Eq. 2 with optional additive smoothing for unseen domain values."""
        if not isfinite(smoothing) or smoothing < 0:
            raise ValueError("Smoothing must be finite and nonnegative")
        indices = {
            attr.key: {value: i for i, value in enumerate(attr.values)}
            for attr in schema.attributes
        }
        counts = {attr.key: [smoothing] * len(attr.values) for attr in schema.attributes}
        size = 0
        for spec in specs:
            for key, value in schema.validate(spec, complete=True).items():
                counts[key][indices[key][value]] += 1.0
            size += 1
        if not size:
            raise ValueError("An empirical prior requires at least one complete specification")
        return cls(
            schema,
            {
                key: tuple(value / fsum(values) for value in values)
                for key, values in counts.items()
            },
            {},
        )

    def marginal_entropy(self, key: str) -> float:
        return entropy(self.marginals[key])

    @property
    def uncertainty(self) -> float:
        return fsum(entropy(probs) for probs in self.marginals.values())

    @property
    def unresolved(self) -> tuple[str, ...]:
        return tuple(key for key in self.schema.keys if key not in self.resolved)

    def condition(self, evidence: Mapping[str, str]) -> Update:
        """Apply validated facts simultaneously. Previously resolved facts are idempotent."""
        facts = self.schema.validate(evidence)
        for key, value in facts.items():
            if key in self.resolved and self.resolved[key] != value:
                raise ValueError(f"Conflicting evidence for {key}")
        new_values = {key: value for key, value in facts.items() if key not in self.resolved}
        if not new_values:
            return Update(self, 0.0, ())
        marginals = dict(self.marginals)
        attribute_index = self.schema.index
        for key, value in new_values.items():
            marginals[key] = tuple(float(option == value) for option in attribute_index[key].values)
        posterior = Belief(self.schema, marginals, {**self.resolved, **new_values})
        gain = fsum(self.marginal_entropy(key) for key in new_values)
        return Update(posterior, gain, tuple(new_values))

    def total_possible_gain(self, keys: Sequence[str]) -> float:
        return fsum(self.marginal_entropy(key) for key in dict.fromkeys(keys))

    def infer(self, *, complete: bool = False) -> dict[str, str]:
        """Use confirmed values, optionally filling gaps with marginal MAP estimates."""
        if not complete:
            return dict(self.resolved)
        result = {}
        for attr in self.schema.attributes:
            probs = self.marginals[attr.key]
            result[attr.key] = attr.values[max(range(len(probs)), key=probs.__getitem__)]
        return result


@dataclass(frozen=True, slots=True)
class Update:
    posterior: Belief
    gain: float
    newly_resolved: tuple[str, ...]
