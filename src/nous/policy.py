"""Small CPU inquiry policy and offline training on static preference groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import exp, fsum, isfinite, sqrt
from pathlib import Path
from random import Random

from .corpus import read_json, write_json
from .objectives import direct_preference, log_softmax, offline_grpo, supervised
from .preferences import FEATURE_NAMES, PreferenceGroup, question_features
from .questions import Context, Question

_METHODS = ("ofgrpo", "sft", "dpo")


@dataclass(frozen=True, slots=True)
class LinearPolicy:
    weights: tuple[float, ...] = (0.0,) * len(FEATURE_NAMES)
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES) or not all(map(isfinite, self.weights)):
            raise ValueError("Weights must be finite and match the policy feature schema")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Sampling temperature must be finite and nonnegative")

    def logits(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        return tuple(
            fsum(weight * value for weight, value in zip(self.weights, row, strict=True))
            for row in features
        )

    def select(self, context: Context, candidates: Sequence[Question], rng: Random) -> Question:
        logits = self.logits([question_features(context, question) for question in candidates])
        if self.temperature == 0:
            return candidates[max(range(len(logits)), key=logits.__getitem__)]
        high = max(logits)
        tempered = tuple((value - high) / self.temperature for value in logits)
        return rng.choices(candidates, weights=[exp(value) for value in tempered], k=1)[0]

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "version": 1,
                "features": list(FEATURE_NAMES),
                "weights": list(self.weights),
                "temperature": self.temperature,
            },
        )

    @classmethod
    def load(cls, path: Path) -> LinearPolicy:
        raw = read_json(path)
        if raw.get("version") != 1 or raw.get("features") != list(FEATURE_NAMES):
            raise ValueError(f"{path}: incompatible policy feature schema")
        return cls(tuple(raw["weights"]), raw.get("temperature", 0.0))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    method: str = "ofgrpo"
    reward: str = "entropy"
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 0.07
    epsilon: float = 0.2
    beta: float = 0.04
    dpo_beta: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.method not in _METHODS or self.reward not in ("entropy", "targets"):
            raise ValueError("Unknown training method or reward")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Epochs and batch size must be positive")
        positive = (self.learning_rate, self.dpo_beta, self.max_grad_norm)
        if any(not isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Learning rate, DPO scale, and gradient limit must be positive")
        if not 0 < self.epsilon < 1 or not isfinite(self.beta) or self.beta < 0:
            raise ValueError("Expected 0 < epsilon < 1 and a nonnegative KL scale")


@dataclass(frozen=True, slots=True)
class EpochStats:
    epoch: int
    loss: float
    kl: float
    clipped_fraction: float
    expected_gain: float
    top_choice_gain: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    policy: LinearPolicy
    epochs: tuple[EpochStats, ...]

    def report(self) -> dict:
        return {"epochs": [asdict(epoch) for epoch in self.epochs]}


def policy_scores(groups: Sequence[PreferenceGroup], policy: LinearPolicy) -> tuple[float, float]:
    expected = 0.0
    top = 0.0
    for group in groups:
        logits = policy.logits([candidate.features for candidate in group.candidates])
        gains = group.rewards()
        expected += fsum(exp(p) * gain for p, gain in zip(log_softmax(logits), gains))
        top += gains[max(range(len(logits)), key=logits.__getitem__)]
    return expected / len(groups), top / len(groups)


def train(
    groups: Sequence[PreferenceGroup],
    config: TrainingConfig,
    *,
    reference: LinearPolicy = LinearPolicy(),
    initial: LinearPolicy | None = None,
) -> TrainingResult:
    if not groups:
        raise ValueError("Training requires preference groups")
    current = initial if initial is not None else reference
    weights = list(current.weights)
    first_moment = [0.0] * len(weights)
    second_moment = [0.0] * len(weights)
    rng = Random(config.seed)
    order = list(range(len(groups)))
    cached = []
    for group in groups:
        features = tuple(candidate.features for candidate in group.candidates)
        cached.append((features, group.rewards(config.reward), reference.logits(features)))

    steps = 0
    metrics = []
    for epoch in range(1, config.epochs + 1):
        rng.shuffle(order)
        losses, divergences, clips = [], [], []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            gradient = [0.0] * len(weights)
            policy = LinearPolicy(tuple(weights))
            for index in batch:
                features, rewards, reference_logits = cached[index]
                logits = policy.logits(features)
                if config.method == "ofgrpo":
                    result = offline_grpo(
                        logits, reference_logits, rewards, epsilon=config.epsilon, beta=config.beta
                    )
                elif config.method == "dpo":
                    result = direct_preference(
                        logits, reference_logits, rewards, beta=config.dpo_beta
                    )
                else:
                    result = supervised(logits, rewards)
                losses.append(result.loss)
                divergences.append(result.kl)
                clips.append(result.clipped_fraction)
                for row, derivative in zip(features, result.gradient):
                    for coordinate, feature in enumerate(row):
                        gradient[coordinate] += derivative * feature
            steps += 1
            norm = sqrt(fsum(value * value for value in gradient)) / len(batch)
            scale = min(1.0, config.max_grad_norm / norm) if norm else 1.0
            for coordinate, value in enumerate(gradient):
                grad = value * scale / len(batch)
                first_moment[coordinate] = 0.9 * first_moment[coordinate] + 0.1 * grad
                second_moment[coordinate] = 0.999 * second_moment[coordinate] + 0.001 * grad**2
                corrected_mean = first_moment[coordinate] / (1 - 0.9**steps)
                corrected_variance = second_moment[coordinate] / (1 - 0.999**steps)
                weights[coordinate] -= (
                    config.learning_rate * corrected_mean / (sqrt(corrected_variance) + 1e-8)
                )
        current = LinearPolicy(tuple(weights), current.temperature)
        expected, top = policy_scores(groups, current)
        metrics.append(
            EpochStats(
                epoch,
                fsum(losses) / len(groups),
                fsum(divergences) / len(groups),
                fsum(clips) / len(groups),
                expected,
                top,
            )
        )
    return TrainingResult(current, tuple(metrics))
