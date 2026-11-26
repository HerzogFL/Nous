"""Group-relative objectives and derivatives for a categorical inquiry policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, fsum, isfinite, log, log1p, sqrt


@dataclass(frozen=True, slots=True)
class Objective:
    loss: float
    gradient: tuple[float, ...]
    kl: float
    clipped_fraction: float


def group_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Normalize rewards with the population standard deviation within one prompt."""
    if len(rewards) < 2 or not all(map(isfinite, rewards)):
        raise ValueError("A reward group needs at least two finite scores")
    average = fsum(rewards) / len(rewards)
    centered = tuple(reward - average for reward in rewards)
    scale = sqrt(fsum(value * value for value in centered) / len(centered))
    if scale <= 1e-12:
        return (0.0,) * len(rewards)
    return tuple(value / scale for value in centered)


def log_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits or not all(map(isfinite, logits)):
        raise ValueError("Expected a nonempty vector of finite logits")
    high = max(logits)
    shift = log(fsum(exp(value - high) for value in logits))
    return tuple(value - high - shift for value in logits)


def _softmax_pullback(
    direct_log_derivatives: Sequence[float], probabilities: Sequence[float]
) -> tuple[float, ...]:
    total = fsum(direct_log_derivatives)
    return tuple(
        derivative - probability * total
        for derivative, probability in zip(direct_log_derivatives, probabilities)
    )


def clipped_surrogate(
    ratio: float,
    advantage: float,
    epsilon: float,
) -> tuple[float, float, bool]:
    """Return the PPO term and its derivative with respect to log ratio."""
    low, high = 1.0 - epsilon, 1.0 + epsilon
    if advantage > 0 and ratio > high:
        return high * advantage, 0.0, True
    if advantage < 0 and ratio < low:
        return low * advantage, 0.0, True
    value = ratio * advantage
    return value, value, False


def offline_grpo(
    logits: Sequence[float],
    reference_logits: Sequence[float],
    rewards: Sequence[float],
    *,
    epsilon: float,
    beta: float,
) -> Objective:
    """Paper Eqs. 8–9 evaluated on the finite candidate distribution.

    The reference policy is frozen and the KL over the candidate set is exact.
    The gradient is with respect to current logits so the small CPU model can
    train without an autodiff dependency.
    """
    if not 0 < epsilon < 1 or not isfinite(beta) or beta < 0:
        raise ValueError("Expected 0 < epsilon < 1 and finite beta >= 0")
    if not len(logits) == len(reference_logits) == len(rewards):
        raise ValueError("Policy, reference, and reward vectors must be aligned")

    advantages = group_advantages(rewards)
    current = log_softmax(logits)
    reference = log_softmax(reference_logits)
    probabilities = tuple(exp(value) for value in current)
    log_ratios = tuple(left - right for left, right in zip(current, reference))
    size = len(logits)
    rewards_with_derivatives = [
        clipped_surrogate(exp(log_ratio), advantage, epsilon)
        for log_ratio, advantage in zip(log_ratios, advantages)
    ]
    surrogate = fsum(value for value, _, _ in rewards_with_derivatives) / size
    kl = fsum(p * ratio for p, ratio in zip(probabilities, log_ratios))
    direct = [
        -derivative / size + beta * p * (log_ratio + 1.0)
        for (_, derivative, _), p, log_ratio in zip(
            rewards_with_derivatives, probabilities, log_ratios
        )
    ]
    return Objective(
        -surrogate + beta * kl,
        _softmax_pullback(direct, probabilities),
        kl,
        sum(clipped for _, _, clipped in rewards_with_derivatives) / size,
    )


def supervised(logits: Sequence[float], rewards: Sequence[float]) -> Objective:
    """Maximum likelihood over the highest-scoring candidates, sharing ties."""
    log_probs = log_softmax(logits)
    if len(log_probs) != len(rewards) or not all(map(isfinite, rewards)):
        raise ValueError("Policy and reward vectors must be aligned and finite")
    best = max(rewards)
    winners = [value == best for value in rewards]
    count = sum(winners)
    if count == len(rewards):
        return Objective(0.0, (0.0,) * count, 0.0, 0.0)
    return Objective(
        -fsum(value for value, winner in zip(log_probs, winners) if winner) / count,
        tuple(exp(value) - winner / count for value, winner in zip(log_probs, winners)),
        0.0,
        0.0,
    )


def direct_preference(
    logits: Sequence[float],
    reference_logits: Sequence[float],
    rewards: Sequence[float],
    *,
    beta: float,
) -> Objective:
    """A best-vs-worst DPO baseline for the same fixed examples."""
    if (
        not isfinite(beta)
        or beta <= 0
        or not len(logits) == len(reference_logits) == len(rewards)
        or len(rewards) < 2
        or not all(map(isfinite, (*logits, *reference_logits, *rewards)))
    ):
        raise ValueError("DPO requires finite aligned groups and a positive scale")
    best = max(range(len(rewards)), key=rewards.__getitem__)
    worst = min(range(len(rewards)), key=rewards.__getitem__)
    gradient = [0.0] * len(rewards)
    if rewards[best] == rewards[worst]:
        return Objective(0.0, tuple(gradient), 0.0, 0.0)
    margin = beta * (
        logits[best] - logits[worst] - reference_logits[best] + reference_logits[worst]
    )
    tail = exp(-abs(margin))
    loss = max(-margin, 0.0) + log1p(tail)
    magnitude = beta * (tail if margin >= 0 else 1.0) / (1.0 + tail)
    gradient[best], gradient[worst] = -magnitude, magnitude
    return Objective(loss, tuple(gradient), 0.0, 0.0)
