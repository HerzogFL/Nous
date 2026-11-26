"""Evaluate information acquisition, specification fidelity, and paired differences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import fsum, isclose
from random import Random

from .belief import Belief
from .corpus import DiagramTask
from .dialogue import DialogueLimits, DialogueResult, InquiryPolicy, initial_context, simulate
from .oracle import PROFILES, SimulatedOracle, stable_seed


@dataclass(frozen=True, slots=True)
class CaseScore:
    task_id: str
    profile: str
    start: str
    turns: int
    stop_reason: str
    gain_bits: float
    normalized_gain: float
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    complete_accuracy: float
    exact_match: bool
    gain_curve: tuple[float, ...]


def score_dialogue(
    task: DiagramTask, result: DialogueResult, *, profile: str, start: str
) -> CaseScore:
    target = task.target
    evidence = result.final.belief.infer()
    complete = result.final.belief.infer(complete=True)
    hits = sum(target[key] == value for key, value in evidence.items())
    precision = hits / len(evidence) if evidence else 0.0
    recall = hits / len(target)
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    accuracy = sum(target[key] == value for key, value in complete.items()) / len(target)
    uncertainty = result.initial.belief.uncertainty
    gain = result.gain
    if not isclose(uncertainty - result.final.belief.uncertainty, gain, abs_tol=1e-9):
        raise ValueError("Turn rewards do not match the change in belief entropy")
    curve = [0.0]
    for turn in result.turns:
        curve.append(curve[-1] + turn.gain)
    return CaseScore(
        task.id,
        profile,
        start,
        len(result.turns),
        result.stop_reason,
        gain,
        min(1.0, gain / uncertainty) if uncertainty else 1.0,
        precision,
        recall,
        f1,
        accuracy,
        evidence == target,
        tuple(curve),
    )


def _mean(values) -> float:
    items = tuple(values)
    return fsum(items) / len(items)


def summarize(scores: Sequence[CaseScore], *, horizons: Sequence[int]) -> dict:
    if not scores:
        raise ValueError("Evaluation requires cases")
    return {
        "cases": len(scores),
        "mean_turns": _mean(score.turns for score in scores),
        "mean_gain_bits": _mean(score.gain_bits for score in scores),
        "mean_normalized_gain": _mean(score.normalized_gain for score in scores),
        "mean_evidence_precision": _mean(score.evidence_precision for score in scores),
        "mean_evidence_recall": _mean(score.evidence_recall for score in scores),
        "mean_evidence_f1": _mean(score.evidence_f1 for score in scores),
        "mean_complete_accuracy": _mean(score.complete_accuracy for score in scores),
        "exact_match_rate": _mean(score.exact_match for score in scores),
        "stop_reasons": dict(Counter(score.stop_reason for score in scores)),
        "gain_bits_at_turn": {
            str(turn): _mean(score.gain_curve[min(turn, score.turns)] for score in scores)
            for turn in horizons
        },
    }


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    profiles: tuple[str, ...] = ("template", "vague", "noisy", "novice", "expert")
    starts: tuple[str, ...] = ("zero", "draft")
    horizons: tuple[int, ...] = (1, 5, 10, 15)
    limits: DialogueLimits = DialogueLimits()
    seed: int = 42
    bootstrap_samples: int = 500

    def __post_init__(self) -> None:
        if not self.profiles or set(self.profiles).difference(PROFILES):
            raise ValueError("Choose one or more known profiles")
        if not self.starts or set(self.starts).difference(("zero", "draft")):
            raise ValueError("Choose zero and/or draft starts")
        if len(set(self.profiles)) != len(self.profiles) or len(set(self.starts)) != len(
            self.starts
        ):
            raise ValueError("Evaluation profiles and starts must be distinct")
        if any(turn < 0 for turn in self.horizons) or self.bootstrap_samples < 0:
            raise ValueError("Horizons and bootstrap samples must be nonnegative")


def _paired_gain(
    scores: Sequence[CaseScore], baseline: Sequence[CaseScore], *, samples: int, seed: int
) -> dict:
    """Resample tasks as clusters, keeping oracle profiles and starts paired."""
    by_task: dict[str, list[float]] = {}
    for left, right in zip(scores, baseline, strict=True):
        if (left.task_id, left.profile, left.start) != (right.task_id, right.profile, right.start):
            raise ValueError("Policies must be evaluated on matching cases")
        by_task.setdefault(left.task_id, []).append(left.gain_bits - right.gain_bits)
    deltas = [_mean(differences) for differences in by_task.values()]
    output = {"mean_gain_delta_bits": _mean(deltas), "ci95_bits": None}
    if samples:
        rng = Random(seed)
        means = sorted(_mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
        low, high = round(0.025 * (samples - 1)), round(0.975 * (samples - 1))
        output["ci95_bits"] = [means[low], means[high]]
    return output


def benchmark(
    tasks: Sequence[DiagramTask],
    prior: Belief,
    policies: Mapping[str, InquiryPolicy],
    config: BenchmarkConfig,
    *,
    baseline: str = "random",
    include_cases: bool = False,
) -> dict:
    if not tasks or baseline not in policies:
        raise ValueError("Evaluate at least one task and include the named baseline")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Evaluation task IDs must be unique")
    scores: dict[str, list[CaseScore]] = {name: [] for name in policies}
    for task in tasks:
        for profile in config.profiles:
            for start in config.starts:
                case_seed = stable_seed(config.seed, task.id, profile, start)
                initial = initial_context(prior, task, use_draft=start == "draft")
                oracle = SimulatedOracle(prior.schema, task.target, PROFILES[profile], case_seed)
                for name, policy in policies.items():
                    result = simulate(initial, oracle, policy, config.limits, seed=case_seed)
                    scores[name].append(score_dialogue(task, result, profile=profile, start=start))

    output = {"config": asdict(config), "baseline": baseline, "policies": {}}
    for name, cases in scores.items():
        entry = summarize(cases, horizons=config.horizons)
        entry["by_profile_and_start"] = {
            f"{profile}/{start}": summarize(
                [case for case in cases if case.profile == profile and case.start == start],
                horizons=config.horizons,
            )
            for profile in config.profiles
            for start in config.starts
        }
        if name != baseline:
            entry["paired_comparison"] = _paired_gain(
                cases, scores[baseline], samples=config.bootstrap_samples, seed=config.seed
            )
        if include_cases:
            entry["cases_detail"] = [asdict(case) for case in cases]
        output["policies"][name] = entry
    return output
