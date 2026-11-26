"""Command line entry points for preparing, training, evaluating, and asking."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from random import Random

from .belief import Belief
from .corpus import load_tasks, read_json, split_corpus, synthetic_corpus, write_json, write_jsonl
from .dialogue import (
    DialogueLimits,
    EntropyPolicy,
    RandomPolicy,
    SingleQuestionPolicy,
    apply_answer,
    render_brief,
    stop_reason,
)
from .evaluation import BenchmarkConfig, benchmark
from .evidence import StructuredEvidenceParser
from .oracle import PROFILES
from .policy import LinearPolicy, TrainingConfig, train
from .preferences import CollectionConfig, collect_groups, load_groups
from .questions import Context, candidate_questions
from .schema import DiagramSchema, scientific_diagram_schema


def _load_prior(work: Path) -> Belief:
    schema = DiagramSchema.from_dict(read_json(work / "schema.json"))
    configuration = read_json(work / "dataset.json")
    training = load_tasks(work / "train.jsonl", schema)
    return Belief.empirical(
        schema, (task.target for task in training), smoothing=configuration["smoothing"]
    )


def _limits(args: argparse.Namespace) -> DialogueLimits:
    return DialogueLimits(args.max_turns, args.entropy_threshold)


def _prepare(args: argparse.Namespace) -> None:
    work = args.work_dir
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise ValueError(f"Choose an empty directory for the dataset: {work}")
    schema = (
        DiagramSchema.from_dict(read_json(args.schema))
        if args.schema
        else scientific_diagram_schema()
    )
    tasks = (
        load_tasks(args.tasks, schema)
        if args.tasks
        else synthetic_corpus(schema, args.cases, seed=args.seed)
    )
    split = split_corpus(tasks, seed=args.seed, test_fraction=args.test_fraction)
    prior = Belief.empirical(
        schema, (task.target for task in split.train), smoothing=args.smoothing
    )
    config = CollectionConfig(
        group_size=args.group_size,
        profiles=tuple(args.profiles),
        draft_fraction=args.draft_fraction,
        exploration=args.exploration,
        limits=_limits(args),
    )
    groups_path = work / "preferences.jsonl"
    group_count = 0

    def records():
        nonlocal group_count
        for group in collect_groups(split.train, prior, config, seed=args.seed):
            group_count += 1
            yield group.to_dict()
        if group_count == 0:
            raise ValueError("No preference groups were generated. Check the schema and limits.")

    write_jsonl(groups_path, records())
    write_json(work / "schema.json", schema.to_dict())
    write_jsonl(work / "train.jsonl", (task.to_dict() for task in split.train))
    write_jsonl(work / "test.jsonl", (task.to_dict() for task in split.test))
    metadata = {
        "seed": args.seed,
        "corpus": "custom" if args.tasks else "synthetic",
        "training_tasks": len(split.train),
        "test_tasks": len(split.test),
        "preference_groups": group_count,
        "smoothing": args.smoothing,
        "test_fraction": args.test_fraction,
        "collection": {
            "group_size": config.group_size,
            "profiles": list(config.profiles),
            "draft_fraction": config.draft_fraction,
            "exploration": config.exploration,
            "max_turns": config.limits.max_turns,
            "entropy_threshold": config.limits.entropy_threshold,
        },
    }
    write_json(work / "dataset.json", metadata)
    print(
        f"Prepared {group_count} groups from {len(split.train)} training cases "
        f"and held out {len(split.test)} cases in {work}."
    )


def _train(args: argparse.Namespace) -> None:
    groups = load_groups(args.work_dir / "preferences.jsonl")
    config = TrainingConfig(
        method=args.method,
        reward=args.reward,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        epsilon=args.epsilon,
        dpo_beta=args.dpo_beta,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    output = args.output or (args.work_dir / "models" / f"{args.method}-{args.reward}.json")
    result = train(groups, config)
    result.policy.save(output)
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "config": asdict(config),
            **result.report(),
        },
    )
    print(f"Saved {config.method} policy to {output}.")
    for epoch in result.epochs:
        print(
            f"  epoch {epoch.epoch:2d} loss={epoch.loss:.5f} "
            f"KL={epoch.kl:.5f} top-choice gain={epoch.top_choice_gain:.3f} bits"
        )


def _train_lm(args: argparse.Namespace) -> None:
    from .lm import LMConfig, train_checkpoint

    groups = load_groups(args.work_dir / "preferences.jsonl")
    if args.max_groups is not None:
        if args.max_groups < 1:
            raise ValueError("Maximum groups must be positive")
        groups = tuple(Random(args.seed).sample(groups, min(args.max_groups, len(groups))))
    config = LMConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        beta=args.beta,
        max_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    report = train_checkpoint(args.model, args.output, groups, config, device=args.device)
    print(f"Saved causal language model to {args.output}.")
    print(json.dumps(report, indent=2))


def _eval(args: argparse.Namespace) -> None:
    prior = _load_prior(args.work_dir)
    test = load_tasks(args.work_dir / "test.jsonl", prior.schema)
    train_tasks = load_tasks(args.work_dir / "train.jsonl", prior.schema)
    if {task.id for task in test}.intersection(task.id for task in train_tasks):
        raise ValueError("Training and evaluation task IDs overlap")

    policies = {
        "random": RandomPolicy(),
        "entropy": EntropyPolicy(),
        "single": SingleQuestionPolicy(),
    }
    for model in args.policies:
        name = model.stem
        if name in policies:
            raise ValueError(f"Duplicate policy name: {name}")
        policies[name] = LinearPolicy.load(model)
    if args.lm_model:
        from .lm import LanguageModelPolicy

        if "lm" in policies:
            raise ValueError("The policy name lm is reserved when --lm-model is supplied")
        policies["lm"] = LanguageModelPolicy.load(
            args.lm_model, device=args.device, max_length=args.lm_max_length
        )
    config = BenchmarkConfig(
        profiles=tuple(args.profiles),
        starts=tuple(args.starts),
        limits=_limits(args),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    report = benchmark(test, prior, policies, config, include_cases=args.include_cases)
    write_json(args.output or args.work_dir / "evaluation.json", report)
    print(f"{'Policy':24} {'Turns':>8} {'Gain (bits)':>12} {'Evidence F1':>12}")
    for name, metrics in report["policies"].items():
        print(
            f"{name:24} {metrics['mean_turns']:8.2f} "
            f"{metrics['mean_gain_bits']:12.3f} {metrics['mean_evidence_f1']:12.3f}"
        )


def _ask(args: argparse.Namespace) -> None:
    prior = (
        _load_prior(args.work_dir) if args.work_dir else Belief.uniform(scientific_diagram_schema())
    )
    policy = LinearPolicy.load(args.policy) if args.policy else SingleQuestionPolicy()
    limits = _limits(args)
    parser = StructuredEvidenceParser()
    rng = Random(args.seed)
    context = Context(args.brief, prior)
    print("Answer with a listed value or key=value assignments. Enter /done to finish.")
    while (reason := stop_reason(context, limits)) is None:
        choices = candidate_questions(context)
        if not choices:
            reason = "no_questions"
            break
        question = policy.select(context, choices, rng)
        print(f"\n{question.text}")
        for key in question.targets:
            print(f"  {key}: {', '.join(prior.schema.index[key].values)}")
        try:
            answer = input("> ").strip()
        except EOFError:
            answer = "/done"
        if answer == "/done":
            reason = "user_finished"
            break
        outcome = apply_answer(context, question, answer, parser)
        for ignored in outcome.ignored:
            print(f"  Unused answer fragment: {ignored}")
        print(f"  Information gain: {outcome.gain:.3f} bits")
        context = outcome.context
    print(f"\nFinal brief ({reason})\n")
    print(render_brief(context, include_estimates=args.include_estimates))


def _add_limits(parser: argparse.ArgumentParser, *, turns: int = 15) -> None:
    parser.add_argument("--max-turns", type=int, default=turns)
    parser.add_argument("--entropy-threshold", type=float, default=0.0)


def _add_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nous", description="Run information-gain-driven clarification experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build train/test tasks and preference groups")
    prepare.add_argument("work_dir", type=Path)
    prepare.add_argument("--tasks", type=Path, help="Use a custom JSONL diagram corpus")
    prepare.add_argument("--schema", type=Path, help="Use a custom JSON diagram schema")
    prepare.add_argument("--cases", type=int, default=160)
    prepare.add_argument("--test-fraction", type=float, default=0.25)
    prepare.add_argument("--smoothing", type=float, default=0.1)
    prepare.add_argument("--group-size", type=int, default=8)
    prepare.add_argument("--draft-fraction", type=float, default=0.5)
    prepare.add_argument("--exploration", type=float, default=0.35)
    prepare.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=["template", "vague", "noisy"]
    )
    _add_limits(prepare, turns=10)
    _add_seed(prepare)
    prepare.set_defaults(handler=_prepare)

    fit = commands.add_parser("train", help="Train a CPU inquiry policy on fixed groups")
    fit.add_argument("work_dir", type=Path)
    fit.add_argument("--output", type=Path)
    fit.add_argument("--method", choices=["ofgrpo", "sft", "dpo"], default="ofgrpo")
    fit.add_argument("--reward", choices=["entropy", "targets"], default="entropy")
    fit.add_argument("--epochs", type=int, default=8)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--learning-rate", type=float, default=0.07)
    fit.add_argument("--epsilon", type=float, default=0.2)
    fit.add_argument("--beta", type=float, default=0.04)
    fit.add_argument("--dpo-beta", type=float, default=0.5)
    fit.add_argument("--max-grad-norm", type=float, default=1.0)
    _add_seed(fit)
    fit.set_defaults(handler=_train)

    fit_lm = commands.add_parser("train-lm", help="Run response-masked offline GRPO on a causal LM")
    fit_lm.add_argument("work_dir", type=Path)
    fit_lm.add_argument("--model", required=True)
    fit_lm.add_argument("--output", type=Path, required=True)
    fit_lm.add_argument("--device", default="auto")
    fit_lm.add_argument("--epochs", type=int, default=1)
    fit_lm.add_argument("--learning-rate", type=float, default=1e-6)
    fit_lm.add_argument("--epsilon", type=float, default=0.2)
    fit_lm.add_argument("--beta", type=float, default=0.04)
    fit_lm.add_argument("--max-length", type=int, default=1024)
    fit_lm.add_argument("--max-grad-norm", type=float, default=1.0)
    fit_lm.add_argument("--max-groups", type=int)
    _add_seed(fit_lm)
    fit_lm.set_defaults(handler=_train_lm)

    evaluate = commands.add_parser("evaluate", help="Compare inquiry policies on held-out cases")
    evaluate.add_argument("work_dir", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--policies", type=Path, nargs="+", default=[])
    evaluate.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    evaluate.add_argument(
        "--starts", nargs="+", choices=["zero", "draft"], default=["zero", "draft"]
    )
    evaluate.add_argument("--bootstrap-samples", type=int, default=500)
    evaluate.add_argument("--include-cases", action="store_true")
    evaluate.add_argument("--lm-model")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--lm-max-length", type=int, default=1024)
    _add_limits(evaluate)
    _add_seed(evaluate)
    evaluate.set_defaults(handler=_eval)

    ask = commands.add_parser("ask", help="Run a local interactive clarification session")
    ask.add_argument("brief")
    ask.add_argument("--work-dir", type=Path)
    ask.add_argument("--policy", type=Path)
    ask.add_argument("--include-estimates", action="store_true")
    _add_limits(ask)
    _add_seed(ask)
    ask.set_defaults(handler=_ask)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except ModuleNotFoundError as error:
        if error.name in {"torch", "transformers"}:
            print(
                "Install optional model dependencies with pip install -e '.[lm]'.", file=sys.stderr
            )
            return 2
        raise
    except (ValueError, KeyError, TypeError, OSError, FloatingPointError) as error:
        print(f"nous: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
