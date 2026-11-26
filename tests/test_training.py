from __future__ import annotations

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nous.belief import Belief
from nous.cli import main
from nous.corpus import read_json, read_jsonl, synthetic_corpus, write_jsonl
from nous.dialogue import DialogueLimits, EntropyPolicy, RandomPolicy
from nous.evaluation import BenchmarkConfig, benchmark
from nous.objectives import direct_preference, group_advantages, offline_grpo, supervised
from nous.policy import LinearPolicy, TrainingConfig, policy_scores, train
from nous.preferences import CollectionConfig, collect_groups, load_groups
from nous.schema import scientific_diagram_schema


class TestLosses(unittest.TestCase):
    def test_reward_normalization_and_ties(self):
        self.assertEqual(group_advantages((4, 4)), (0, 0))
        advantages = group_advantages((3, 0, 1))
        self.assertAlmostEqual(sum(advantages), 0)
        self.assertAlmostEqual(sum(value**2 for value in advantages) / 3, 1)

    def test_grpo_gradient_including_kl_and_clipping(self):
        logits = (0.65, -0.65, 0.20)
        reference = (0.05, -0.03, 0.10)
        rewards = (2.0, 0.0, 1.0)

        def objective(parameters):
            return offline_grpo(parameters, reference, rewards, epsilon=0.2, beta=0.08)

        result = objective(logits)
        self.assertGreater(result.kl, 0.0)
        self.assertGreater(result.clipped_fraction, 0.0)
        self.assertAlmostEqual(sum(result.gradient), 0.0)
        for index, expected in enumerate(result.gradient):
            left, right = list(logits), list(logits)
            left[index] -= 1e-6
            right[index] += 1e-6
            derivative = (objective(right).loss - objective(left).loss) / 2e-6
            self.assertAlmostEqual(derivative, expected, places=7)

    def test_objectives_move_preference_in_expected_direction(self):
        logits, rewards = (0.0, 0.0), (0.0, 2.0)
        for result in (
            offline_grpo(logits, logits, rewards, epsilon=0.2, beta=0.0),
            direct_preference(logits, logits, rewards, beta=0.5),
            supervised(logits, rewards),
        ):
            self.assertGreater(result.gradient[0], 0)
            self.assertLess(result.gradient[1], 0)
        flat = offline_grpo(logits, logits, (2, 2), epsilon=0.2, beta=0.1)
        self.assertEqual(flat.loss, 0)
        self.assertEqual(flat.gradient, (0, 0))


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = scientific_diagram_schema()
        tasks = synthetic_corpus(schema, 14, seed=12)
        cls.groups = tuple(
            collect_groups(
                tasks,
                Belief.uniform(schema),
                CollectionConfig(profiles=("template",), limits=DialogueLimits(5)),
                seed=12,
            )
        )

    def test_all_training_methods_improve_static_group_scores(self):
        initial, _ = policy_scores(self.groups, LinearPolicy())
        for method in ("ofgrpo", "sft", "dpo"):
            with self.subTest(method=method):
                result = train(self.groups, TrainingConfig(method=method, epochs=6, batch_size=8))
                self.assertGreater(result.epochs[-1].expected_gain, initial)
                self.assertTrue(all(math.isfinite(epoch.loss) for epoch in result.epochs))

    def test_checkpoint_and_preferences_roundtrip(self):
        config = TrainingConfig(epochs=2)
        result = train(self.groups, config)
        self.assertEqual(result, train(self.groups, config))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "policy.json"
            result.policy.save(checkpoint)
            self.assertEqual(LinearPolicy.load(checkpoint), result.policy)
            records = root / "preferences.jsonl"
            write_jsonl(records, (group.to_dict() for group in self.groups))
            self.assertEqual(load_groups(records), self.groups)
            write_jsonl(records, [self.groups[0].to_dict()] * 2)
            with self.assertRaisesRegex(ValueError, r"preferences.jsonl:2: Duplicate"):
                load_groups(records)

    def test_failed_jsonl_write_leaves_prior_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl"
            write_jsonl(path, [{"a": 1}])
            with self.assertRaises(ValueError):
                write_jsonl(path, [{"changed": 1}, {"invalid": math.nan}])
            self.assertEqual(list(read_jsonl(path)), [(1, {"a": 1})])
            self.assertEqual(list(Path(folder).iterdir()), [path])


class TestEvaluationCLI(unittest.TestCase):
    def test_paired_benchmark_reproducible(self):
        schema = scientific_diagram_schema()
        tasks = synthetic_corpus(schema, 8, seed=17)
        config = BenchmarkConfig(
            profiles=("template", "noisy"), limits=DialogueLimits(4), bootstrap_samples=31
        )
        prior = Belief.uniform(schema)
        policies = {"random": RandomPolicy(), "entropy": EntropyPolicy()}
        result = benchmark(tasks, prior, policies, config, include_cases=True)
        self.assertEqual(result, benchmark(tasks, prior, policies, config, include_cases=True))
        greedy = result["policies"]["entropy"]
        self.assertGreater(greedy["paired_comparison"]["mean_gain_delta_bits"], 0.0)
        self.assertEqual(greedy["cases"], 32)
        self.assertEqual(len(greedy["by_profile_and_start"]), 4)
        self.assertEqual(len(greedy["cases_detail"]), greedy["cases"])

    def test_cli_pipeline_and_holdout(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            work = Path(folder) / "experiment"
            self.assertEqual(main(["prepare", str(work), "--cases", "12", "--max-turns", "4"]), 0)
            summary = read_json(work / "dataset.json")
            test_ids = {record["id"] for _, record in read_jsonl(work / "test.jsonl")}
            groups = load_groups(work / "preferences.jsonl")
            self.assertFalse(test_ids & {group.task_id for group in groups})
            self.assertEqual(summary["preference_groups"], len(groups))
            self.assertEqual(main(["train", str(work), "--epochs", "2"]), 0)
            checkpoint = work / "models" / "ofgrpo-entropy.json"
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        str(work),
                        "--policies",
                        str(checkpoint),
                        "--profiles",
                        "template",
                        "vague",
                        "--starts",
                        "draft",
                        "--include-cases",
                    ]
                ),
                0,
            )
            report = read_json(work / "evaluation.json")
            self.assertEqual(len(report["policies"]), 4)
            self.assertEqual(
                {case["task_id"] for case in report["policies"]["ofgrpo-entropy"]["cases_detail"]},
                test_ids,
            )

    def test_cli_interactive_and_prepare_protects_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            existing = work / "keep.txt"
            existing.write_text("original", encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                self.assertEqual(main(["prepare", str(work)]), 2)
            self.assertIn("empty directory", errors.getvalue())
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("builtins.input", side_effect=["/done"]):
            self.assertEqual(main(["ask", "A diagram of wind turbines"]), 0)
        self.assertIn("Final brief (user_finished)", output.getvalue())
        self.assertIn("A diagram of wind turbines", output.getvalue())


if __name__ == "__main__":
    unittest.main()
