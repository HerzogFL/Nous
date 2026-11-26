from __future__ import annotations

import unittest
from dataclasses import replace
from math import log2
from random import Random

from nous.belief import Belief, entropy
from nous.corpus import DiagramTask, split_corpus, synthetic_corpus
from nous.dialogue import (
    DialogueLimits,
    EntropyPolicy,
    apply_answer,
    initial_context,
    render_brief,
    simulate,
)
from nous.evidence import StructuredEvidenceParser
from nous.oracle import PROFILES, SimulatedOracle
from nous.preferences import CollectionConfig, balanced_sample, collect_groups, question_features
from nous.questions import Context, Question, candidate_questions
from nous.schema import Attribute, DiagramSchema, scientific_diagram_schema


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.schema = scientific_diagram_schema()

    def test_roundtrip_and_sections(self):
        self.assertEqual(DiagramSchema.from_dict(self.schema.to_dict()), self.schema)
        self.assertEqual(
            [attribute.key for group in self.schema.sections().values() for attribute in group],
            list(self.schema.keys),
        )

    def test_rejects_invalid_domains_and_incomplete_targets(self):
        with self.assertRaisesRegex(ValueError, "Duplicate values"):
            Attribute("diagram.layout", "layout", ("left", "left"), "Overview")
        with self.assertRaisesRegex(ValueError, "outside the domain"):
            self.schema.validate({"diagram.layout": "curved"})
        with self.assertRaisesRegex(ValueError, "Unknown attributes"):
            self.schema.validate({"wrong.key": "horizontal"})
        with self.assertRaisesRegex(ValueError, "Missing attributes"):
            self.schema.validate({"diagram.layout": "horizontal"}, complete=True)

    def test_duplicate_schema_keys_are_rejected(self):
        attribute = self.schema.attributes[0]
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            DiagramSchema((attribute, attribute))


class TestBelief(unittest.TestCase):
    def setUp(self):
        self.schema = DiagramSchema(
            (
                Attribute("layout", "layout", ("left", "right"), "Diagram"),
                Attribute("palette", "colors", ("gray", "blue", "green"), "Diagram"),
            )
        )

    def test_empirical_prior_and_smoothing(self):
        rows = [
            {"layout": "left", "palette": "gray"},
            {"layout": "right", "palette": "gray"},
            {"layout": "right", "palette": "blue"},
        ]
        mle = Belief.empirical(self.schema, rows)
        self.assertEqual(mle.marginals["palette"], (2 / 3, 1 / 3, 0))
        self.assertAlmostEqual(mle.uncertainty, 2 * entropy((1 / 3, 2 / 3)))
        smoothed = Belief.empirical(self.schema, rows, smoothing=1)
        self.assertEqual(smoothed.marginals["palette"], (0.5, 1 / 3, 1 / 6))
        self.assertEqual(mle.resolved, {})

    def test_independent_hard_update_and_repeat(self):
        prior = Belief.uniform(self.schema)
        a = prior.condition({"palette": "blue"})
        b = a.posterior.condition({"palette": "blue", "layout": "right"})
        self.assertEqual(prior.marginals["layout"], a.posterior.marginals["layout"])
        self.assertEqual(a.posterior.marginals["palette"], (0.0, 1.0, 0.0))
        self.assertAlmostEqual(a.gain, log2(3))
        self.assertAlmostEqual(b.gain, 1.0)
        self.assertEqual(b.newly_resolved, ("layout",))
        self.assertEqual(b.posterior.uncertainty, 0.0)
        self.assertAlmostEqual(a.gain + b.gain, prior.uncertainty)
        self.assertEqual(prior.resolved, {})
        self.assertEqual(b.posterior.condition({"layout": "right"}).gain, 0)
        with self.assertRaisesRegex(ValueError, "Conflicting evidence"):
            b.posterior.condition({"layout": "left"})

    def test_invalid_prior_is_rejected_and_map_values_break_ties_by_domain_order(self):
        with self.assertRaisesRegex(ValueError, "Invalid probability"):
            Belief(self.schema, {"layout": (1.2, -0.2), "palette": (1.0, 0, 0)}, {})
        with self.assertRaisesRegex(ValueError, "at least one"):
            Belief.empirical(self.schema, [])
        self.assertEqual(
            Belief.uniform(self.schema).infer(complete=True), {"layout": "left", "palette": "gray"}
        )


class TestEvidenceDialogue(unittest.TestCase):
    def setUp(self):
        self.schema = scientific_diagram_schema()
        self.prior = Belief.uniform(self.schema)
        self.task = synthetic_corpus(self.schema, 1, seed=21)[0]
        self.context = initial_context(self.prior, self.task, use_draft=False)
        self.parser = StructuredEvidenceParser()
        self.question = Question("layout", "How should we lay it out?", ("diagram.layout",), "open")

    def test_parser_extracts_only_valid_values(self):
        result = self.parser.parse(
            "diagram.layout=grid; style.legend=not_an_option; Keep a copy please",
            self.schema,
            self.question.targets,
        )
        self.assertEqual(result.facts, {"diagram.layout": "grid"})
        self.assertEqual(len(result.ignored), 2)
        self.assertEqual(
            self.parser.parse("horizontal", self.schema, self.question.targets).facts,
            {"diagram.layout": "horizontal"},
        )

    def test_parser_handles_conflicts_and_duplicate_json_fields(self):
        for answer in (
            "diagram.layout=grid; diagram.layout=radial",
            '{"diagram.layout":"grid","diagram.layout":"radial"}',
            '{"diagram.layout":',
        ):
            with self.subTest(answer=answer):
                result = self.parser.parse(answer, self.schema, self.question.targets)
                self.assertEqual(result.facts, {})
                self.assertTrue(result.ignored)
        self.assertEqual(
            self.parser.parse('{"diagram.layout": "grid"}', self.schema, ()).facts,
            {"diagram.layout": "grid"},
        )

    def test_conflicting_answer_preserves_existing_fact(self):
        first = apply_answer(self.context, self.question, "grid", self.parser)
        next_turn = apply_answer(
            first.context,
            self.question,
            "diagram.layout=horizontal; style.palette=muted",
            self.parser,
        )
        self.assertEqual(
            next_turn.context.belief.resolved,
            {
                "diagram.layout": "grid",
                "style.palette": "muted",
            },
        )
        self.assertEqual(next_turn.newly_resolved, ("style.palette",))
        self.assertAlmostEqual(next_turn.gain, 2)
        self.assertIn("needs review", next_turn.ignored[0])

    def test_noise_and_draft(self):
        zero = initial_context(self.prior, self.task, use_draft=False)
        draft = initial_context(self.prior, self.task, use_draft=True)
        self.assertGreater(zero.belief.uncertainty, draft.belief.uncertainty)
        oracle = SimulatedOracle(self.schema, self.task.target, PROFILES["noisy"], 0)
        result = simulate(draft, oracle, EntropyPolicy(), DialogueLimits(), seed=0)
        self.assertEqual(result.final.belief.resolved, self.task.target)
        self.assertEqual(result.stop_reason, "resolved")
        self.assertAlmostEqual(result.gain, draft.belief.uncertainty)
        self.assertTrue(any(turn.ignored for turn in result.turns))
        self.assertIn("unspecified", render_brief(zero))
        self.assertIn("(prior estimate)", render_brief(zero, include_estimates=True))
        self.assertNotIn("unspecified", render_brief(result.final))

    def test_turn_limit_and_entropy_threshold(self):
        oracle = SimulatedOracle(self.schema, self.task.target, PROFILES["template"], 0)
        short = simulate(self.context, oracle, EntropyPolicy(), DialogueLimits(1), seed=0)
        self.assertEqual((short.stop_reason, len(short.turns)), ("turn_limit", 1))
        stopped = simulate(self.context, oracle, EntropyPolicy(), DialogueLimits(15, 50.0), seed=0)
        self.assertEqual((stopped.stop_reason, len(stopped.turns)), ("entropy_threshold", 0))

    def test_candidate_rotation_and_visible_features(self):
        candidates = candidate_questions(self.context)
        subset = balanced_sample(candidates, 4, Random(10))
        self.assertEqual(
            {question.style for question in subset}, {"open", "choice", "pair", "section"}
        )
        question = next(item for item in candidates if item.id == "open:diagram.layout")
        missed = apply_answer(self.context, question, "I am unsure", self.parser).context
        self.assertNotIn(question, candidate_questions(missed))
        choice = next(
            item for item in candidate_questions(missed) if item.id == "choice:diagram.layout"
        )
        self.assertEqual(question_features(missed, choice)[-1], 1.0)


class TestCollection(unittest.TestCase):
    def setUp(self):
        self.schema = scientific_diagram_schema()
        self.tasks = synthetic_corpus(self.schema, 12, seed=6)
        self.prior = Belief.uniform(self.schema)

    def test_reproducible_disjoint_split(self):
        split = split_corpus(self.tasks, seed=9)
        self.assertEqual(split, split_corpus(reversed(self.tasks), seed=9))
        self.assertEqual((len(split.train), len(split.test)), (9, 3))
        self.assertFalse({task.id for task in split.train} & {task.id for task in split.test})
        with self.assertRaisesRegex(ValueError, "unique"):
            split_corpus([self.tasks[0], self.tasks[0]], seed=9)

    def test_candidates_share_pre_answer_context_and_only_one_advances(self):
        task = self.tasks[0]
        config = CollectionConfig(
            group_size=8,
            profiles=("template",),
            exploration=0.0,
            draft_fraction=0.0,
            limits=DialogueLimits(3),
        )
        groups = list(collect_groups((task,), self.prior, config, seed=9))
        self.assertEqual(groups, list(collect_groups((task,), self.prior, config, seed=9)))
        self.assertEqual(len(groups), 3)
        first, second = groups[:2]
        self.assertEqual(first.prompt, Context(task.brief, self.prior).prompt())
        for alternative in first.candidates:
            self.assertAlmostEqual(
                alternative.gain, self.prior.total_possible_gain(alternative.question.targets)
            )
        winner = max(first.candidates, key=lambda candidate: candidate.gain)
        self.assertIn(f"User: {winner.answer}", second.prompt)
        self.assertEqual(second.prompt.count("Assistant:"), 1)
        self.assertEqual([group.turn for group in groups], [0, 1, 2])

    def test_task_requires_consistent_draft(self):
        task = self.tasks[0]
        altered = next(
            value
            for value in self.schema.index["diagram.layout"].values
            if value != task.target["diagram.layout"]
        )
        with self.assertRaisesRegex(ValueError, "must agree"):
            DiagramTask("impossible", "Make a diagram.", task.target, {"diagram.layout": altered})
        with self.assertRaisesRegex(ValueError, "Missing attributes"):
            replace(task, target={}, draft={}).validate(self.schema)


if __name__ == "__main__":
    unittest.main()
