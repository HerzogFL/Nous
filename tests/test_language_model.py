from __future__ import annotations

import copy
import importlib.util
import unittest
from types import SimpleNamespace

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch

    from nous.belief import Belief
    from nous.corpus import synthetic_corpus
    from nous.dialogue import DialogueLimits, initial_context
    from nous.lm import (
        LanguageModelPolicy,
        LMConfig,
        encode_group,
        language_model_loss,
        response_scores,
        train_language_model,
    )
    from nous.preferences import CollectionConfig, collect_groups
    from nous.questions import candidate_questions
    from nous.schema import scientific_diagram_schema

    class TinyTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, *, add_special_tokens, verbose):
            return [ord(ch) % 13 + 3 for ch in text]

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transitions = torch.nn.Embedding(16, 16)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.transitions(input_ids))


@unittest.skipUnless(_HAS_TORCH, "Install the lm extra to exercise autoregressive training")
class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.tokenizer = TinyTokenizer()

    def test_group_uses_identical_truncated_prompt_and_masks_padding(self):
        batch = encode_group(self.tokenizer, "long prefix", ["xy", "x"], max_length=6)
        self.assertEqual(batch.ids[0, :3].tolist(), batch.ids[1, :3].tolist())
        self.assertEqual(
            batch.response.tolist(),
            [
                [False, False, False, True, True, True],
                [False, False, False, True, True, False],
            ],
        )
        self.assertEqual(batch.attention[1, -1].item(), 0)
        self.assertEqual(batch.ids[1, 4].item(), self.tokenizer.eos_token_id)
        with self.assertRaisesRegex(ValueError, "full response"):
            encode_group(self.tokenizer, "a", ["too long"], max_length=3)

    def test_scores_match_manual_causal_response_logprob(self):
        model = TinyModel()
        batch = encode_group(self.tokenizer, "long prefix", ["xy", "x"], max_length=6)
        sequence_scores, token_distributions = response_scores(model, batch)
        log_probs = model.transitions.weight.log_softmax(dim=-1)
        expected = []
        for row in range(batch.ids.shape[0]):
            positions = batch.response[row].nonzero().flatten().tolist()
            expected.append(
                sum(
                    log_probs[batch.ids[row, position - 1], batch.ids[row, position]]
                    for position in positions
                )
            )
        torch.testing.assert_close(sequence_scores, torch.stack(expected))
        self.assertEqual(token_distributions.shape, (5, 16))

    def test_objective_masks_ties_and_does_not_update_reference(self):
        model = TinyModel()
        reference = copy.deepcopy(model)
        batch = encode_group(self.tokenizer, "prefix", ["x", "y"], max_length=10)
        tied, options = language_model_loss(model, reference, batch, (1.0, 1.0), LMConfig())
        self.assertAlmostEqual(tied.item(), 0.0)
        self.assertEqual(options["clipped_fraction"], 0.0)
        loss, _ = language_model_loss(model, reference, batch, (1.0, 2.0), LMConfig())
        loss.backward()
        self.assertGreater(model.transitions.weight.grad.norm().item(), 0.0)
        self.assertIsNone(reference.transitions.weight.grad)

    def test_training_step_and_ranking(self):
        schema = scientific_diagram_schema()
        tasks = synthetic_corpus(schema, 1, seed=11)
        prior = Belief.uniform(schema)
        groups = tuple(
            collect_groups(
                tasks,
                prior,
                CollectionConfig(
                    group_size=4,
                    profiles=("template",),
                    limits=DialogueLimits(1),
                ),
                seed=11,
            )
        )
        model = TinyModel()
        reference = copy.deepcopy(model)
        before = model.transitions.weight.detach().clone()
        report = train_language_model(
            model,
            reference,
            self.tokenizer,
            groups,
            LMConfig(epochs=2, learning_rate=0.01, max_length=128),
        )
        self.assertEqual(len(report), 2)
        self.assertFalse(torch.equal(before, model.transitions.weight))
        self.assertTrue(torch.equal(before, reference.transitions.weight))
        self.assertFalse(reference.transitions.weight.requires_grad)

        context = initial_context(prior, tasks[0], use_draft=False)
        candidates = candidate_questions(context)
        from random import Random

        ranker = LanguageModelPolicy(model, self.tokenizer, max_length=256, batch_size=3)
        selected = ranker.select(context, candidates, Random(3))
        self.assertIn(selected, candidates)
        with self.assertRaisesRegex(ValueError, "cannot share"):
            train_language_model(model, model, self.tokenizer, groups, LMConfig())


if __name__ == "__main__":
    unittest.main()
