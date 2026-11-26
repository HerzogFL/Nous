"""Response-masked offline GRPO for autoregressive language models.

This module is optional. The rest of the package uses only the standard library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from math import fsum, isfinite
from pathlib import Path
from random import Random

import torch
from torch import Tensor

from .corpus import write_json
from .objectives import group_advantages
from .preferences import PreferenceGroup
from .questions import Context, Question


@dataclass(frozen=True, slots=True)
class TokenBatch:
    ids: Tensor
    attention: Tensor
    response: Tensor

    def to(self, device: torch.device) -> TokenBatch:
        return TokenBatch(self.ids.to(device), self.attention.to(device), self.response.to(device))


def encode_group(
    tokenizer, prompt: str, responses: Sequence[str], *, max_length: int
) -> TokenBatch:
    """Reserve a common prompt prefix and include an EOS token in every response."""
    if not responses or max_length < 3:
        raise ValueError("Expected responses and a maximum length of at least three")
    eos = tokenizer.eos_token_id
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    if eos is None:
        raise ValueError("The tokenizer must define an EOS token")
    if pad is None:
        pad = eos

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, verbose=False)
    prompt_ids = ([bos] if bos is not None else []) + encoded_prompt
    if not prompt_ids:
        prompt_ids = [eos]
    continuations = []
    for response in responses:
        ids = tokenizer.encode(response, add_special_tokens=False, verbose=False)
        if not ids:
            raise ValueError("A response must contain at least one token")
        if ids[-1] != eos:
            ids.append(eos)
        if len(ids) >= max_length:
            raise ValueError("A full response and at least one prompt token must fit")
        continuations.append(ids)
    room = max_length - max(map(len, continuations))
    if len(prompt_ids) > room:
        # Preserve a short instruction prefix and prioritize the latest exchanges.
        prefix = room // 3
        prompt_ids = prompt_ids[:prefix] + prompt_ids[-(room - prefix) :]
    width = len(prompt_ids) + max(map(len, continuations))

    tokens, attention, mask = [], [], []
    for continuation in continuations:
        leading, trailing = len(prompt_ids), len(continuation)
        padding = width - leading - trailing
        tokens.append(prompt_ids + continuation + [pad] * padding)
        attention.append([1] * (leading + trailing) + [0] * padding)
        mask.append([False] * leading + [True] * trailing + [False] * padding)
    return TokenBatch(
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )


def response_scores(model, batch: TokenBatch) -> tuple[Tensor, Tensor]:
    """Return summed continuation log probabilities and distributions at scored positions."""
    logits = model(input_ids=batch.ids, attention_mask=batch.attention).logits
    response_mask = batch.response[:, 1:] & batch.attention[:, 1:].bool()
    if not bool(response_mask.any(dim=1).all()):
        raise ValueError("Each sequence must have a response token with a causal predecessor")

    log_probs = logits[:, :-1, :][response_mask].float().log_softmax(dim=-1)
    selected_ids = batch.ids[:, 1:][response_mask]
    token_scores = log_probs.gather(-1, selected_ids.unsqueeze(-1)).squeeze(-1)
    row_ids = response_mask.nonzero(as_tuple=True)[0]
    sequence_scores = token_scores.new_zeros(batch.ids.shape[0]).index_add(0, row_ids, token_scores)
    return sequence_scores, log_probs


@dataclass(frozen=True, slots=True)
class LMConfig:
    epochs: int = 1
    learning_rate: float = 1e-6
    epsilon: float = 0.2
    beta: float = 0.04
    max_length: int = 1024
    max_grad_norm: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.max_length < 3:
            raise ValueError("Epochs must be positive and max length at least three")
        if not 0 < self.epsilon < 1 or not isfinite(self.beta) or self.beta < 0:
            raise ValueError("Expected 0 < epsilon < 1 and a nonnegative KL scale")
        if any(not isfinite(v) or v <= 0 for v in (self.learning_rate, self.max_grad_norm)):
            raise ValueError("Learning rate and gradient limit must be finite and positive")


def language_model_loss(
    model, reference, batch: TokenBatch, rewards: Sequence[float], config: LMConfig
) -> tuple[Tensor, dict[str, float]]:
    """Clipped response ratios with a conditional KL penalty on observed prefixes."""
    if len(rewards) != batch.ids.shape[0]:
        raise ValueError("Reward and response counts must match")
    scores, current_tokens = response_scores(model, batch)
    with torch.no_grad():
        ref_scores, ref_tokens = response_scores(reference, batch)

    advantages = scores.new_tensor(group_advantages(rewards))
    log_ratio = scores - ref_scores
    # Positive advantages saturate at 1+epsilon. Evaluate the selected
    # branch in log space so large positive inputs never reach exp().
    clipped_log_ratio = log_ratio.clamp(
        min=scores.new_tensor(1.0 - config.epsilon).log(),
        max=scores.new_tensor(1.0 + config.epsilon).log(),
    )
    effective = torch.where(
        advantages > 0,
        torch.minimum(log_ratio, clipped_log_ratio),
        torch.maximum(log_ratio, clipped_log_ratio),
    )
    # Guard the remaining adverse branch against numerical overflow. Tied
    # rewards contribute zero even when the raw sequence ratio is large.
    effective = torch.where(advantages == 0, 0.0, effective)
    surrogate = (effective.clamp(max=30).exp() * advantages).mean()
    kl = (current_tokens.exp() * (current_tokens - ref_tokens)).sum(-1).mean()
    loss = -surrogate + config.beta * kl
    with torch.no_grad():
        clipped = ((advantages != 0) & (effective != log_ratio)).float().mean()
        metrics = {
            "loss": float(loss),
            "surrogate": float(surrogate),
            "kl": max(0.0, float(kl)),
            "clipped_fraction": float(clipped),
        }
    return loss, metrics


def train_language_model(
    model, reference, tokenizer, groups: Sequence[PreferenceGroup], config: LMConfig
) -> list[dict[str, float]]:
    if not groups:
        raise ValueError("Training requires preference groups")
    if model is reference or {id(p) for p in model.parameters()}.intersection(
        id(p) for p in reference.parameters()
    ):
        raise ValueError("The frozen reference cannot share parameters with the policy")
    device = next(model.parameters()).device
    reference.to(device)
    reference.requires_grad_(False).eval()
    # Both policies use deterministic forward passes when scoring the same
    # responses. Evaluation mode disables dropout while preserving gradients.
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    torch.manual_seed(config.seed)
    rng = Random(config.seed)
    order = list(groups)
    report = []
    for epoch in range(1, config.epochs + 1):
        rng.shuffle(order)
        statistics = []
        for group in order:
            batch = encode_group(
                tokenizer,
                group.prompt,
                [item.question.text for item in group.candidates],
                max_length=config.max_length,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = language_model_loss(model, reference, batch, group.rewards(), config)
            group_name = f"{group.task_id}/{group.turn}"
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite LM loss for {group_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            statistics.append(metrics)
        report.append(
            {
                "epoch": epoch,
                **{
                    key: fsum(item[key] for item in statistics) / len(statistics)
                    for key in statistics[0]
                },
            }
        )
    return report


def auto_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _context_length(requested: int, model, tokenizer) -> int:
    declared = (
        getattr(model.config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    )
    limits = [limit for limit in declared if isinstance(limit, int) and limit > 0]
    return min([requested, *limits])


def train_checkpoint(
    model_name: str,
    output: Path,
    groups: Sequence[PreferenceGroup],
    config: LMConfig,
    *,
    device: str,
) -> list[dict[str, float]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose an empty directory for the model checkpoint: {output}")
    target = auto_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy = AutoModelForCausalLM.from_pretrained(model_name).to(target)
    reference = AutoModelForCausalLM.from_pretrained(model_name)
    effective = replace(config, max_length=_context_length(config.max_length, policy, tokenizer))
    report = train_language_model(policy, reference, tokenizer, groups, effective)
    policy.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(
        output / "nous_training.json",
        {
            "base_model": model_name,
            "config": asdict(effective),
            "epochs": report,
        },
    )
    return report


class LanguageModelPolicy:
    """Rank supplied clarifying questions by average response log likelihood."""

    def __init__(self, model, tokenizer, *, max_length: int, batch_size: int = 8):
        if batch_size < 1:
            raise ValueError("Ranking batch size must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    @classmethod
    def load(cls, name: str, *, device: str, max_length: int) -> LanguageModelPolicy:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(name).to(auto_device(device))
        tokenizer = AutoTokenizer.from_pretrained(name)
        return cls(model, tokenizer, max_length=_context_length(max_length, model, tokenizer))

    @torch.inference_mode()
    def select(self, context: Context, candidates: Sequence[Question], rng: Random) -> Question:
        batch = encode_group(
            self.tokenizer,
            context.prompt(),
            [candidate.text for candidate in candidates],
            max_length=self.max_length,
        )
        device = next(self.model.parameters()).device
        scores = []
        for offset in range(0, len(candidates), self.batch_size):
            part = slice(offset, offset + self.batch_size)
            chunk = TokenBatch(batch.ids[part], batch.attention[part], batch.response[part]).to(
                device
            )
            sequence, _ = response_scores(self.model, chunk)
            scores.extend((sequence / chunk.response.sum(dim=1)).tolist())
        return candidates[max(range(len(scores)), key=scores.__getitem__)]
