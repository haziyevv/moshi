# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


def multinomial(
    input: torch.Tensor, num_samples: int, replacement=False, *, generator=None
):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """
    input_ = input.reshape(-1, input.shape[-1])
    # We should probably be able to remove this once the following PR has landed:
    # https://github.com/pytorch/pytorch/pull/134818/files
    # In the meantime, we specialize the case no-replacement, nsamples=1 so as not
    # to have a synchronization point.
    if replacement or num_samples != 1:
        output_ = torch.multinomial(
            input_,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
    else:
        q = torch.empty_like(input_).exponential_(1, generator=generator)
        q = input_ / q
        output_ = q.argmax(dim=-1, keepdim=True)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    k = min(k, probs.shape[-1])
    probs, indices = torch.topk(probs, k, dim=-1)
    next_token = multinomial(probs, num_samples=1)
    next_token = indices.gather(-1, next_token)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def apply_repetition_penalty(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Apply repetition penalty to logits based on previously generated tokens.

    Uses scatter-based ops instead of .unique() to remain CUDA-graph compatible.

    Args:
        logits: [*, Card] logits to modify (any number of leading dims).
        previous_tokens: [B, N] tensor of previously generated token ids.
        penalty: Penalty factor (1.0 = no penalty, >1.0 = discourage repetition).
    """
    if penalty == 1.0:
        return logits
    orig_shape = logits.shape
    card = orig_shape[-1]
    flat_logits = logits.reshape(-1, card).clone()
    B = flat_logits.shape[0]
    prev_B = previous_tokens.shape[0]
    if prev_B != B:
        repeat_factor = B // prev_B
        previous_tokens = previous_tokens.repeat_interleave(repeat_factor, dim=0)
    # Clamp token ids to valid range
    prev_clamped = previous_tokens.clamp(0, card - 1)
    # Build a mask: which vocab entries appeared in previous_tokens? [B, Card]
    seen = torch.zeros(B, card, device=logits.device, dtype=torch.bool)
    seen.scatter_(1, prev_clamped, True)
    # Gather scores for seen tokens and apply penalty
    scores = flat_logits.clone()
    penalized = torch.where(scores > 0, scores / penalty, scores * penalty)
    flat_logits = torch.where(seen, penalized, flat_logits)
    return flat_logits.reshape(orig_shape)


def sample_token(
    logits: torch.Tensor,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    previous_tokens: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Given logits of shape [*, Card], returns a LongTensor of shape [*]."""
    # Apply repetition penalty before temperature scaling
    if previous_tokens is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, previous_tokens, repetition_penalty)
    # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
    if use_sampling and temp > 0.0:
        probs = torch.softmax(logits / temp, dim=-1)
        if top_p > 0.0:
            next_token = sample_top_p(probs, p=top_p)
        elif top_k > 0:
            next_token = sample_top_k(probs, k=top_k)
        else:
            next_token = multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    assert next_token.shape[-1] == 1
    return next_token[..., 0]


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    ps = torch.tensor([5.0, 2.0, 12.0, 6.0, 8.0, 1.0, 0.0, 4.0], device=device)
    cnts = torch.zeros(ps.shape, dtype=torch.long, device=device)
    total_samples = 1000
    for _ in range(total_samples):
        vs = multinomial(ps, num_samples=1, replacement=False)
        cnts[vs] += 1
    diff = cnts / cnts.sum() - ps / ps.sum()
    max_diff = diff.abs().max().cpu().item()
    print(ps / ps.sum())
    print(cnts / cnts.sum())
    assert max_diff < 1.5e-2
