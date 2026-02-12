# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Wrapper around HuggingFace Qwen 2.5 model adapted for Moshi's interface patterns."""

import logging
import typing as tp

import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "Qwen/Qwen2.5-3B"


class QwenWrapper(nn.Module):
    """Wraps a HuggingFace Qwen 2.5 model with a Moshi-compatible interface.

    This wrapper provides a consistent API matching Moshi's text generation
    interface, exposing ``forward(input_ids) -> logits`` in a way compatible
    with Moshi's sampling utilities (``moshi.utils.sampling.sample_token``).

    Args:
        model: A HuggingFace ``PreTrainedModel`` (typically ``Qwen2ForCausalLM``).
        tokenizer: The corresponding ``PreTrainedTokenizerBase``.
    """

    def __init__(self, model: nn.Module, tokenizer: tp.Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str = DEFAULT_HF_REPO,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        quantize_bits: int | None = None,
    ) -> "QwenWrapper":
        """Load a pretrained Qwen 2.5 model from HuggingFace.

        Args:
            hf_repo: HuggingFace repository id (e.g. ``"Qwen/Qwen2.5-3B"``).
            device: Target device.
            dtype: Model dtype (default ``torch.bfloat16``).
            quantize_bits: If set to 4 or 8, loads the model with bitsandbytes
                quantization (requires ``bitsandbytes`` to be installed and a
                CUDA device).

        Returns:
            A ``QwenWrapper`` instance with the loaded model and tokenizer.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading tokenizer from %s", hf_repo)
        tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)

        load_kwargs: dict[str, tp.Any] = {
            "trust_remote_code": True,
        }

        if quantize_bits is not None:
            if quantize_bits == 4:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif quantize_bits == 8:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"quantize_bits must be 4 or 8, got {quantize_bits}")
            load_kwargs["device_map"] = "auto"
            logger.info("Loading model from %s with %d-bit quantization", hf_repo, quantize_bits)
        else:
            load_kwargs["torch_dtype"] = dtype
            load_kwargs["device_map"] = str(device) if isinstance(device, torch.device) else device
            logger.info("Loading model from %s (dtype=%s, device=%s)", hf_repo, dtype, device)

        model = AutoModelForCausalLM.from_pretrained(hf_repo, **load_kwargs)
        model.eval()

        wrapper = cls(model=model, tokenizer=tokenizer)
        return wrapper

    @property
    def device(self) -> torch.device:
        return next(iter(self.model.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.model.parameters())).dtype

    @property
    def text_card(self) -> int:
        """Vocabulary size."""
        return self.model.config.vocab_size  # type: ignore

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return logits.

        Args:
            input_ids: Token ids of shape ``[B, T]``.

        Returns:
            Logits tensor of shape ``[B, T, vocab_size]``.
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_sampling: bool = True,
        streamer: tp.Any | None = None,
    ) -> str:
        """Generate text given a prompt using the HuggingFace generate API.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            use_sampling: If ``True``, samples from the distribution; otherwise
                uses greedy decoding.
            streamer: Optional HuggingFace ``TextStreamer`` for streaming output.

        Returns:
            The generated text string (excluding the prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generate_kwargs: dict[str, tp.Any] = {
            "max_new_tokens": max_new_tokens,
        }
        if use_sampling:
            generate_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            })
        else:
            generate_kwargs["do_sample"] = False

        if streamer is not None:
            generate_kwargs["streamer"] = streamer

        output_ids = self.model.generate(
            **inputs,
            **generate_kwargs,
        )

        # Strip the prompt tokens from the output
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def generate_step_by_step(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.0,
        use_sampling: bool = True,
    ) -> tp.Generator[tuple[int, str], None, None]:
        """Generate tokens one at a time, yielding ``(token_id, text_piece)`` pairs.

        This is useful for streaming output or integration with Moshi's
        ``sample_token`` utility.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            use_sampling: If ``True``, samples from the distribution; otherwise
                uses greedy decoding.

        Yields:
            Tuples of ``(token_id, decoded_piece)`` for each generated token.
        """
        from ..utils.sampling import sample_token

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        past_key_values = None
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            next_token = sample_token(
                logits,
                use_sampling=use_sampling,
                temp=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # [B]

            token_id = next_token[0].item()

            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            text_piece = self.tokenizer.decode([token_id], skip_special_tokens=False)
            yield token_id, text_piece

            input_ids = next_token.unsqueeze(-1)
