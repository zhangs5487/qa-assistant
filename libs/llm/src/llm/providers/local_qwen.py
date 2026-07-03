"""Local Qwen3 chat provider — runs Qwen3-1.7B locally via transformers.

Zero API cost, no rate limits, data stays local.

Requires:
    pip install transformers torch

Usage:
    Set CHAT_PROVIDER=local_llm and LOCAL_LLM_PATH=./models/qwen3-1.7b in .env
"""

import logging
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import ChatConfig, ChatMessage, ChatProvider

logger = logging.getLogger(__name__)


class LocalQwenChat(ChatProvider):
    """Chat provider backed by a local Qwen3 model via HuggingFace transformers.

    Loads the model once and keeps it in memory for repeated inference.
    Supports CPU and CUDA auto-detection.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
    ):
        """
        Args:
            model_path: Local path to the Qwen3 model directory.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        """
        self._model_path = model_path
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        logger.info(
            "Loading local Qwen model from %s on device=%s ...",
            self._model_path, self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        if self._device == "cpu":
            self._model = self._model.to("cpu")

        logger.info("Local Qwen model loaded successfully.")

    def chat(self, messages: list[ChatMessage], config: ChatConfig | None = None) -> str:
        """Non-streaming chat completion using local Qwen3 model."""
        self._load()

        cfg = config or ChatConfig()
        temperature = cfg.temperature if cfg.temperature > 0 else 0.1
        max_tokens = cfg.max_tokens or 2048

        # Convert ChatMessage list to dict format for the template
        dialog = [{"role": m.role, "content": m.content} for m in messages]

        # Apply Qwen3 chat template (disable thinking mode for faster RAG responses)
        text = self._tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=cfg.top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def chat_stream(
        self, messages: list[ChatMessage], config: ChatConfig | None = None
    ) -> Iterator[str]:
        """Streaming chat completion. Yields tokens as generated."""
        self._load()

        cfg = config or ChatConfig()
        temperature = cfg.temperature if cfg.temperature > 0 else 0.1
        max_tokens = cfg.max_tokens or 2048

        dialog = [{"role": m.role, "content": m.content} for m in messages]
        text = self._tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            generated_ids = inputs.input_ids[0].tolist()
            for _ in range(max_tokens):
                outputs = self._model.generate(
                    **{k: v.to(self._model.device) for k, v in inputs.items()},
                    max_new_tokens=1,
                    temperature=temperature,
                    top_p=cfg.top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
                new_token_id = outputs[0][-1].item()
                generated_ids.append(new_token_id)

                if new_token_id == self._tokenizer.eos_token_id:
                    break

                token_str = self._tokenizer.decode(
                    [new_token_id], skip_special_tokens=True
                )
                if token_str:
                    yield token_str

                # Update inputs for next iteration
                inputs = {
                    "input_ids": outputs[:, -1:],
                    "attention_mask": torch.cat(
                        [inputs["attention_mask"], torch.ones((1, 1), device=self._model.device)],
                        dim=1,
                    ),
                }

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        self._load()
        return len(self._tokenizer.encode(text))
