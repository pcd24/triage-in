"""Minimal LLM provider interface used by data generation, label validation and
zero-shot evaluation.

- GeminiProvider: Google Gemini API (free tier works; key in GEMINI_API_KEY).
- LocalHFProvider: an open-weights instruct model run locally with transformers
  (used on a free Kaggle/Colab GPU), so the "second, independent LLM" costs nothing.

Heavy SDKs are imported lazily so this module imports cleanly in any environment.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    name: str
    model: str

    def generate(
        self, system: str, prompt: str, *, temperature: float, max_tokens: int, json_mode: bool
    ) -> LLMResult: ...

    def generate_many(
        self,
        system: str,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> list[LLMResult]: ...


class GeminiProvider:
    """Google Gemini via the official `google-genai` SDK (verified against v2.33.0)."""

    name = "gemini"

    def __init__(
        self,
        model: str,
        requests_per_minute: float = 10.0,
        max_retries: int = 6,
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        from google import genai  # lazy: only needed when this provider is used

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set (put it in .env or Kaggle Secrets).")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.min_interval = 60.0 / requests_per_minute  # free-tier friendly pacing
        self.max_retries = max_retries
        self._last_call = 0.0

    def available_models(self) -> list[str]:
        """Model ids this key can call; used to verify config before spending quota."""
        return [m.name.removeprefix("models/") for m in self.client.models.list()]

    def generate(
        self, system: str, prompt: str, *, temperature: float, max_tokens: int, json_mode: bool
    ) -> LLMResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        delay = 5.0
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                usage = resp.usage_metadata
                return LLMResult(
                    text=resp.text or "",
                    input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                    output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0)
                    + int(getattr(usage, "thoughts_token_count", 0) or 0),
                )
            except Exception as e:  # rate limits / transient server errors
                if attempt == self.max_retries:
                    raise
                logger.warning("Gemini call failed (%s); retry %d in %.0fs", e, attempt, delay)
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
        raise AssertionError("unreachable")

    def generate_many(
        self,
        system: str,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> list[LLMResult]:
        return [
            self.generate(
                system, p, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
            )
            for p in prompts
        ]


class LocalHFProvider:
    """Open-weights chat model run with transformers (free on a Kaggle/Colab GPU).

    `generate_many` batches several prompts per forward pass (left padding), which
    is what makes labelling thousands of examples feasible on a T4.
    """

    name = "local"

    def __init__(self, model: str, max_input_tokens: int = 4096) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"  # decoder-only models must be left-padded to batch
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, device_map="auto")
        self.lm.eval()
        self.max_input_tokens = max_input_tokens

    def _chat(self, system: str, prompt: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_many(
        self,
        system: str,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> list[LLMResult]:
        import torch

        texts = [self._chat(system, p) for p in prompts]
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.lm.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tok.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.lm.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1] :]  # drop the prompt part
        results = []
        for i in range(len(prompts)):
            n_in = int(enc["attention_mask"][i].sum())
            ids = new_tokens[i]
            n_out = int((ids != self.tok.pad_token_id).sum())
            results.append(
                LLMResult(
                    text=self.tok.decode(ids, skip_special_tokens=True),
                    input_tokens=n_in,
                    output_tokens=n_out,
                )
            )
        return results

    def generate(
        self, system: str, prompt: str, *, temperature: float, max_tokens: int, json_mode: bool
    ) -> LLMResult:
        return self.generate_many(
            system, [prompt], temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )[0]


class FakeProvider:
    """Offline stand-in used ONLY by --smoke runs and tests: no network, no cost.

    It reads the ids in the prompt and answers from `lookup` (id -> spec/record).
    Smoke outputs go to separate smoke/ folders and are never used for results.
    """

    name = "fake"
    model = "fake-smoke"
    _WORDS = [
        "bhai",
        "yaar",
        "please",
        "abhi",
        "tak",
        "kab",
        "order",
        "app",
        "sir",
        "ra",
        "inka",
        "raledu",
        "chala",
        "jaldi",
        "help",
        "kya",
        "hai",
        "nahi",
        "ok",
        "thanks",
    ]

    def __init__(self, lookup: dict[str, dict[str, Any]], disagree_rate: float = 0.1) -> None:
        import random

        self.lookup = lookup
        self.disagree_rate = disagree_rate
        self.rng = random.Random(0)

    def _ids(self, prompt: str) -> list[str]:
        import re

        return [i for i in re.findall(r'"?id"?\s*[:=]?\s*"([^"]+)"', prompt) if i in self.lookup]

    def generate_many(
        self,
        system: str,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> list[LLMResult]:
        import json

        from triage_in.labels import TASKS

        out = []
        for p in prompts:
            items = []
            for i in dict.fromkeys(self._ids(p)):  # unique, order kept
                s = self.lookup[i]
                if "Label these" in p:  # labelling request
                    lab = {t: s[t] for t in TASKS}
                    if self.rng.random() < self.disagree_rate:
                        lab["category"] = self.rng.choice(TASKS["category"])
                    items.append({"id": i, **lab})
                else:  # generation request
                    words = " ".join(self.rng.choices(self._WORDS, k=6))
                    items.append({"id": i, "text": f"{s['category']} {s['domain']} {words} {i}"})
            out.append(
                LLMResult(json.dumps(items, ensure_ascii=False), len(p) // 4, 20 * len(items))
            )
        return out

    def generate(
        self, system: str, prompt: str, *, temperature: float, max_tokens: int, json_mode: bool
    ) -> LLMResult:
        return self.generate_many(
            system, [prompt], temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )[0]


def build_provider(spec: dict[str, Any]) -> Provider:
    """Create a provider from a config block like {provider: gemini, model: ...}."""
    kind = spec.get("provider")
    model = spec.get("model")
    if not model:
        raise ValueError(f"No model set for provider {kind!r}; fill it in the config.")
    if kind == "gemini":
        return GeminiProvider(model, requests_per_minute=spec.get("requests_per_minute", 10.0))
    if kind == "local":
        return LocalHFProvider(model, max_input_tokens=spec.get("max_input_tokens", 4096))
    raise ValueError(f"Unknown provider {kind!r} (expected 'gemini' or 'local').")
