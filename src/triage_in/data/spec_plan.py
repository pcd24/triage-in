"""Plan the label combination for every synthetic example *before* generation.

Labels come from construction: the LLM is told exactly which labels to write for,
so the planner controls the label and language mix of the dataset.
"""

from __future__ import annotations

import random
from typing import Any

from triage_in.labels import LANG_MIXES, TASKS


def exact_counts(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Split n into integer counts proportional to weights (largest-remainder method)."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    leftover = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:leftover]:
        counts[k] += 1
    return counts


def _pick(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def plan_specs(n: int, cfg: dict[str, Any], seed: int, id_prefix: str) -> list[dict[str, Any]]:
    """Return n generation specs (labels + style knobs), shuffled deterministically.

    cfg keys: lang_mix_target, sentiment_weights, urgency_given_sentiment, domains,
    styles, length_choices, typo_rate, native_script_rate.
    """
    rng = random.Random(seed)
    langs: list[str] = []
    for lang, c in exact_counts(n, cfg["lang_mix_target"]).items():
        if lang not in LANG_MIXES:
            raise ValueError(f"Unknown lang_mix {lang!r}")
        langs += [lang] * c
    # Category: exactly balanced, so the 6-way task is not skewed by construction.
    cats: list[str] = []
    for cat, c in exact_counts(n, dict.fromkeys(TASKS["category"], 1.0)).items():
        cats += [cat] * c
    rng.shuffle(langs)
    rng.shuffle(cats)

    # Optional category -> compatible domains (e.g. no "delivery" complaint about a UPI
    # app). Only used when configured, so plans made without it stay byte-identical.
    by_cat = cfg.get("domains_by_category") or {}

    def domains_for(cat: str) -> list[str]:
        return by_cat.get(cat, cfg["domains"])

    specs = []
    for i, (lang, cat) in enumerate(zip(langs, cats, strict=True)):
        sentiment = _pick(rng, cfg["sentiment_weights"])
        urgency = _pick(rng, cfg["urgency_given_sentiment"][sentiment])
        if lang in ("hinglish", "tenglish"):
            script = "native" if rng.random() < cfg["native_script_rate"] else "roman"
        else:
            script = "latin"
        specs.append(
            {
                "id": f"{id_prefix}_{i:06d}",
                "category": cat,
                "sentiment": sentiment,
                "urgency": urgency,
                "lang_mix": lang,
                "script": script,
                "domain": rng.choice(domains_for(cat)),
                "style": rng.choice(cfg["styles"]),
                "length": rng.choice(cfg["length_choices"]),
                "typos": rng.random() < cfg["typo_rate"],
            }
        )
    return specs


def batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
