"""Batched, cached, resumable LLM generation shared by the train and test generators.

Every raw reply is cached to `<cache_dir>/<batch>.json` before parsing, so a crash or
quota limit never wastes calls: rerunning skips finished batches.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from triage_in.data.llm_json import extract_json_list
from triage_in.data.spec_plan import batched
from triage_in.llm_eval.providers import Provider

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.5  # rough; code-mixed text tokenizes worse than plain English
OUT_TOKENS_PER_WORD = 2.2


def estimate(
    specs: list[dict[str, Any]],
    batch_size: int,
    system: str,
    prompt_fn: Callable[[list[dict[str, Any]]], str],
    prices: dict[str, float | None] | None,
) -> dict[str, Any]:
    """Count calls and rough tokens; cost only if real prices are configured."""
    batches = batched(specs, batch_size)
    in_tok = sum((len(system) + len(prompt_fn(b))) / CHARS_PER_TOKEN for b in batches)
    out_tok = sum(s["length"] * OUT_TOKENS_PER_WORD + 15 for s in specs)
    est: dict[str, Any] = {
        "n_examples": len(specs),
        "n_calls": len(batches),
        "est_input_tokens": int(in_tok),
        "est_output_tokens": int(out_tok),
        "est_cost_usd": None,
    }
    if prices and prices.get("input") is not None and prices.get("output") is not None:
        est["est_cost_usd"] = round(
            in_tok / 1e6 * prices["input"] + out_tok / 1e6 * prices["output"], 4
        )
    return est


def _cache_ok(path: Path, ids: list[str] | None = None) -> bool:
    """True if this batch is cached and usable.

    A batch that previously failed to parse is re-parsed from its cached reply first
    (the parser may have improved since), so a fix never costs another API call.
    """
    if not path.exists():
        return False
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if entry.get("ok"):
        return True
    if ids is None or not entry.get("response"):
        return False
    try:
        texts = _parse_texts(entry["response"], ids)
    except ValueError:
        return False
    if len(texts) < 0.8 * len(ids):
        return False
    entry.update(ok=True, n_parsed=len(texts))
    entry.pop("error", None)
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def _parse_texts(reply: str, ids: list[str]) -> dict[str, str]:
    items = extract_json_list(reply)
    wanted = set(ids)
    texts = {}
    for it in items:
        if str(it.get("id")) in wanted and isinstance(it.get("text"), str) and it["text"].strip():
            texts[str(it["id"])] = it["text"].strip()
    return texts


def run_generation(
    specs: list[dict[str, Any]],
    provider: Provider,
    *,
    system: str,
    prompt_fn: Callable[[list[dict[str, Any]]], str],
    cache_dir: Path,
    batch_size: int,
    calls_per_step: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    skip: set[int] | None = None,
) -> None:
    """Generate text for all specs, writing one cache file per batch.

    `skip`: batch indices already done elsewhere (e.g. by another model's cache).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    batches = list(enumerate(batched(specs, batch_size)))
    todo = [
        (i, b)
        for i, b in batches
        if i not in (skip or set())
        and not _cache_ok(cache_dir / f"b{i:04d}.json", [s["id"] for s in b])
    ]
    logger.info(
        "%d/%d batches already cached; %d to run", len(batches) - len(todo), len(batches), len(todo)
    )
    for step in batched(todo, calls_per_step):
        prompts = [prompt_fn(b) for _, b in step]
        results = provider.generate_many(
            system, prompts, temperature=temperature, max_tokens=max_tokens, json_mode=True
        )
        for (i, b), prompt, res in zip(step, prompts, results, strict=True):
            ids = [s["id"] for s in b]
            entry: dict[str, Any] = {
                "batch": i,
                "provider": provider.name,
                "model": provider.model,
                "prompt": prompt,
                "response": res.text,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
            }
            try:
                texts = _parse_texts(res.text, ids)
                # Accept a batch if most items came back; missing ids are logged, not invented.
                entry["ok"] = len(texts) >= 0.8 * len(ids)
                entry["n_parsed"] = len(texts)
            except ValueError as e:
                entry.update(ok=False, error=str(e))
            (cache_dir / f"b{i:04d}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            if not entry["ok"]:
                logger.warning(
                    "Batch %d not usable (%s); it will be retried on rerun",
                    i,
                    entry.get("error", "too few items"),
                )
        logger.info("cached batches up to %d", step[-1][0])


def done_batches(specs: list[dict[str, Any]], cache_dirs: list[Path], batch_size: int) -> set[int]:
    """Indices of batches that already have a usable reply in any of `cache_dirs`."""
    return {
        i
        for i, b in enumerate(batched(specs, batch_size))
        if any(_cache_ok(d / f"b{i:04d}.json", [s["id"] for s in b]) for d in cache_dirs)
    }


def collect(
    specs: list[dict[str, Any]],
    cache_dirs: Path | list[Path],
    batch_size: int,
    source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join cached texts with their planned labels. Returns (records, usage summary).

    With several cache dirs, each batch is taken from the first dir holding a usable reply,
    so batches finished by one model are kept when another model fills the gaps.
    """
    dirs = [cache_dirs] if isinstance(cache_dirs, Path) else list(cache_dirs)
    records: list[dict[str, Any]] = []
    usage: dict[str, Any] = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "missing": 0,
        "failed_batches": 0,
        "batches_by_model": {},
    }
    for i, b in enumerate(batched(specs, batch_size)):
        entries = []
        for d in dirs:
            path = d / f"b{i:04d}.json"
            if path.exists():
                entries.append(json.loads(path.read_text(encoding="utf-8")))
        usage["calls"] += len(entries)
        usage["input_tokens"] += sum(e.get("input_tokens", 0) for e in entries)
        usage["output_tokens"] += sum(e.get("output_tokens", 0) for e in entries)
        entry = next((e for e in entries if e.get("ok")), None)
        if entry is None:
            usage["failed_batches"] += 1
            usage["missing"] += len(b)
            continue
        model = entry.get("model")
        usage["batches_by_model"][model] = usage["batches_by_model"].get(model, 0) + 1
        texts = _parse_texts(entry["response"], [s["id"] for s in b])
        for s in b:
            if s["id"] not in texts:
                usage["missing"] += 1
                continue
            records.append(
                {
                    "id": s["id"],
                    "text": texts[s["id"]],
                    "category": s["category"],
                    "sentiment": s["sentiment"],
                    "urgency": s["urgency"],
                    "lang_mix": s["lang_mix"],
                    "source": source,
                    "gen_model": model,
                    "domain": s["domain"],
                }
            )
    return records, usage
