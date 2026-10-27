"""Phase 1b: a second, independent LLM labels every synthetic example blind.

The labeller never sees the intended labels. Examples where it disagrees on
category are dropped; agreement rates for all three tasks are logged, because
low urgency/sentiment agreement says something about how fuzzy those labels are.

    python -m vaani.data.validate_labels --yes
    python -m vaani.data.validate_labels --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from vaani import prompts
from vaani.data.llm_json import extract_json_list
from vaani.data.spec_plan import batched
from vaani.labels import TASK_NAMES, TASKS
from vaani.llm_eval.providers import FakeProvider, Provider, build_provider
from vaani.utils import data_dirs, load_config, read_jsonl, setup_logging, write_jsonl

logger = logging.getLogger(__name__)


def parse_labels(reply: str, ids: list[str]) -> dict[str, dict[str, str]]:
    """id -> {task: label}; drops items with missing ids or out-of-schema labels."""
    wanted, out = set(ids), {}
    for it in extract_json_list(reply):
        i = str(it.get("id"))
        labs = {t: str(it.get(t, "")).strip().lower() for t in TASK_NAMES}
        if i in wanted and all(labs[t] in TASKS[t] for t in TASK_NAMES):
            out[i] = labs
    return out


def agreement_report(
    records: list[dict[str, Any]], predicted: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Per-task agreement between intended and independent labels."""
    labelled = [r for r in records if r["id"] in predicted]
    report: dict[str, Any] = {"n_total": len(records), "n_labelled": len(labelled), "agreement": {}}
    for t in TASK_NAMES:
        same = sum(predicted[r["id"]][t] == r[t] for r in labelled)
        report["agreement"][t] = round(same / len(labelled), 4) if labelled else None
    by_lang: dict[str, dict[str, float]] = {}
    for lang in sorted({r["lang_mix"] for r in labelled}):
        sub = [r for r in labelled if r["lang_mix"] == lang]
        by_lang[lang] = {
            t: round(sum(predicted[r["id"]][t] == r[t] for r in sub) / len(sub), 4)
            for t in TASK_NAMES
        }
    report["agreement_by_lang_mix"] = by_lang
    return report


def label_all(
    records: list[dict[str, Any]],
    provider: Provider,
    cache_dir: Path,
    items_per_call: int,
    calls_per_step: int,
    max_tokens: int,
) -> dict[str, dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    batches = list(enumerate(batched(records, items_per_call)))
    todo = [(i, b) for i, b in batches if not (cache_dir / f"b{i:04d}.json").exists()]
    logger.info(
        "%d/%d label batches cached; %d to run", len(batches) - len(todo), len(batches), len(todo)
    )
    for step in batched(todo, calls_per_step):
        ps = [
            prompts.label_prompt([{"id": r["id"], "text": r["text"]} for r in b]) for _, b in step
        ]
        results = provider.generate_many(
            prompts.LABEL_SYSTEM, ps, temperature=0.0, max_tokens=max_tokens, json_mode=True
        )
        for (i, _), p, res in zip(step, ps, results, strict=True):
            entry = {"model": provider.model, "prompt": p, "response": res.text}
            (cache_dir / f"b{i:04d}.json").write_text(
                json.dumps(entry, ensure_ascii=False), encoding="utf-8"
            )

    predicted: dict[str, dict[str, str]] = {}
    parse_failures = 0
    for i, b in batches:
        path = cache_dir / f"b{i:04d}.json"
        if not path.exists():
            continue
        try:
            predicted.update(
                parse_labels(
                    json.loads(path.read_text(encoding="utf-8"))["response"], [r["id"] for r in b]
                )
            )
        except ValueError:
            parse_failures += 1
    if parse_failures:
        logger.warning(
            "%d label batches could not be parsed; delete their cache files to retry",
            parse_failures,
        )
    return predicted


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm and run the labeller")
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    vcfg = cfg["validation"]
    dirs = data_dirs(cfg, args.smoke)
    records = read_jsonl(dirs["interim"] / "synthetic.jsonl")
    n_calls = -(-len(records) // vcfg["items_per_call"])
    logger.info(
        "Labeller %s: %d examples in %d calls (local model: $0)",
        vcfg["labeler"]["model"],
        len(records),
        n_calls,
    )

    if args.smoke:
        provider: Provider = FakeProvider({r["id"]: r for r in records})
    else:
        if not args.yes:
            logger.info("Estimate only. Re-run with --yes to label.")
            return
        provider = build_provider(vcfg["labeler"])

    predicted = label_all(
        records,
        provider,
        dirs["raw"] / "validation",
        vcfg["items_per_call"],
        vcfg["calls_per_step"],
        vcfg["max_tokens"],
    )
    report = agreement_report(records, predicted)
    kept = [
        r
        for r in records
        if r["id"] in predicted and predicted[r["id"]]["category"] == r["category"]
    ]
    report["n_kept"] = len(kept)
    report["labeler_model"] = provider.model
    write_jsonl(dirs["interim"] / "validated.jsonl", kept)
    (dirs["interim"] / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("Agreement: %s | kept %d of %d", report["agreement"], len(kept), len(records))


if __name__ == "__main__":
    main()
