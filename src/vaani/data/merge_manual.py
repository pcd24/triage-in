"""Phase 1d (part 2): build the human-verified test set.

Combines
- the reviewed LLM test CSV (every row must have verified = yes or drop), and
- your hand-written messages in data/manual/test_manual.csv,
into data/interim/test.jsonl. Refuses to run while any reviewed row is unchecked,
so unverified labels can never reach the test set.

    python -m vaani.data.merge_manual
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from vaani.labels import TASK_NAMES, validate_record
from vaani.utils import REPO_ROOT, data_dirs, load_config, setup_logging, write_jsonl

logger = logging.getLogger(__name__)

YES = {"yes", "y", "true", "1", "ok"}
DROP = {"drop", "no", "n", "remove"}


def _clean(row: dict[str, str]) -> dict[str, Any]:
    rec: dict[str, Any] = {k: (row.get(k) or "").strip() for k in ("id", "text", "lang_mix")}
    for t in TASK_NAMES:
        rec[t] = (row.get(t) or "").strip().lower() or None
    rec["lang_mix"] = rec["lang_mix"].lower()
    return rec


def read_reviewed(path: Path, allow_unverified: bool = False) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    unchecked = [
        r["id"] for r in rows if (r.get("verified") or "").strip().lower() not in YES | DROP
    ]
    if unchecked and not allow_unverified:
        raise SystemExit(
            f"{len(unchecked)} rows in {path.name} are not reviewed yet (e.g. {unchecked[:3]}). "
            "Set verified to yes or drop for every row."
        )
    out = []
    for r in rows:
        v = (r.get("verified") or "").strip().lower()
        if v in DROP:
            continue
        if v in YES or allow_unverified:
            out.append({**_clean(r), "source": "synthetic_test_reviewed"})
    return out


def read_manual(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for i, r in enumerate(csv.DictReader(path.open(encoding="utf-8-sig"))):
        if not (r.get("text") or "").strip():
            continue
        rec = _clean(r)
        rec["id"] = rec["id"] or f"man_{i:04d}"
        out.append({**rec, "source": "manual"})
    return out


def check(records: list[dict[str, Any]]) -> None:
    problems = []
    for r in records:
        probs = validate_record(r)
        if any(r[t] is None for t in TASK_NAMES):
            probs.append("test rows need all three labels")
        if probs:
            problems.append(f"{r['id']}: {'; '.join(probs)}")
    dup = [i for i, c in Counter(r["id"] for r in records).items() if c > 1]
    if dup:
        problems.append(f"duplicate ids: {dup[:5]}")
    if problems:
        raise SystemExit("Test set problems:\n  " + "\n  ".join(problems[:30]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument(
        "--smoke", action="store_true", help="accept unreviewed rows (pipeline test only)"
    )
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    dirs = data_dirs(cfg, args.smoke)

    if args.smoke:
        reviewed = read_reviewed(
            dirs["interim"] / "test_synthetic_review.csv", allow_unverified=True
        )
        manual: list[dict[str, Any]] = []
    else:
        reviewed = read_reviewed(REPO_ROOT / cfg["paths"]["review_test_csv"])
        manual = read_manual(REPO_ROOT / cfg["paths"]["manual_test_csv"])
    records = reviewed + manual
    check(records)
    write_jsonl(dirs["interim"] / "test.jsonl", records)
    logger.info(
        "Test set: %d reviewed + %d hand-written = %d", len(reviewed), len(manual), len(records)
    )


if __name__ == "__main__":
    main()
