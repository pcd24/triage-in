"""Phase 1c: load REAL public code-mixed sentiment data.

Sources (sentiment only, so category and urgency are null and masked in training):
- CMTET: Telugu-English tweets, Kusampudi et al., RANLP 2021. Downloaded from a public
  GitHub copy and checked against a pinned SHA-256.
- SentiMix: SemEval-2020 Task 9 Hinglish tweets. Official CoNLL files; you download them
  and set the paths in configs/data.yaml (skipped with a warning if not set).

    python -m triage_in.data.load_public
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from triage_in.labels import validate_record
from triage_in.utils import REPO_ROOT, data_dirs, load_config, setup_logging, write_jsonl

logger = logging.getLogger(__name__)

CMTET_LABELS = {"NEG": "negative", "NTL": "neutral", "POS": "positive"}
SENTIMIX_LABELS = {"negative": "negative", "neutral": "neutral", "positive": "positive"}


def lang_mix_from_tags(tags: list[str], native: str, mixed_name: str) -> str:
    """Word-level language tags -> our lang_mix bucket.

    Any word in the Indian language (native tag) -> the code-mixed bucket (romanized
    single-language Telugu/Hindi tweets are counted there too); English-only -> english.
    """
    t = {x.lower() for x in tags}
    if native in t:
        return mixed_name
    if "en" in t or "eng" in t:
        return "english"
    return "other"


def parse_cmtet(raw: str) -> list[dict[str, Any]]:
    """CMTET format: 'LAB: text' line, then a line of word-level language tags, blank-separated."""
    records = []
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        head = lines[0]
        lab, sep, text = head.partition(": ")
        if not sep or lab not in CMTET_LABELS or not text.strip():
            continue
        tags = lines[1].split() if len(lines) > 1 else []
        records.append(
            {
                "id": f"cmtet_{len(records):05d}",
                "text": text.strip(),
                "category": None,
                "sentiment": CMTET_LABELS[lab],
                "urgency": None,
                "lang_mix": lang_mix_from_tags(tags, native="te", mixed_name="tenglish"),
                "source": "cmtet",
            }
        )
    return records


def parse_sentimix_conll(raw: str, split_name: str) -> list[dict[str, Any]]:
    """SentiMix CoNLL: 'meta <uid> <label>' then one 'token<TAB>lang' per line."""
    records: list[dict[str, Any]] = []
    uid, label, toks, tags = None, None, [], []

    def flush() -> None:
        if uid is not None and label in SENTIMIX_LABELS and toks:
            records.append(
                {
                    "id": f"sentimix_{split_name}_{uid}",
                    "text": " ".join(toks),
                    "category": None,
                    "sentiment": SENTIMIX_LABELS[label],
                    "urgency": None,
                    "lang_mix": lang_mix_from_tags(tags, native="hin", mixed_name="hinglish"),
                    "source": "sentimix",
                }
            )

    for line in raw.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "meta" and len(parts) >= 3:
            flush()
            uid, label, toks, tags = parts[1], parts[2].lower(), [], []
        elif uid is not None:
            toks.append(parts[0])
            tags.append(parts[1] if len(parts) > 1 else "O")
    flush()
    return records


def fetch_verified(url: str, dest: Path, sha256: str) -> str:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL from config
            dest.write_bytes(r.read())
    data = dest.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(
            f"Checksum mismatch for {dest} (got {digest}); the source may have changed."
        )
    return data.decode("utf-8")


def sample(records: list[dict[str, Any]], k: int | None, seed: int) -> list[dict[str, Any]]:
    if k is None or len(records) <= k:
        return records
    return random.Random(seed).sample(records, k)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    pcfg = cfg["public"]
    dirs = data_dirs(cfg, args.smoke)
    real_raw = data_dirs(cfg, False)["raw"] / "public"  # downloads are shared with smoke runs
    cap = (lambda k: cfg["smoke_n"] // 2) if args.smoke else (lambda k: k)

    out: list[dict[str, Any]] = []
    c = pcfg["cmtet"]
    cmtet = parse_cmtet(fetch_verified(c["url"], real_raw / "cmtet.txt", c["sha256"]))
    logger.info(
        "CMTET: %d parsed, labels %s", len(cmtet), dict(Counter(r["sentiment"] for r in cmtet))
    )
    out += sample(cmtet, cap(c["max_examples"]), cfg["seed"])

    s = pcfg["sentimix"]
    if s.get("train_path"):
        sm = []
        for key, name in (("train_path", "train"), ("dev_path", "dev")):
            if s.get(key):
                sm += parse_sentimix_conll((REPO_ROOT / s[key]).read_text(encoding="utf-8"), name)
        logger.info(
            "SentiMix: %d parsed, labels %s", len(sm), dict(Counter(r["sentiment"] for r in sm))
        )
        out += sample(sm, cap(s["max_examples"]), cfg["seed"])
    else:
        logger.warning(
            "SentiMix paths not set in configs/data.yaml -> skipping Hinglish public data."
        )

    bad = [r["id"] for r in out if validate_record(r)]
    if bad:
        raise SystemExit(f"{len(bad)} public records fail the schema, e.g. {bad[:3]}")
    write_jsonl(dirs["interim"] / "public.jsonl", out)
    summary = {
        src: dict(Counter(r["lang_mix"] for r in out if r["source"] == src))
        for src in {r["source"] for r in out}
    }
    logger.info("Wrote %d public records; lang_mix by source %s", len(out), json.dumps(summary))


if __name__ == "__main__":
    main()
