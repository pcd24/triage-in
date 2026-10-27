"""Phase 1e: deduplicate and build data/processed/{train,val,test}.jsonl.

Steps
1. Pool = validated synthetic + public data. Test = human-verified test set (1d).
2. Exact duplicates (after text normalization) are removed within the pool and test.
3. Near-duplicates (TF-IDF char n-gram cosine > threshold) are removed within the pool.
4. Leakage guard: any pool example too similar to a test example is removed.
5. Pool -> train/val 90/10, stratified by category (public rows, category=null, form
   their own stratum).
6. data_stats.json: label distributions per split and per lang_mix.

    python -m triage_in.data.build_splits
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from triage_in.labels import TASK_NAMES, validate_record
from triage_in.text_norm import normalize
from triage_in.utils import data_dirs, load_config, read_jsonl, set_seed, setup_logging, write_jsonl

logger = logging.getLogger(__name__)
NULL = "_null"  # stratum / stats key for a missing label


def exact_dedup(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first record of each normalized text."""
    seen: set[str] = set()
    kept = []
    for r in records:
        key = normalize(r["text"])
        if key and key not in seen:
            seen.add(key)
            kept.append(r)
    return kept, len(records) - len(kept)


def char_vectorizer() -> TfidfVectorizer:
    # Character n-grams catch near-copies with small spelling/spacing changes,
    # which matters for romanized code-mixed text.
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalize, sublinear_tf=True
    )


def near_dup_mask(X: sparse.csr_matrix, threshold: float, chunk: int = 512) -> np.ndarray:
    """Boolean mask of rows to DROP: row j is dropped if an earlier kept row i has
    cosine(i, j) > threshold. Rows of X must be L2-normalized (TfidfVectorizer default)."""
    n = X.shape[0]
    drop = np.zeros(n, dtype=bool)
    XT = X.T.tocsc()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = (X[start:stop] @ XT).tocsr()
        for local in range(stop - start):
            i = start + local
            if drop[i]:
                continue
            row = sims.getrow(local)
            js = row.indices[row.data > threshold]
            js = js[js > i]
            drop[js] = True
    return drop


def leakage_mask(
    X_pool: sparse.csr_matrix, X_test: sparse.csr_matrix, threshold: float, chunk: int = 2048
) -> np.ndarray:
    """Rows of the pool whose max cosine to any test row exceeds the threshold."""
    out = np.zeros(X_pool.shape[0], dtype=bool)
    if X_test.shape[0] == 0:
        return out
    XT = X_test.T.tocsc()
    for start in range(0, X_pool.shape[0], chunk):
        sims = X_pool[start : start + chunk] @ XT
        out[start : start + chunk] = np.asarray(sims.max(axis=1).todense()).ravel() > threshold
    return out


def dedup_and_guard(
    pool: list[dict[str, Any]], test: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    pool, n_exact = exact_dedup(pool)
    test, n_exact_test = exact_dedup(test)
    vec = char_vectorizer().fit([r["text"] for r in pool + test])
    X_pool = vec.transform([r["text"] for r in pool])
    X_test = vec.transform([r["text"] for r in test])
    near = near_dup_mask(X_pool, threshold)
    leak = leakage_mask(X_pool, X_test, threshold) & ~near
    kept = [r for r, d, lk in zip(pool, near, leak, strict=True) if not (d or lk)]
    counts = {
        "exact_dups_pool": n_exact,
        "exact_dups_test": n_exact_test,
        "near_dups_pool": int(near.sum()),
        "removed_for_test_similarity": int(leak.sum()),
    }
    return kept, test, counts


def split_train_val(pool: list[dict[str, Any]], val_fraction: float, key: str, seed: int):
    strata = [r[key] if r[key] is not None else NULL for r in pool]
    train, val = train_test_split(pool, test_size=val_fraction, stratify=strata, random_state=seed)
    return train, val


def distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {"n": len(records)}
    for t in TASK_NAMES:
        d[t] = dict(Counter(r[t] if r[t] is not None else NULL for r in records))
    d["lang_mix"] = dict(Counter(r["lang_mix"] for r in records))
    d["source"] = dict(Counter(r["source"] for r in records))
    return d


def build_stats(
    splits: dict[str, list[dict[str, Any]]], dedup_counts: dict[str, int]
) -> dict[str, Any]:
    stats: dict[str, Any] = {"dedup": dedup_counts, "splits": {}}
    for name, recs in splits.items():
        by_lang = {
            lang: distribution([r for r in recs if r["lang_mix"] == lang])
            for lang in sorted({r["lang_mix"] for r in recs})
        }
        stats["splits"][name] = {"overall": distribution(recs), "by_lang_mix": by_lang}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    scfg = cfg["splits"]
    dirs = data_dirs(cfg, args.smoke)

    pool = read_jsonl(dirs["interim"] / "validated.jsonl")
    public_path = dirs["interim"] / "public.jsonl"
    if public_path.exists():
        pool += read_jsonl(public_path)
    test_path = dirs["interim"] / "test.jsonl"
    if not test_path.exists():
        raise SystemExit(
            "data/interim/test.jsonl is missing: finish 1d (review + merge_manual) first."
        )
    test = read_jsonl(test_path)

    bad = [r["id"] for r in pool + test if validate_record(r)]
    if bad:
        raise SystemExit(f"{len(bad)} records fail the schema, e.g. {bad[:3]}")
    ids = Counter(r["id"] for r in pool + test)
    if dup := [i for i, c in ids.items() if c > 1]:
        raise SystemExit(f"Duplicate ids across inputs: {dup[:5]}")

    pool, test, counts = dedup_and_guard(pool, test, scfg["near_dup_threshold"])
    train, val = split_train_val(pool, scfg["val_fraction"], scfg["stratify_by"], cfg["seed"])
    splits = {"train": train, "val": val, "test": test}
    for name, recs in splits.items():
        write_jsonl(dirs["processed"] / f"{name}.jsonl", recs)
    stats = build_stats(splits, counts)
    (dirs["processed"] / "data_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Dedup: %s", counts)
    for name, recs in splits.items():
        o = stats["splits"][name]["overall"]
        logger.info(
            "%-5s n=%-6d lang=%s category=%s", name, len(recs), o["lang_mix"], o["category"]
        )


if __name__ == "__main__":
    main()
