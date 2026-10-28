"""Phase 1d (part 1): generate ~350 candidate TEST messages for human review.

Uses a different model and different prompts than the training generator, so the
test set does not inherit the training generator's phrasing. Output is a CSV you
check by hand: fix any wrong label and set `verified` to `yes` (or `drop`).

    python -m triage_in.data.generate_test --yes
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from triage_in import prompts
from triage_in.data.gen_runner import collect, done_batches, run_generation
from triage_in.data.spec_plan import plan_specs
from triage_in.llm_eval.providers import FakeProvider, GeminiProvider, Provider, build_provider
from triage_in.utils import REPO_ROOT, data_dirs, load_config, setup_logging

logger = logging.getLogger(__name__)

REVIEW_COLUMNS = [
    "id",
    "text",
    "category",
    "sentiment",
    "urgency",
    "lang_mix",
    "verified",
    "notes",
    "gen_model",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm and run the generator")
    ap.add_argument("--force", action="store_true", help="overwrite an existing review CSV")
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    tcfg = cfg["test_synthetic"]
    n = cfg["smoke_n"] // 4 if args.smoke else tcfg["n_examples"]
    plan_cfg = {**cfg["plan"], "domains_by_category": tcfg.get("domains_by_category")}
    specs = plan_specs(n, plan_cfg, seed=tcfg["seed"], id_prefix="tst")
    dirs = data_dirs(cfg, args.smoke)
    review_csv = (
        dirs["interim"] / "test_synthetic_review.csv"
        if args.smoke
        else REPO_ROOT / cfg["paths"]["review_test_csv"]
    )
    if review_csv.exists() and not args.force and has_review_work(review_csv):
        raise SystemExit(
            f"{review_csv} already has reviewed rows. Use --force to overwrite your corrections."
        )

    logger.info(
        "Test generator %s: %d messages in %d calls (local model: $0)",
        tcfg["generator"]["model"],
        n,
        -(-n // tcfg["batch_size"]),
    )
    if args.smoke:
        provider: Provider = FakeProvider({s["id"]: s for s in specs})
    else:
        if not args.yes:
            logger.info("Estimate only. Re-run with --yes to generate.")
            return
        provider = build_provider(tcfg["generator"])
        if (
            isinstance(provider, GeminiProvider)
            and provider.model not in provider.available_models()
        ):
            raise SystemExit(f"Model {provider.model!r} is not available to this API key.")

    # One cache folder per generator model, so switching models never reuses another
    # model's replies by accident. Batches already finished by an earlier model listed in
    # `reuse_cache_from` are kept; the current generator only fills the remaining ones.
    root = dirs["raw"] / "test_synthetic"
    cache = root / tcfg["generator"]["model"].replace("/", "__")
    reuse = [root / m.replace("/", "__") for m in tcfg.get("reuse_cache_from") or []]
    done = done_batches(specs, reuse, tcfg["batch_size"])
    if done:
        logger.info("Reusing %d batches from %s", len(done), [d.name for d in reuse])
    run_generation(
        specs,
        provider,
        system=prompts.TEST_GEN_SYSTEM,
        prompt_fn=prompts.test_gen_prompt,
        cache_dir=cache,
        batch_size=tcfg["batch_size"],
        calls_per_step=tcfg["calls_per_step"],
        temperature=tcfg["temperature"],
        max_tokens=tcfg["max_tokens"],
        skip=done,
    )
    records, usage = collect(specs, [*reuse, cache], tcfg["batch_size"], source="synthetic_test")
    write_review_csv(review_csv, records)
    logger.info("Wrote %d rows to %s for review; usage %s", len(records), review_csv, usage)


def has_review_work(path: Path) -> bool:
    """True if any row has been marked in `verified` (i.e. a human started reviewing)."""
    with path.open(encoding="utf-8-sig") as f:
        return any((row.get("verified") or "").strip() for row in csv.DictReader(f))


def write_review_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow({**{k: r.get(k, "") for k in REVIEW_COLUMNS}, "verified": "", "notes": ""})


if __name__ == "__main__":
    main()
