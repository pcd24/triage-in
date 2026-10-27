"""Phase 1a: generate synthetic training/validation complaints with an LLM.

Labels come from construction (each prompt states the target labels). Without
--yes the script only prints the planned calls, tokens and cost, then exits.

    python -m triage_in.data.generate_synthetic            # estimate only
    python -m triage_in.data.generate_synthetic --yes      # run (resumable)
    python -m triage_in.data.generate_synthetic --smoke    # offline fake run, 200 rows
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from triage_in import prompts
from triage_in.data.gen_runner import collect, estimate, run_generation
from triage_in.data.spec_plan import plan_specs
from triage_in.llm_eval.providers import FakeProvider, GeminiProvider, Provider, build_provider
from triage_in.utils import data_dirs, load_config, set_seed, setup_logging, write_jsonl

logger = logging.getLogger(__name__)


def check_budget(est: dict[str, Any], max_budget: float | None) -> None:
    cost = est["est_cost_usd"]
    if cost is not None and max_budget is not None and cost > max_budget:
        raise SystemExit(f"Estimated ${cost} exceeds max_budget_usd=${max_budget}. Aborting.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--smoke", action="store_true", help="offline fake provider, ~200 rows")
    ap.add_argument("--yes", action="store_true", help="confirm and actually call the LLM")
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    scfg = cfg["synthetic"]
    n = cfg["smoke_n"] if args.smoke else scfg["n_examples"]
    specs = plan_specs(n, cfg["plan"], seed=cfg["seed"], id_prefix="syn")
    dirs = data_dirs(cfg, args.smoke)
    cache_dir = dirs["raw"] / "synthetic"

    est = estimate(
        specs,
        scfg["batch_size"],
        prompts.TRAIN_GEN_SYSTEM,
        prompts.train_gen_prompt,
        scfg["prices"],
    )
    logger.info("Plan: %s", json.dumps(est))
    if est["est_cost_usd"] is None and not args.smoke:
        logger.info("No prices configured -> assuming the free tier (cost $0, rate-limited).")
    check_budget(est, scfg.get("max_budget_usd"))

    provider: Provider
    if args.smoke:
        provider = FakeProvider({s["id"]: s for s in specs})
    else:
        if not args.yes:
            logger.info("Estimate only. Re-run with --yes to generate.")
            return
        provider = build_provider(scfg["generator"])
        if isinstance(provider, GeminiProvider):
            models = provider.available_models()
            if provider.model not in models:
                raise SystemExit(
                    f"Model {provider.model!r} not available to this key. Options: {models}"
                )

    run_generation(
        specs,
        provider,
        system=prompts.TRAIN_GEN_SYSTEM,
        prompt_fn=prompts.train_gen_prompt,
        cache_dir=cache_dir,
        batch_size=scfg["batch_size"],
        temperature=scfg["temperature"],
        max_tokens=scfg["max_tokens"],
    )
    records, usage = collect(specs, cache_dir, scfg["batch_size"], source="synthetic")
    out = dirs["interim"] / "synthetic.jsonl"
    write_jsonl(out, records)
    (dirs["interim"] / "synthetic_usage.json").write_text(
        json.dumps(usage, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d records to %s; usage %s", len(records), out, usage)


if __name__ == "__main__":
    main()
