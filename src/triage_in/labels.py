"""Single source of truth for the label schema.

No torch / tensorflow imports: this module is shared by both environments and its
JSON export (data/labels.json) is bundled into the Android app.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Order matters: the index of a label is its class id in every model and on Android.
TASKS: dict[str, list[str]] = {
    "category": ["refund", "delivery", "payment", "account", "product_quality", "other"],
    "sentiment": ["negative", "neutral", "positive"],
    "urgency": ["low", "medium", "high"],
}
TASK_NAMES: list[str] = list(TASKS)
LANG_MIXES: list[str] = ["hinglish", "tenglish", "english", "other"]

# Value used for a missing label so CrossEntropyLoss(ignore_index=-100) masks it.
IGNORE_INDEX = -100

LABEL2ID: dict[str, dict[str, int]] = {
    t: {lab: i for i, lab in enumerate(v)} for t, v in TASKS.items()
}
ID2LABEL: dict[str, dict[int, str]] = {t: dict(enumerate(v)) for t, v in TASKS.items()}


def num_classes(task: str) -> int:
    return len(TASKS[task])


def encode(task: str, label: str | None) -> int:
    """Label string -> class id; None (unknown) -> IGNORE_INDEX."""
    if label is None:
        return IGNORE_INDEX
    try:
        return LABEL2ID[task][label]
    except KeyError as e:
        raise ValueError(f"Unknown {task!r} label: {label!r}") from e


def decode(task: str, idx: int) -> str:
    return ID2LABEL[task][idx]


def validate_record(rec: dict) -> list[str]:
    """Return a list of schema problems for one JSONL record (empty list = valid)."""
    problems: list[str] = []
    for key in ("id", "text", "lang_mix", "source"):
        if not isinstance(rec.get(key), str) or not rec.get(key):
            problems.append(f"missing or empty field {key!r}")
    if rec.get("lang_mix") not in LANG_MIXES:
        problems.append(f"bad lang_mix {rec.get('lang_mix')!r}")
    for task, labels in TASKS.items():
        if task not in rec:
            problems.append(f"missing task key {task!r} (use null if unknown)")
        elif rec[task] is not None and rec[task] not in labels:
            problems.append(f"bad {task} label {rec[task]!r}")
    if all(rec.get(t) is None for t in TASKS):
        problems.append("all task labels are null")
    return problems


def export_labels_json(path: str | Path) -> Path:
    """Write the schema used by the Android app."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tasks": TASKS, "task_order": TASK_NAMES, "lang_mixes": LANG_MIXES}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the label schema to JSON.")
    parser.add_argument("--out", default="data/labels.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    export_labels_json(args.out)


if __name__ == "__main__":
    main()
