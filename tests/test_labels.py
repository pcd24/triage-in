import json

import pytest

from vaani import labels as L


def test_schema_sizes() -> None:
    assert [L.num_classes(t) for t in L.TASK_NAMES] == [6, 3, 3]


def test_encode_decode_roundtrip() -> None:
    for task, names in L.TASKS.items():
        for i, name in enumerate(names):
            assert L.encode(task, name) == i
            assert L.decode(task, i) == name


def test_null_label_is_ignore_index() -> None:
    assert L.encode("urgency", None) == L.IGNORE_INDEX == -100


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        L.encode("category", "shipping")


def _rec(**overrides):
    rec = {
        "id": "syn_000001",
        "text": "bhai refund abhi tak nahi aaya",
        "category": "refund",
        "sentiment": "negative",
        "urgency": "medium",
        "lang_mix": "hinglish",
        "source": "synthetic",
    }
    rec.update(overrides)
    return rec


def test_validate_record_ok() -> None:
    assert L.validate_record(_rec()) == []
    # Public sentiment data: category/urgency unknown -> still valid.
    assert L.validate_record(_rec(category=None, urgency=None)) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"category": "shipping"},
        {"lang_mix": "tamil"},
        {"text": ""},
        {"category": None, "sentiment": None, "urgency": None},
    ],
)
def test_validate_record_catches_problems(bad) -> None:
    assert L.validate_record(_rec(**bad))


def test_export_labels_json(tmp_path) -> None:
    out = L.export_labels_json(tmp_path / "labels.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tasks"] == L.TASKS
    assert data["task_order"] == ["category", "sentiment", "urgency"]
