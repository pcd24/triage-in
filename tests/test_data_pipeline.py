"""Tests for Phase 1: LLM JSON parsing, planning, public-data parsing, dedup, test-set merge."""

import csv
from collections import Counter

import pytest

from triage_in.data.build_splits import (
    char_vectorizer,
    dedup_and_guard,
    exact_dedup,
    near_dup_mask,
    split_train_val,
)
from triage_in.data.llm_json import extract_json, extract_json_list
from triage_in.data.load_public import lang_mix_from_tags, parse_cmtet, parse_sentimix_conll
from triage_in.data.merge_manual import check, read_manual, read_reviewed
from triage_in.data.spec_plan import batched, exact_counts, plan_specs
from triage_in.data.validate_labels import agreement_report, parse_labels
from triage_in.labels import TASKS, validate_record
from triage_in.utils import load_config

# --------------------------------------------------------------------------- llm json


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n[{"id": "a", "text": "hi"}]\n```') == [{"id": "a", "text": "hi"}]
    assert extract_json('Sure! Here you go: {"x": 1} hope it helps') == {"x": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_extract_json_list_unwraps_single_list_object() -> None:
    assert extract_json_list('{"messages": [{"id": "1", "text": "t"}]}') == [
        {"id": "1", "text": "t"}
    ]
    with pytest.raises(ValueError):
        extract_json_list('{"a": [], "b": []}')


# --------------------------------------------------------------------------- planning


def test_exact_counts_sums_to_n() -> None:
    c = exact_counts(6000, {"hinglish": 0.4, "tenglish": 0.3, "english": 0.25, "other": 0.05})
    assert c == {"hinglish": 2400, "tenglish": 1800, "english": 1500, "other": 300}
    assert sum(exact_counts(7, {"a": 1, "b": 1, "c": 1}).values()) == 7


def test_plan_specs_mix_is_exact_and_deterministic() -> None:
    plan_cfg = load_config("configs/data.yaml")["plan"]
    a = plan_specs(600, plan_cfg, seed=1, id_prefix="syn")
    b = plan_specs(600, plan_cfg, seed=1, id_prefix="syn")
    assert a == b
    assert Counter(s["lang_mix"] for s in a) == {
        "hinglish": 240,
        "tenglish": 180,
        "english": 150,
        "other": 30,
    }
    assert set(Counter(s["category"] for s in a).values()) == {100}  # balanced categories
    # Positive messages are never planned as high urgency (weight 0 in config).
    assert not [s for s in a if s["sentiment"] == "positive" and s["urgency"] == "high"]
    assert all(s["script"] == "latin" for s in a if s["lang_mix"] in ("english", "other"))
    assert [len(x) for x in batched(a, 250)] == [250, 250, 100]


# ------------------------------------------------------------------------ public data

CMTET_SAMPLE = (
    "NEG: worst service bro chala late ga vachindi\nen en en te te en te\n\n\n"
    "POS: super delivery thanks\nen en en\n\n\n"
    "XXX: malformed label line\nen\n\n\n"
    "NTL: Bhaya enti review\nne te en\n"
)

SENTIMIX_SAMPLE = """meta\t101\tpositive
bahut\tHin
accha\tHin
service\tEng

meta\t102\tnegative
worst\tEng
app\tEng

meta\t103\tmixed
kuch\tHin
"""


def test_parse_cmtet() -> None:
    recs = parse_cmtet(CMTET_SAMPLE)
    assert [r["sentiment"] for r in recs] == ["negative", "positive", "neutral"]
    assert [r["lang_mix"] for r in recs] == ["tenglish", "english", "tenglish"]
    assert all(r["category"] is None and r["urgency"] is None for r in recs)
    assert all(validate_record(r) == [] for r in recs)


def test_parse_sentimix() -> None:
    recs = parse_sentimix_conll(SENTIMIX_SAMPLE, "train")
    assert [(r["id"], r["sentiment"], r["lang_mix"]) for r in recs] == [
        ("sentimix_train_101", "positive", "hinglish"),
        ("sentimix_train_102", "negative", "english"),
    ]  # 'mixed' is not in our schema and is skipped
    assert recs[0]["text"] == "bahut accha service"


def test_lang_mix_from_tags() -> None:
    assert lang_mix_from_tags(["univ", "ne"], native="te", mixed_name="tenglish") == "other"


# -------------------------------------------------------------------------- dedup


def _r(i: str, text: str, cat: str | None = "refund") -> dict:
    return {
        "id": i,
        "text": text,
        "category": cat,
        "sentiment": "negative",
        "urgency": None if cat is None else "low",
        "lang_mix": "hinglish",
        "source": "t",
    }


def test_exact_dedup_uses_normalization() -> None:
    kept, n = exact_dedup(
        [_r("1", "Refund NAHI aaya"), _r("2", "refund   nahi aaya"), _r("3", "other msg")]
    )
    assert [r["id"] for r in kept] == ["1", "3"] and n == 1


def test_near_dup_keeps_first_and_drops_copies() -> None:
    texts = [
        "bhai mera refund abhi tak nahi aaya, 3 baar call kiya",
        "bhai mera refund abhi tak nahi aaya, 3 baar call kiyaa",  # near copy
        "pizza was cold and delivery boy was rude",
    ]
    X = char_vectorizer().fit_transform(texts)
    assert near_dup_mask(X, 0.9, chunk=2).tolist() == [False, True, False]


def test_leakage_guard_removes_pool_rows_similar_to_test() -> None:
    pool = [
        _r("p1", "my upi payment failed but money got debited twice"),
        _r("p2", "app crashes on login screen"),
    ]
    test = [_r("t1", "My UPI payment failed but money got debited twice!!")]
    kept, test_out, counts = dedup_and_guard(pool, test, 0.9)
    assert [r["id"] for r in kept] == ["p2"]
    assert counts["removed_for_test_similarity"] == 1 and len(test_out) == 1


def test_split_is_stratified_including_null_stratum() -> None:
    pool = [_r(f"c{i}", f"text {i}", TASKS["category"][i % 6]) for i in range(120)]
    pool += [_r(f"n{i}", f"public {i}", None) for i in range(60)]
    train, val = split_train_val(pool, 0.1, "category", seed=0)
    assert len(val) == 18 and len(train) == 162
    assert Counter(r["category"] for r in val)[None] == 6


# ---------------------------------------------------------------------- validation


def test_parse_labels_rejects_out_of_schema() -> None:
    reply = (
        '[{"id": "a", "category": "Refund", "sentiment": "negative", "urgency": "high"},'
        ' {"id": "b", "category": "shipping", "sentiment": "negative", "urgency": "high"},'
        ' {"id": "zzz", "category": "refund", "sentiment": "negative", "urgency": "high"}]'
    )
    assert parse_labels(reply, ["a", "b"]) == {
        "a": {"category": "refund", "sentiment": "negative", "urgency": "high"}
    }


def test_agreement_report() -> None:
    recs = [_r("a", "x"), _r("b", "y")]
    pred = {
        "a": {"category": "refund", "sentiment": "negative", "urgency": "low"},
        "b": {"category": "payment", "sentiment": "negative", "urgency": "high"},
    }
    rep = agreement_report(recs, pred)
    assert rep["agreement"] == {"category": 0.5, "sentiment": 1.0, "urgency": 0.5}


# ------------------------------------------------------------------ test-set merge

FIELDS = ["id", "text", "category", "sentiment", "urgency", "lang_mix", "verified", "notes"]


def _write_csv(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_reviewed_csv_must_be_fully_checked(tmp_path) -> None:
    row = {
        "id": "tst_1",
        "text": "refund kab aayega",
        "category": "refund",
        "sentiment": "neutral",
        "urgency": "medium",
        "lang_mix": "hinglish",
        "verified": "",
        "notes": "",
    }
    p = tmp_path / "review.csv"
    _write_csv(p, [row, {**row, "id": "tst_2", "verified": "drop"}])
    with pytest.raises(SystemExit):
        read_reviewed(p)
    _write_csv(p, [{**row, "verified": "yes"}, {**row, "id": "tst_2", "verified": "drop"}])
    out = read_reviewed(p)
    assert [r["id"] for r in out] == ["tst_1"]
    check(out)


def test_manual_csv_gets_ids_and_bad_labels_fail(tmp_path) -> None:
    p = tmp_path / "manual.csv"
    _write_csv(
        p,
        [
            {
                "id": "",
                "text": "Order nahi aaya",
                "category": "Delivery",
                "sentiment": "negative",
                "urgency": "medium",
                "lang_mix": "Hinglish",
                "verified": "",
                "notes": "",
            }
        ],
    )
    recs = read_manual(p)
    assert recs[0]["id"] == "man_0000" and recs[0]["category"] == "delivery"
    check(recs)
    with pytest.raises(SystemExit):
        check([{**recs[0], "urgency": None}])  # test rows need every label


def test_extract_json_accepts_unquoted_keys() -> None:
    # Seen from gemini-3.5-flash-lite: JS-style keys. Colons inside text must survive.
    reply = '[\n  {\n    id: "syn_1",\n    text: "Order: 12345, status: pending"\n  }\n]'
    assert extract_json_list(reply) == [{"id": "syn_1", "text": "Order: 12345, status: pending"}]


def test_failed_cached_batch_is_reparsed_without_new_call(tmp_path) -> None:
    import json as _json

    from triage_in.data.gen_runner import _cache_ok

    p = tmp_path / "b0000.json"
    reply = '[{id: "a", text: "refund kab aayega"}, {id: "b", text: "order late hai"}]'
    p.write_text(_json.dumps({"ok": False, "response": reply, "error": "old parser"}))
    assert _cache_ok(p, ["a", "b"]) is True
    assert _json.loads(p.read_text())["ok"] is True


def test_training_plan_is_frozen() -> None:
    # Cached generator replies are joined to planned labels by id, so the training
    # plan must never change silently. Update this hash only if you regenerate 1a.
    import hashlib
    import json as _json

    cfg = load_config("configs/data.yaml")
    specs = plan_specs(6000, cfg["plan"], seed=cfg["seed"], id_prefix="syn")
    digest = hashlib.sha256(_json.dumps(specs, sort_keys=True).encode()).hexdigest()
    assert digest == "1453718748d404dfee0c7123c59cb594858c042b78a80287db724b4dfbb615be"


def test_domains_by_category_is_respected() -> None:
    cfg = load_config("configs/data.yaml")
    plan_cfg = {**cfg["plan"], "domains_by_category": cfg["test_synthetic"]["domains_by_category"]}
    specs = plan_specs(600, plan_cfg, seed=3, id_prefix="tst")
    assert not [s for s in specs if s["category"] == "delivery" and s["domain"] == "banking/UPI"]


def test_filter_and_mask() -> None:
    from triage_in.data.validate_labels import filter_and_mask

    recs = [_r("a", "x"), _r("b", "y"), _r("c", "z")]  # refund / negative / low
    pred = {
        "a": {"category": "refund", "sentiment": "negative", "urgency": "low"},  # full agree
        "b": {"category": "refund", "sentiment": "neutral", "urgency": "high"},  # mask 2
        "c": {"category": "payment", "sentiment": "negative", "urgency": "low"},  # drop
    }
    out = filter_and_mask(recs, pred, ["category"], ["sentiment", "urgency"])
    assert [r["id"] for r in out] == ["a", "b"]
    assert (out[1]["sentiment"], out[1]["urgency"], out[1]["category"]) == (None, None, "refund")
    assert recs[1]["urgency"] == "low"  # input not mutated


def test_review_csv_overwrite_guard(tmp_path) -> None:
    from triage_in.data.generate_test import has_review_work

    p = tmp_path / "r.csv"
    p.write_text("id,text,verified\na,x,\nb,y,\n", encoding="utf-8")
    assert has_review_work(p) is False  # untouched -> safe to regenerate
    p.write_text("id,text,verified\na,x,yes\nb,y,\n", encoding="utf-8")
    assert has_review_work(p) is True


def test_collect_prefers_first_usable_cache(tmp_path) -> None:
    import json as _json

    from triage_in.data.gen_runner import collect, done_batches

    specs = [
        {
            "id": f"t{i}",
            "category": "refund",
            "sentiment": "negative",
            "urgency": "low",
            "lang_mix": "hinglish",
            "domain": "e-commerce",
        }
        for i in range(4)
    ]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    def put(d, i, ok, model):
        ids = [s["id"] for s in specs[i * 2 : i * 2 + 2]]
        resp = _json.dumps([{"id": x, "text": f"{model} {x}"} for x in ids])
        (d / f"b{i:04d}.json").write_text(_json.dumps({"ok": ok, "response": resp, "model": model}))

    put(a, 0, True, "m1")  # batch 0 done by model 1
    put(b, 0, True, "m2")  # also by model 2 -> model 1 must win
    put(b, 1, True, "m2")  # batch 1 only by model 2
    assert done_batches(specs, [a], 2) == {0}
    recs, usage = collect(specs, [a, b], 2, source="x")
    assert [r["gen_model"] for r in recs] == ["m1", "m1", "m2", "m2"]
    assert usage["batches_by_model"] == {"m1": 1, "m2": 1}
