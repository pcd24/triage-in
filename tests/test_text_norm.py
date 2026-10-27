import subprocess
import sys

import pytest

from triage_in.text_norm import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pleeeeease", "pleeease"),
        ("pleeease", "pleeease"),  # exactly 3 stays
        ("  Bhai   REFUND\tabhi\nTAK nahi aaya  ", "bhai refund abhi tak nahi aaya"),
        ("!!!!!!", "!!!"),
        ("😡😡😡😡😡", "😡😡😡"),
        ("", ""),
        ("   ", ""),
        ("Order #12345 NOT delivered", "order #12345 not delivered"),
    ],
)
def test_basic_rules(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_devanagari_and_telugu_preserved() -> None:
    # Indic scripts must survive untouched (no stripping, no case change).
    assert normalize("रिफंड नहीं मिला") == "रिफंड नहीं मिला"
    assert normalize("డబ్బులు రాలేదు") == "డబ్బులు రాలేదు"
    assert normalize("Refund ఇంకా రాలేదు BRO") == "refund ఇంకా రాలేదు bro"


def test_only_latin_is_lowercased() -> None:
    # Cyrillic and Greek are cased scripts; the spec lowercases Latin only.
    assert normalize("ПРИВЕТ ABC") == "ПРИВЕТ abc"
    assert normalize("ΩMEGA") == "Ωmega"


def test_nfc_composition() -> None:
    decomposed = "Café"  # e + combining acute
    assert normalize(decomposed) == "café"


def test_idempotent() -> None:
    s = "Yaaaaar   KYA bakwaas SERVICE hai!!!!!"
    assert normalize(normalize(s)) == normalize(s)


def test_shared_modules_do_not_import_frameworks() -> None:
    code = (
        "import sys, triage_in.labels, triage_in.text_norm;"
        "bad=[m for m in ('torch','tensorflow') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
