"""Text normalization shared by every model (and mirrored in the Kotlin tokenizer).

No torch / tensorflow imports. Keep the rules simple: each step must be easy to
reimplement exactly on Android.
"""

from __future__ import annotations

import re
import unicodedata

import regex  # third-party: supports \p{Script=...}; the stdlib `re` does not

_LATIN_RE = regex.compile(r"\p{Script=Latin}+")
# A character followed by 3+ copies of itself (4+ in a row) -> keep exactly 3.
_REPEAT_RE = re.compile(r"(.)\1{3,}", flags=re.DOTALL)
_WS_RE = re.compile(r"\s+")
MAX_REPEAT = 3


def _lower_latin(text: str) -> str:
    """Lowercase only Latin-script characters.

    Devanagari and Telugu have no case, but other scripts (e.g. Cyrillic) do; the spec
    asks for Latin-only lowercasing so behaviour is explicit and portable.
    """
    # Unicode Script=Latin (same property Kotlin's Character.UnicodeScript.LATIN uses),
    # so Android lowercases exactly the same characters.
    return _LATIN_RE.sub(lambda m: m.group(0).lower(), text)


def normalize(text: str) -> str:
    """NFC -> lowercase Latin -> cap repeated chars at 3 -> collapse whitespace -> strip.

    >>> normalize("  Pleeeeease   REFUND  ")
    'pleeease refund'
    """
    text = unicodedata.normalize("NFC", text)
    text = _lower_latin(text)
    text = _REPEAT_RE.sub(lambda m: m.group(1) * MAX_REPEAT, text)
    text = _WS_RE.sub(" ", text)
    return text.strip()
