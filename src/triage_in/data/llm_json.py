"""Robust JSON extraction from LLM replies (models sometimes add code fences or prose)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)
# JavaScript-style unquoted keys right after "{" or ",", e.g.  { id: "x", text: "..." }
_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:(?=\s*["\[{\d\-tfn])')


def extract_json(text: str) -> Any:
    """Parse the first JSON array/object in `text`.

    Tries, in order: the whole string, the string without ``` fences, then the
    outermost [...] or {...} span. Raises ValueError if nothing parses.
    """
    candidates = [text.strip(), _FENCE_RE.sub("", text).strip()]
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    # Fallback only if strict parsing fails: some models emit unquoted keys.
    candidates += [_BARE_KEY_RE.sub(r'\1"\2":', c) for c in candidates]
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"No valid JSON found in LLM reply: {text[:200]!r}")


def extract_json_list(text: str) -> list[dict[str, Any]]:
    """Like extract_json but always returns a list of dicts.

    Accepts a bare list, or an object wrapping one list (e.g. {"items": [...]}).
    """
    data = extract_json(text)
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) != 1:
            raise ValueError("Expected a JSON list or an object containing exactly one list.")
        data = lists[0]
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError("Expected a JSON list of objects.")
    return data
