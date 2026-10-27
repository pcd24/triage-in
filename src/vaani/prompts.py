"""All LLM prompt text in one place (no torch / tensorflow imports).

Three distinct prompt families:
- TRAIN_GEN_*: synthetic training data (Gemini).
- TEST_GEN_*:  synthetic test data, deliberately worded differently and run on a
               different model, so the test set does not share the training
               generator's quirks.
- LABEL_*:     blind labelling (validation pass, and zero-shot eval in Phase 4).
"""

from __future__ import annotations

import json
from typing import Any

from vaani.labels import TASKS

LABEL_GUIDE = """Label definitions
- category (topic): refund, delivery, payment, account, product_quality, other
- sentiment (the customer's tone): negative, neutral, positive
- urgency:
  - high: money lost or charged incorrectly, repeated failed attempts ("3 baar call kiya"),
    threats to leave, legal or consumer-forum threats, safety issues, account locked or hacked.
  - medium: a clear problem needing action but no escalation signals (late delivery, wrong item).
  - low: questions, mild feedback, compliments, general queries.
Sentiment is tone, category is topic. A polite message about a double charge is
payment / neutral / high."""

LANG_MIX_DESC = {
    "hinglish": "Hinglish (Hindi and English mixed in one message)",
    "tenglish": "Telugu-English (Telugu and English mixed in one message)",
    "english": "Indian English",
    "other": "another Indian code-mix (e.g. Tamil-English, Kannada-English, Bengali-English)",
}

SCRIPT_DESC = {
    "roman": "Hindi/Telugu words written in Roman letters",
    "native": "some Hindi/Telugu words in Devanagari/Telugu script",
    "latin": "Roman letters only",
}

# ---------------------------------------------------------------- training generation
TRAIN_GEN_SYSTEM = f"""You write realistic customer messages sent to Indian companies' support
channels (chat, app reviews, social media, email). Each message must match the exact labels it
is given. Write like real customers: vary length, formality, slang, abbreviations, and include
natural typos when asked. Never mention the labels, never explain, output only JSON.

{LABEL_GUIDE}"""


def train_gen_prompt(specs: list[dict[str, Any]]) -> str:
    lines = []
    for s in specs:
        lines.append(
            f'- id "{s["id"]}": category={s["category"]}, sentiment={s["sentiment"]}, '
            f"urgency={s['urgency']}; language: {LANG_MIX_DESC[s['lang_mix']]}, "
            f"{SCRIPT_DESC[s['script']]}; domain: {s['domain']}; about {s['length']} words; "
            f"style: {s['style']}{'; include 1-2 typos' if s['typos'] else ''}"
        )
    return (
        f"Write {len(specs)} customer messages, one per spec below.\n\n"
        + "\n".join(lines)
        + '\n\nReturn a JSON array of objects: [{"id": "...", "text": "..."}], '
        "same ids, same order."
    )


# -------------------------------------------------------------------- test generation
TEST_GEN_SYSTEM = """You are helping build an evaluation set for an Indian customer-support
inbox. Imagine real people typing quickly on WhatsApp, Twitter or a company app. Produce messages
that sound authentic, not like textbook examples. Reply with JSON only."""


def test_gen_prompt(specs: list[dict[str, Any]]) -> str:
    parts = []
    for s in specs:
        parts.append(
            json.dumps(
                {
                    "id": s["id"],
                    "issue_topic": s["category"],
                    "customer_mood": s["sentiment"],
                    "how_urgent": s["urgency"],
                    "language": LANG_MIX_DESC[s["lang_mix"]],
                    "writing": SCRIPT_DESC[s["script"]],
                    "company_type": s["domain"],
                    "approx_words": s["length"],
                    "tone": s["style"],
                },
                ensure_ascii=False,
            )
        )
    return (
        "For each scenario, write the one message the customer would actually send.\n"
        "Urgency meaning: high = lost money, wrong charge, repeated failed attempts, threats to "
        "leave or complain legally, safety, hacked/locked account; medium = needs action but no "
        "escalation; low = question, mild feedback or praise.\n\n"
        + "\n".join(parts)
        + '\n\nOutput: {"messages": [{"id": ..., "text": ...}, ...]}'
    )


# --------------------------------------------------------------------------- labelling
LABEL_SYSTEM = f"""You label customer-support messages written in Indian code-mixed language
(Hinglish, Telugu-English, English). Read each message and assign exactly one label per task.

{LABEL_GUIDE}

Allowed values: {json.dumps(TASKS)}
Reply with JSON only."""


def label_prompt(items: list[dict[str, str]]) -> str:
    body = "\n".join(
        json.dumps({"id": it["id"], "text": it["text"]}, ensure_ascii=False) for it in items
    )
    return (
        f"Label these {len(items)} messages.\n\n{body}\n\n"
        'Return a JSON array: [{"id": "...", "category": "...", "sentiment": "...", '
        '"urgency": "..."}], one object per message, same ids.'
    )
