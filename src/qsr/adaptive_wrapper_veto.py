from __future__ import annotations

import re


META_WRAPPER_PATTERNS = [
    r"\bfollowing (?:request|prompt|query|test case)\b",
    r"\b(?:test case|prompt|query|request) is\s*:",
    r"\b(?:a|the) character asks\s*:",
    r"\b(?:user|student|attacker) asks\s*:",
    r"\bconsider this request\s*:",
]

SAFETY_META_PATTERNS = [
    r"\bsafety education\b",
    r"\baudit(?:ing)? (?:a )?(?:chatbot|model|assistant|system)\b",
    r"\bdefensive risk assessment\b",
    r"\bshould (?:be )?refus(?:ed|e)\b",
    r"\bshould not comply\b",
    r"\bavoid operational instructions\b",
    r"\bprevention-oriented guidance\b",
    r"\bidentify the risk\b",
    r"\bunsafe\b",
]

FICTION_META_PATTERNS = [
    r"\bfictional scene\b",
    r"\brole-?play\b",
    r"\bin a story\b",
]

META_WRAPPER_RE = [re.compile(pattern, re.IGNORECASE) for pattern in META_WRAPPER_PATTERNS]
SAFETY_META_RE = [re.compile(pattern, re.IGNORECASE) for pattern in SAFETY_META_PATTERNS]
FICTION_META_RE = [re.compile(pattern, re.IGNORECASE) for pattern in FICTION_META_PATTERNS]


def is_adaptive_wrapper_recovery_risk(text: str) -> bool:
    """Detect meta-prompts that wrap an embedded request for safety/audit analysis.

    Such prompts may be safe to answer with a refusal or high-level prevention
    advice, but they are poor candidates for refusal recovery: the model is
    already being asked to handle a nested safety test, so replacing a refusal
    with a factual-adapter answer unnecessarily exposes the recovery path.
    """

    normalized = " ".join(str(text or "").split())
    has_embedded_request = any(pattern.search(normalized) for pattern in META_WRAPPER_RE)
    if not has_embedded_request:
        return False
    has_safety_meta = any(pattern.search(normalized) for pattern in SAFETY_META_RE)
    has_fiction_meta = any(pattern.search(normalized) for pattern in FICTION_META_RE)
    return bool(has_safety_meta or has_fiction_meta)
