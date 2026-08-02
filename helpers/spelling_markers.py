"""
spelling_markers.py
-------------------
Builds US / UK / AU spelling-marker dictionaries from SCOWL/VarCon and breame.
Spelling differences only — no terminology.

Usage (from a notebook at the project root):
    from helpers.spelling_markers import load_spelling_markers, check_dialect, compute_lsr

    markers = load_spelling_markers()
    print(check_dialect("colour", markers))   # ['UK', 'AU']
    print(check_dialect("color",  markers))   # ['US']

    result = compute_lsr(text, markers)
    print(result['lsr'])                       # 0.0 – 1.0
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from breame.spelling import AMERICAN_ENGLISH_SPELLINGS, BRITISH_ENGLISH_SPELLINGS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_valid_scowl_pair(us_word: str, au_word: str) -> bool:
    """
    Return True if (us_word, au_word) is a genuine spelling-variant pair.

    Filters false matches produced by SCOWL when US proper nouns (city names)
    happen to share characters with unrelated Australian words, e.g.
    "Farmington" → "lamington".
    """
    # A proper-noun on one side paired with a common word on the other is
    # always a false match.
    if us_word[0].isupper() != au_word[0].isupper():
        return False

    us = us_word.lower().rstrip("’s").rstrip("'s")
    au = au_word.lower().rstrip("’s").rstrip("'s")

    # Accept if ≥ 50 % of characters match at the same position
    shorter = min(len(us), len(au))
    common = sum(1 for i in range(shorter) if us[i] == au[i])
    if common / max(len(us), len(au)) >= 0.5:
        return True

    # Explicit word-form pairs (same concept, different form)
    known = {
        "airplane": "aeroplane", "airplanes": "aeroplanes",
        "airfoil":  "aerofoil",  "airfoils":  "aerofoils",
        "analog":   "analogue",  "catalog":   "catalogue",
        "dialog":   "dialogue",  "epilog":    "epilogue",
        "monolog":  "monologue", "prolog":    "prologue",
    }
    return known.get(us) == au


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_spelling_markers(
    scowl_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """
    Build dialect spelling-marker dictionaries.

    Sources
    -------
    * SCOWL/VarCon  – ``us_au_dictionary.json`` for US ↔ AU spelling pairs
    * breame        – ``AMERICAN_ENGLISH_SPELLINGS`` and
                      ``BRITISH_ENGLISH_SPELLINGS`` for US ↔ UK pairs

    Parameters
    ----------
    scowl_path
        Path to ``us_au_dictionary.json``.
        Defaults to ``spelling_files/us_au_dictionary.json`` at the project root
        (one level up from this file, which now lives in ``helpers/``).

    Returns
    -------
    dict
        ``{'US': {us_form: au_or_uk_equiv, ...},
           'UK': {uk_form: us_equiv, ...},
           'AU': {au_form: us_equiv, ...}}``

        Each value is a dict mapping a dialect-specific spelling to its
        equivalent in the other dialect, for use as a translation reference.
        All keys are as they appear in the source data (typically lowercase
        for common words).
    """
    if scowl_path is None:
        scowl_path = Path(__file__).parent.parent / "spelling_files" / "us_au_dictionary.json"

    with open(scowl_path, encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)

    scowl = {k: v for k, v in raw.items() if _is_valid_scowl_pair(k, v)}

    # US: SCOWL (common words only, no proper nouns) + breame
    us: dict[str, str] = {k: v for k, v in scowl.items() if k[0].islower()}
    for word, equiv in AMERICAN_ENGLISH_SPELLINGS.items():
        if word not in us:
            us[word] = equiv

    # UK: breame
    uk: dict[str, str] = dict(BRITISH_ENGLISH_SPELLINGS)

    # AU: SCOWL AU-side (lowercase) + breame UK forms (AU spelling ≈ UK spelling)
    au: dict[str, str] = {}
    for us_word, au_word in scowl.items():
        if au_word and au_word[0].islower():
            au[au_word] = us_word
    for word, equiv in BRITISH_ENGLISH_SPELLINGS.items():
        if word not in au:
            au[word] = equiv

    return {"US": us, "UK": uk, "AU": au}


def check_dialect(word: str, markers: dict[str, dict[str, str]]) -> list[str]:
    """
    Return which dialect(s) *word* is a spelling marker for.

    Parameters
    ----------
    word    : the token to look up (any case)
    markers : output of :func:`load_spelling_markers`

    Returns
    -------
    list of dialect codes, e.g. ``['UK', 'AU']``, or ``['neutral']``.
    """
    w = word.lower()
    found = [d for d in ("US", "UK", "AU") if w in markers[d]]
    return found or ["neutral"]


def strip_quotes(text: str, placeholder: str = "<QUOTE>") -> str:
    """
    Replace quoted spans in *text* with *placeholder* before dialect analysis.

    Prevents quoted speech from being counted as dialect switches — e.g. an
    AU article quoting an American source should not trigger AU→US flags.

    Handles:
    * Straight double quotes  ``"..."``
    * Curly double quotes     ``“...”``
    * Curly single quotes     ``‘...’``

    Straight single quotes are left untouched to avoid collisions with
    apostrophes inside words (``don't``, ``it's``, etc.).

    Parameters
    ----------
    text        : original article or summary text
    placeholder : replacement string (default ``'<QUOTE>'``)

    Returns
    -------
    str with all quoted spans replaced by *placeholder*
    """
    # Curly double quotes  "..."
    text = re.sub(r'“[^”]*”', placeholder, text)
    # Straight double quotes  "..."
    text = re.sub(r'"[^"]*"', placeholder, text)
    # Curly single quotes  '...'  (typographic, distinct from apostrophe U+0027)
    text = re.sub(r'‘[^’]*’', placeholder, text)
    return text


def find_switches(
    text: str,
    expected_dialect: str,
    markers: dict[str, dict[str, str]],
) -> list[dict]:
    """
    Find every token in *text* whose spelling belongs to a dialect other
    than *expected_dialect*.

    Parameters
    ----------
    text             : text to scan
    expected_dialect : 'US', 'UK', or 'AU' — the dialect the text should be in
    markers          : output of :func:`load_spelling_markers`

    Returns
    -------
    list of dicts, one per switched token::

        {
            "token":         "color",       # word found in text
            "found_dialect": "US",          # dialect that word belongs to
            "equivalent":    "colour",      # what it should be in expected_dialect
            "switch":        "AU -> US",    # human-readable direction
        }
    """
    tokens = re.findall(r"\b[a-zA-Z'\-]+\b", text.lower())
    others = [d for d in ("US", "UK", "AU") if d != expected_dialect]
    results = []
    for tok in tokens:
        for dialect in others:
            if tok in markers[dialect]:
                # Skip if the token is also valid in the expected dialect
                # (e.g. "kilometres" is in both AU and UK markers — not a switch)
                if tok in markers[expected_dialect]:
                    break
                results.append({
                    "token":         tok,
                    "found_dialect": dialect,
                    "equivalent":    markers[dialect][tok],
                    "switch":        f"{expected_dialect} -> {dialect}",
                })
                break
    return results


def compute_lsr(text: str, markers: dict[str, dict[str, str]]) -> dict:
    """
    Compute the Lexical Switch Rate (LSR) from spelling markers alone.

    LSR = US spelling markers found / total dialect spelling markers found.

    A value near **1.0** means the text is predominantly American English;
    near **0.0** means little or no American English spelling is present.

    Parameters
    ----------
    text    : the generated summary or article to evaluate
    markers : output of :func:`load_spelling_markers`

    Returns
    -------
    dict with keys:
        lsr, us_count, uk_count, au_count, total_dialect_tokens,
        sample_us_words, sample_uk_words, sample_au_words
    """
    tokens = re.findall(r"\b[a-zA-Z'\-]+\b", text.lower())

    us_sp = set(markers["US"])
    uk_sp = set(markers["UK"])
    au_sp = set(markers["AU"])

    us_found: list[str] = []
    uk_found: list[str] = []
    au_found: list[str] = []

    for tok in tokens:
        if tok in us_sp:
            us_found.append(tok)
        elif tok in uk_sp:
            uk_found.append(tok)
        elif tok in au_sp:
            au_found.append(tok)

    total = len(us_found) + len(uk_found) + len(au_found)
    lsr = len(us_found) / total if total else 0.0

    return {
        "lsr":                  round(lsr, 4),
        "us_count":             len(us_found),
        "uk_count":             len(uk_found),
        "au_count":             len(au_found),
        "total_dialect_tokens": total,
        "sample_us_words":      list(dict.fromkeys(us_found))[:10],
        "sample_uk_words":      list(dict.fromkeys(uk_found))[:10],
        "sample_au_words":      list(dict.fromkeys(au_found))[:10],
    }
