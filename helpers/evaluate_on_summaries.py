"""
evaluate_on_summaries.py
=========================
Out-of-domain sanity check: run all three dialect classifiers — spelling
markers, POS n-gram Bayes, and fine-tuned BERT — against the LLM-generated
summaries in summaries/, not the news articles they were trained on.

Why this matters more than the article-level test-set numbers
----------------------------------------------------------------
The real deployment target for this project is classifying AI-generated
summaries, not news articles. Article-level test accuracy told us the models
learned *something* separating ap_us / bbc / abc_au text, but a
TF-IDF+LogisticRegression baseline hit 87.7% on the same split and BERT hit
100% — both consistent with the models keying on outlet/topic fingerprints
that survive despite cleanup (see bert_dialect_classifier.py's
clean_article_text() docstring), not necessarily genuine dialect signal.

Summaries strip almost all of that away: no datelines, no "- Published" /
"In short:" boilerplate, no outlet-specific structure — just ~15-100 words of
an LLM's own prose. If a classifier still separates dialects here, that is
much stronger evidence it learned real linguistic markers. If it collapses to
chance, that tells us the article-trained signal doesn't transfer to the
actual use case, which is a more important finding than any article-level
metric.

Data sources (both already in summaries/):
  * task2_constrained.jsonl — Gemini/Llama summaries EXPLICITLY prompted to
    target a given dialect. Has a reliable ground-truth `target_dialect`
    field — this is the primary eval set.
  * task1_unconstrained.jsonl — summaries with no dialect instruction. Has
    only `article_dialect` (the SOURCE article's dialect) as a much weaker
    proxy label — it tests whether unconstrained summarization preserves the
    source's dialect at all, or whether the LLM defaults to its own house
    style regardless of input.

Sample sizes here are tiny (single digits) because make_summaries.ipynb is
still a work in progress — this script is meant to be re-run as that corpus
grows, not to produce a statistically meaningful accuracy number yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from helpers.dialect_classifier import DIALECTS, build_classifier, evaluate_dialect, scores_to_probs
from helpers.bert_dialect_classifier import evaluate_dialect_bert, load_bert_classifier
from helpers.spelling_markers import compute_lsr, load_spelling_markers

SUMMARIES_DIR = Path("./summaries")

# spelling_markers.py uses "AU"; the rest of the project uses "AUS"
_SPELLING_TO_PROJECT_DIALECT = {"US": "US", "UK": "UK", "AU": "AUS"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def predict_spelling(text: str, markers: dict) -> tuple[str, dict]:
    """
    Predict dialect from spelling markers alone.

    CAVEAT: UK and AU spelling are nearly identical (colour, organisation,
    programme are the same in both), and load_spelling_markers() builds the
    AU dict by falling back to the same breame UK list wherever SCOWL has no
    AU-specific entry. compute_lsr() checks UK before AU, so a shared word
    always counts as a UK hit, not AU. In practice this means spelling
    markers can reliably flag "US" vs "not US" but are structurally biased
    toward predicting UK over AUS whenever both are candidates — that's a
    property of the lexical signal itself (AU and UK really don't spell
    differently), not a bug to patch here.
    """
    result = compute_lsr(text, markers)
    counts = {"US": result["us_count"], "UK": result["uk_count"], "AU": result["au_count"]}
    if result["total_dialect_tokens"] == 0:
        return "no signal", counts
    predicted = max(counts, key=counts.get)
    return _SPELLING_TO_PROJECT_DIALECT[predicted], counts


def predict_pos_ngram(text: str) -> tuple[str, dict]:
    scores = evaluate_dialect(text)
    probs = scores_to_probs(scores)
    return max(probs, key=probs.get), probs


def predict_bert(text: str) -> tuple[str, dict]:
    probs = evaluate_dialect_bert(text)
    return max(probs, key=probs.get), probs


def evaluate_file(rows: list[dict], label_key: str, markers: dict, title: str) -> None:
    if not rows:
        print(f"\n[{title}] no rows found — skipping")
        return

    print(f"\n{'=' * 88}\n{title}  (n={len(rows)}, label field: '{label_key}')\n{'=' * 88}")

    correct = {"spelling": 0, "pos_ngram": 0, "bert": 0}
    header = f"{'true':<5} {'model':<7} {'spelling':<10} {'pos_ngram':<10} {'bert':<6}  summary"
    print(header)
    print("-" * len(header))

    for row in rows:
        text = row["summary"]
        true_label = row[label_key]

        sp_pred, _ = predict_spelling(text, markers)
        pn_pred, _ = predict_pos_ngram(text)
        bert_pred, _ = predict_bert(text)

        correct["spelling"] += int(sp_pred == true_label)
        correct["pos_ngram"] += int(pn_pred == true_label)
        correct["bert"] += int(bert_pred == true_label)

        mark = lambda p: "✓" if p == true_label else "✗"
        print(
            f"{true_label:<5} {row.get('model', '?'):<7} "
            f"{sp_pred:<9}{mark(sp_pred)} {pn_pred:<9}{mark(pn_pred)} {bert_pred:<5}{mark(bert_pred)}  "
            f"{text[:60]}…"
        )

    n = len(rows)
    print(f"\nAccuracy — spelling: {correct['spelling']}/{n}  "
          f"pos_ngram: {correct['pos_ngram']}/{n}  "
          f"bert: {correct['bert']}/{n}")
    print("(n is small — read this as a diagnostic sample, not a reliable accuracy estimate)")


def main() -> None:
    print("Loading spelling markers …")
    markers = load_spelling_markers()

    print("Training POS n-gram classifier on articles/ …")
    build_classifier()

    print(f"\nLoading fine-tuned BERT from ./bert_dialect_model …")
    load_bert_classifier()

    task2 = _load_jsonl(SUMMARIES_DIR / "task2_constrained.jsonl")
    task1 = _load_jsonl(SUMMARIES_DIR / "task1_unconstrained.jsonl")

    evaluate_file(
        task2, "target_dialect", markers,
        "TASK 2 — constrained (explicit target dialect, reliable ground truth)",
    )
    evaluate_file(
        task1, "article_dialect", markers,
        "TASK 1 — unconstrained (does summarization preserve SOURCE dialect? weak proxy label)",
    )


if __name__ == "__main__":
    main()
