"""
bert_dialect_classifier.py
===========================
Fine-tunes a small BERT variant (DistilBERT by default) to classify English
news text as US / UK / AUS dialect, trained on the same ``articles/`` corpus
used by ``dialect_classifier.py`` (POS n-gram Bayes) and ``spelling_markers.py``
(dictionary spelling markers).

This is the third leg of a three-model comparison, each reading a different
level of linguistic signal:

    1. spelling_markers.py    — lexical   (individual word spellings)
    2. dialect_classifier.py  — syntactic (POS n-gram structure, Bayes)
    3. bert_dialect_classifier.py — contextual (learned subword representations)

Why leakage-aware cleaning matters
-----------------------------------
Dialect in this corpus is 1:1 with source outlet (ap_us→US, bbc→UK,
abc_au→AUS), and each outlet has near-universal formatting fingerprints that
have nothing to do with dialect: AP wire datelines ("WASHINGTON (AP) —"),
BBC's "- Published" boilerplate, ABC AU's "Thu 18 Jun 2026 at 11:24pm"
timestamp line, its "In short: / What's next?" lead-summary box labels, and
embedded-widget artifacts ("Loading..."). A model as
powerful as BERT will happily learn these as shortcuts and report near-100%
accuracy without ever learning a dialectal cue. ``clean_article_text()``
strips the known fingerprints before tokenization, and ``run_bow_baseline()``
provides a cheap TF-IDF + logistic-regression sanity check — if the bag-of-
words baseline scores nearly as high as BERT, that is a leakage smell, not a
BERT win.

Pipeline:
  1. Walk articles/ → (text, dialect, topic, article_id) records
  2. Clean known outlet boilerplate + mask quotes (same masking as the POS
     n-gram model, for comparability — quoted speech reflects the speaker's
     dialect, not the outlet's house style)
  3. Split at the ARTICLE level (stratified by dialect x topic) before any
     chunking, so no article ever has chunks in both train and test
  4. Tokenize with sliding-window overflow (articles run ~1000 words, well
     over BERT's 512-token limit)
  5. Fine-tune DistilBERT with a 3-way classification head
  6. Aggregate chunk-level predictions back to article-level (mean softmax)
     for the metrics that are actually reported
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from helpers.dialect_classifier import ARTICLES_DIR, DEMO_TEXTS, DIALECTS, OUTLET_TO_DIALECT, mask_quotes

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# "Small BERT" — DistilBERT keeps ~97% of BERT-base's language understanding
# at 40% of the parameters (66M), which matters a lot when fine-tuning on a
# few hundred articles: fewer parameters to overfit with, faster iteration
# on a laptop (CPU/MPS, no GPU required). Swap in "prajjwal1/bert-tiny" or
# "prajjwal1/bert-mini" if you want an even smaller/faster model to compare.
MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = "./bert_dialect_model"

MAX_LENGTH = 256      # tokens per chunk
STRIDE = 32            # token overlap between consecutive chunks of one article
TEST_SIZE = 0.15
VAL_SIZE = 0.15        # taken from the remaining train portion
RANDOM_SEED = 42

NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

LABEL2ID = {d: i for i, d in enumerate(DIALECTS)}
ID2LABEL = {i: d for d, i in LABEL2ID.items()}


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LEAKAGE-AWARE TEXT CLEANING
# ──────────────────────────────────────────────────────────────────────────────

# AP wire dateline: "WASHINGTON (AP) —", "BOGOTA, Colombia (AP) —", etc.
_AP_DATELINE_RE = re.compile(
    r"^[A-Z][A-Za-z.'\- ]{2,40}(?:,\s*[A-Za-z.' ]{2,30})?\s*\(AP\)\s*[—-]\s*"
)

# BBC's boilerplate "- Published" marker (appears right after the headline)
_BBC_PUBLISHED_RE = re.compile(r"\s*-\s*Published\b")

# ABC AU's on-page timestamp: "Thu 18 Jun 2026 at 11:24pm"
_ABC_AU_TIMESTAMP_RE = re.compile(
    r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}\s+at\s+\d{1,2}:\d{2}(?:am|pm)\b"
)

# Embedded-widget scraping artifact seen in ABC AU pieces with social embeds
_LOADING_ARTIFACT_RE = re.compile(r"Loading\.\.\.")

# ABC (AU) template labels around its bullet-point lead-summary box. The
# bullet content itself is left in place (it's real reported prose), but the
# labels are a literal, outlet-specific template fingerprint just like BBC's
# "- Published" — an easy shortcut a model could key on instead of dialect.
_ABC_BOX_LABEL_RE = re.compile(r"\bIn short:\s*|\bWhat's next\?\s*")

# Headline repeated verbatim back-to-back at the start of AP-sourced text,
# e.g. "X happened X happened WASHINGTON (AP) — ..." — a scraping artifact
# (title + first <h1> both captured), not a stylistic feature.
_DUP_HEADLINE_RE = re.compile(r"^(.{8,200}?)\s*\1\s*", re.DOTALL)

_URL_RE = re.compile(r"https?://\S+")


def clean_article_text(text: str) -> str:
    """
    Strip known outlet-fingerprint boilerplate that would let a model
    "cheat" by detecting the source rather than the dialect.

    This is intentionally conservative: it targets literal, mechanical
    artifacts of how each outlet's pages were scraped (wire-service tags,
    publish-date stamps, widget placeholders, duplicated headlines) and
    leaves actual prose — including genuine dialect markers like spelling,
    vocabulary, and honorific punctuation ("Mr" vs "Mr.") — untouched.

    It is a best-effort pass, not an exhaustive one. Residual leakage is
    exactly what run_bow_baseline() is for: a cheap way to sanity-check that
    BERT's accuracy isn't just rediscovering a fingerprint this function
    missed.
    """
    cleaned = _DUP_HEADLINE_RE.sub("", text, count=1)
    cleaned = _AP_DATELINE_RE.sub("", cleaned)
    cleaned = _BBC_PUBLISHED_RE.sub(" ", cleaned)
    cleaned = _ABC_AU_TIMESTAMP_RE.sub(" ", cleaned)
    cleaned = _LOADING_ARTIFACT_RE.sub(" ", cleaned)
    cleaned = _ABC_BOX_LABEL_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def _load_full_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ("full_text", "content", "body", "text", "description"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return " ".join(v for v in data.values() if isinstance(v, str))


def load_labeled_articles(articles_dir: str = ARTICLES_DIR) -> list[dict]:
    """
    Walk articles/<topic>/<outlet>/article_NNN.json and return one record per
    article, carrying enough metadata (topic, article_id) to do a leakage-safe,
    stratified article-level split later.

    Only the per-article-file layout is used (not the top_headlines bundle
    files), since those bundles lack a stable per-article id to group on.

    Exact-duplicate articles (same normalized text under a different article_id,
    possibly even a different topic) are dropped, keeping the first occurrence.
    This matters more than it would for a bag-of-words model: split_articles()
    below splits by article_id, which is blind to content identity, so a
    duplicate pair can land on opposite sides of train/test and let the model
    be "tested" on text it saw verbatim during training. Confirmed empirically
    against this corpus before this fix existed — see project notes.

    Returns:
        [{"article_id": str, "text": str, "dialect": str, "topic": str}, ...]
    """
    records: list[dict] = []
    root = Path(articles_dir)
    seen_hashes: set[str] = set()
    n_dupes = 0

    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for outlet_dir in sorted(p for p in topic_dir.iterdir() if p.is_dir()):
            dialect = OUTLET_TO_DIALECT.get(outlet_dir.name)
            if dialect is None:
                continue
            for fpath in sorted(outlet_dir.glob("*.json")):
                raw_text = _load_full_text(fpath)
                if not raw_text.strip():
                    continue
                text = clean_article_text(raw_text)
                text = mask_quotes(text)
                if not text.strip():
                    continue
                h = hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    n_dupes += 1
                    continue
                seen_hashes.add(h)
                records.append({
                    "article_id": f"{topic_dir.name}/{outlet_dir.name}/{fpath.stem}",
                    "text": text,
                    "dialect": dialect,
                    "topic": topic_dir.name,
                })

    if n_dupes:
        print(f"load_labeled_articles: skipped {n_dupes} duplicate-content article(s)")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ARTICLE-LEVEL STRATIFIED SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def split_articles(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split whole articles (not chunks) into train/val/test.

    Splitting must happen BEFORE tokenization/chunking — otherwise two
    overlapping windows from the same article could land on opposite sides
    of the split, letting the model "recognize" a test article it partially
    saw during training. Stratifying on dialect+topic jointly (rather than
    just dialect) keeps topic balance similar across splits, which matters
    given only ~48 articles per (dialect, topic) cell.
    """
    from sklearn.model_selection import train_test_split

    strata = [f"{r['dialect']}|{r['topic']}" for r in records]

    train_val, test = train_test_split(
        records, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strata
    )
    strata_tv = [f"{r['dialect']}|{r['topic']}" for r in train_val]
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=strata_tv
    )
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CHUNKING + TOKENIZATION
# ──────────────────────────────────────────────────────────────────────────────

def build_chunk_dataset(records: list[dict], tokenizer):
    """
    Turn article-level records into a chunk-level HF ``Dataset``.

    Each article is tokenized with ``return_overflowing_tokens=True`` so
    articles longer than MAX_LENGTH become several overlapping windows
    (STRIDE tokens of context carried across the boundary so a class-word
    right at a cut point isn't stranded alone). Every resulting chunk
    inherits its parent article's dialect label and article_id — the
    article_id is what lets us re-aggregate chunk predictions back to
    document level after inference.
    """
    from datasets import Dataset

    ds = Dataset.from_dict({
        "text": [r["text"] for r in records],
        "label": [LABEL2ID[r["dialect"]] for r in records],
        "article_id": [r["article_id"] for r in records],
    })

    def tokenize_with_overflow(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding=False,
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        tokenized["label"] = [batch["label"][i] for i in sample_map]
        tokenized["article_id"] = [batch["article_id"][i] for i in sample_map]
        return tokenized

    return ds.map(
        tokenize_with_overflow,
        batched=True,
        remove_columns=ds.column_names,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — METRICS (CHUNK-LEVEL, USED DURING TRAINING)
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ARTICLE-LEVEL AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_to_article_level(chunk_dataset, logits: np.ndarray) -> dict[str, dict]:
    """
    Mean-pool chunk softmax probabilities by article_id.

    Chunk-level accuracy is a misleading headline number here: articles vary
    in length (172–3,676 words in this corpus), so a chunk-level score is
    implicitly weighted toward long articles and double-counts a single
    document's stylistic signal many times over. Reporting at the article
    level matches how the POS n-gram model (dialect_classifier.py) scores —
    one prediction per document — which is what makes the two models'
    numbers comparable.

    Returns:
        {article_id: {"true": dialect_str, "probs": {dialect: float, ...}}}
    """
    probs = _softmax(logits)
    per_article_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    per_article_true: dict[str, int] = {}

    for row_probs, article_id, label in zip(
        probs, chunk_dataset["article_id"], chunk_dataset["label"]
    ):
        per_article_probs[article_id].append(row_probs)
        per_article_true[article_id] = label

    result = {}
    for article_id, prob_list in per_article_probs.items():
        mean_probs = np.mean(np.stack(prob_list), axis=0)
        result[article_id] = {
            "true": ID2LABEL[per_article_true[article_id]],
            "probs": {ID2LABEL[i]: float(p) for i, p in enumerate(mean_probs)},
        }
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def report_article_level_metrics(article_results: dict[str, dict], title: str) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = [v["true"] for v in article_results.values()]
    y_pred = [max(v["probs"], key=v["probs"].get) for v in article_results.values()]

    print(f"\n{'=' * 64}\n{title} — article-level ({len(y_true)} articles)\n{'=' * 64}")
    print(classification_report(y_true, y_pred, labels=DIALECTS, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):", DIALECTS)
    print(confusion_matrix(y_true, y_pred, labels=DIALECTS))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — BAG-OF-WORDS LEAKAGE SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

def run_bow_baseline(train: list[dict], test: list[dict]) -> float:
    """
    Cheap TF-IDF + logistic-regression baseline on the SAME cleaned,
    article-level split used for BERT.

    This exists purely as a leakage tripwire. Bag-of-words has no access to
    context, word order, or subword composition — if it scores anywhere near
    BERT's accuracy, that is strong evidence some outlet fingerprint (a
    boilerplate phrase, a recurring proper noun, a byline pattern) survived
    clean_article_text() and both models are exploiting it rather than
    learning dialect. A healthy result looks like: BoW noticeably below
    BERT, and BoW's confusions concentrated on function words / spelling
    variants (colour/color, organisation/organization) rather than topic
    nouns.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform([r["text"] for r in train])
    X_test = vectorizer.transform([r["text"] for r in test])
    y_train = [r["dialect"] for r in train]
    y_test = [r["dialect"] for r in test]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n[Leakage sanity check] TF-IDF + LogisticRegression article-level accuracy: {acc:.3f}")
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — TRAINING ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def build_bert_classifier(articles_dir: str = ARTICLES_DIR, output_dir: str = OUTPUT_DIR):
    """
    Full pipeline: load → clean → split → chunk → fine-tune → evaluate.

    Returns:
        (trainer, tokenizer, test_article_results) — the last is the
        article-level dict from aggregate_to_article_level() on the held-out
        test split, ready for report_article_level_metrics().
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    random.seed(RANDOM_SEED)

    print(f"Loading corpus from '{articles_dir}' …")
    records = load_labeled_articles(articles_dir)
    for dialect in DIALECTS:
        n = sum(1 for r in records if r["dialect"] == dialect)
        print(f"  {dialect}: {n} articles")

    train_records, val_records, test_records = split_articles(records)
    print(f"\nArticle split → train: {len(train_records)}  val: {len(val_records)}  test: {len(test_records)}")

    # Leakage tripwire, run before the expensive part
    run_bow_baseline(train_records, test_records)

    print(f"\nLoading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(DIALECTS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    print("Chunking + tokenizing …")
    train_ds = build_chunk_dataset(train_records, tokenizer)
    val_ds = build_chunk_dataset(val_records, tokenizer)
    test_ds = build_chunk_dataset(test_records, tokenizer)
    print(f"  chunks → train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        report_to=[],
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Article-level evaluation on held-out test split ──────────────────────
    val_logits = trainer.predict(val_ds).predictions
    val_results = aggregate_to_article_level(val_ds, val_logits)
    report_article_level_metrics(val_results, "Validation")

    test_logits = trainer.predict(test_ds).predictions
    test_results = aggregate_to_article_level(test_ds, test_logits)
    report_article_level_metrics(test_results, "Test")

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to '{output_dir}'")

    return trainer, tokenizer, test_results


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — MODULE-LEVEL INFERENCE API (mirrors dialect_classifier.py)
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def load_bert_classifier(model_dir: str = OUTPUT_DIR) -> None:
    """Load a previously fine-tuned model/tokenizer from disk for inference."""
    global _model, _tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _model.eval()


def evaluate_dialect_bert(text: str) -> dict[str, float]:
    """
    Public one-shot API, parallel to dialect_classifier.evaluate_dialect():
    classify raw text and return a probability per dialect.

    Long text is chunked exactly as in training and chunk probabilities are
    mean-pooled, so this scores a whole document the same way
    build_bert_classifier()'s evaluation does.
    """
    import torch

    if _model is None or _tokenizer is None:
        raise RuntimeError("No model loaded — call load_bert_classifier() first.")

    cleaned = mask_quotes(clean_article_text(text))
    encoding = _tokenizer(
        cleaned,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    encoding.pop("overflow_to_sample_mapping", None)

    with torch.no_grad():
        logits = _model(**encoding).logits.numpy()

    mean_probs = _softmax(logits).mean(axis=0)
    return {ID2LABEL[i]: float(p) for i, p in enumerate(mean_probs)}


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — ENTRY POINT & DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer, tokenizer, test_results = build_bert_classifier()

    print("\n" + "=" * 64)
    print("BERT DIALECT EVALUATION DEMO (same probe sentences as dialect_classifier.py)")
    print("=" * 64)

    load_bert_classifier(OUTPUT_DIR)
    for label, text in DEMO_TEXTS.items():
        probs = evaluate_dialect_bert(text)
        predicted = max(probs, key=probs.get)
        print(f"\n[{label}]")
        print(f"  Text : {text[:90]}…")
        for dialect, p in sorted(probs.items(), key=lambda x: -x[1]):
            marker = " ◄" if dialect == predicted else ""
            print(f"  {dialect:<8}  {p:>11.2%}{marker}")
