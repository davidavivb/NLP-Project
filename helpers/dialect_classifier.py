"""
Structural Dialect Classifier — POS N-gram Bayesian Model
==========================================================
Classifies English text as US / UK / AUS dialect using Part-of-Speech
bigrams and trigrams as structural features. Trained on a corpus of
news articles scraped from Associated Press (US), BBC, and ABC AU.

Pipeline:
  1. Quote masking  →  <QUOTE> token replaces all quoted segments
  2. spaCy POS tagging  →  fine-grained Penn-Treebank-style tags
  3. N-gram extraction  →  bigrams + trigrams over the tag sequence
  4. Laplace-smoothed Naive Bayes  →  log-posterior per dialect
"""

import hashlib
import os
import re
import json
import math
from collections import Counter, defaultdict

import spacy
from spacy.language import Language
from spacy.tokens import Doc

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ARTICLES_DIR = "./articles"

# Map outlet folder names (and bare stem of bundle .json files) to dialect labels
OUTLET_TO_DIALECT: dict[str, str] = {
    "ap_us":  "US",
    "bbc":    "UK",
    "abc_au": "AUS",
}

DIALECTS: list[str] = ["US", "UK", "AUS"]

QUOTE_TOKEN = "<QUOTE>"   # replaces every quoted segment verbatim
QUOTE_TAG   = "MQ"        # custom POS tag assigned to QUOTE_TOKEN

N_GRAM_SIZES: list[int] = [2, 3]   # bigrams and trigrams


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def mask_quotes(text: str) -> str:
    """
    Replace every quoted segment with the literal token ``<QUOTE>``.

    Handles four quotation styles in priority order so that inner/nested
    quotes are not double-replaced:
      1. Unicode “curly” double quotes  (“ ... “)
      2. Unicode ‘curly’ single quotes  (‘ ... ‘)
      3. Straight double quotes         (“...”)
      4. Straight single quotes used as delimiters, NOT contractions.
         The negative look-around avoids matching apostrophes inside
         words like “don’t” or “Britain’s”.
    """
    # Curly double quotes (most reliable — always quotation marks)
    text = re.sub(r'“[^”]*”', QUOTE_TOKEN, text)
    # Curly single quotes
    text = re.sub(r'‘[^’]*’', QUOTE_TOKEN, text)
    # Straight double quotes
    text = re.sub(r'"[^"]*"', QUOTE_TOKEN, text)
    # Straight single quotes — only when not flanked by word characters
    text = re.sub(r"(?<!\w)'([^']{2,})'(?!\w)", QUOTE_TOKEN, text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SPACY PIPELINE SETUP
# ──────────────────────────────────────────────────────────────────────────────

def build_nlp_pipeline() -> Language:
    """
    Load ``en_core_web_sm`` and patch it so ``<QUOTE>`` is:
      - Treated as a single atomic token (tokenizer special-case rule).
      - Assigned the custom POS tag ``MQ`` after the standard tagger runs,
        overriding whatever the statistical model guessed for it.
    """
    nlp = spacy.load("en_core_web_sm")

    # Tokenizer special-case: treat <QUOTE> as one indivisible token
    from spacy.symbols import ORTH
    nlp.tokenizer.add_special_case(QUOTE_TOKEN, [{ORTH: QUOTE_TOKEN}])

    # Custom pipeline component registered once (guard avoids re-registration
    # when this function is called more than once in the same Python session).
    # Language.has_factory() is the correct spaCy v3 API for this check.
    component_name = "quote_pos_tagger"
    if not Language.has_factory(component_name):
        @Language.component(component_name)
        def _quote_pos_tagger(doc: Doc) -> Doc:
            """Overwrite the fine-grained tag on every <QUOTE> token with MQ.

            spaCy requires token.pos_ to be a valid UD category, so we only
            set token.tag_ (Penn-Treebank-style), which accepts arbitrary values.
            Our n-gram features read from tag_, so this is sufficient.
            """
            for token in doc:
                if token.text == QUOTE_TOKEN:
                    token.tag_ = QUOTE_TAG   # fine-grained: accepts any string
            return doc

    # Insert AFTER the statistical tagger so we always win the override
    nlp.add_pipe(component_name, after="tagger")
    return nlp


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — N-GRAM EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_ngrams(tags: list[str], n: int) -> list[tuple]:
    """Slide a window of width *n* over *tags* and return every tuple."""
    return [tuple(tags[i : i + n]) for i in range(len(tags) - n + 1)]


def text_to_pos_ngrams(text: str, nlp: Language,
                        n_sizes: list[int] = N_GRAM_SIZES) -> list[tuple]:
    """
    Full preprocessing → POS-tag → n-gram extraction for a single text.

    Steps:
      1. Mask quoted segments with ``<QUOTE>``.
      2. Run the patched spaCy pipeline.
      3. Collect fine-grained ``token.tag_`` values (Penn Treebank style,
         e.g. NNS, VBZ) — more discriminative than coarse ``token.pos_``.
         ``<QUOTE>`` tokens get the custom tag ``MQ``.
      4. Extract bigrams and trigrams from the resulting tag sequence.

    Returns a flat list of n-gram tuples ready for counting.
    """
    masked = mask_quotes(text)
    doc    = nlp(masked)

    tags = [
        QUOTE_TAG if token.text == QUOTE_TOKEN else token.tag_
        for token in doc
        if not token.is_space and token.tag_  # skip untagged whitespace
    ]

    ngrams: list[tuple] = []
    for n in n_sizes:
        ngrams.extend(extract_ngrams(tags, n))
    return ngrams


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATA INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def _load_individual_file(path: str) -> str:
    """
    Load a single-article JSON file.
    Expected schema: ``{"full_text": "...", ...}``
    Falls back to ``content`` or ``description`` if ``full_text`` is absent.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for key in ("full_text", "content", "body", "text", "description"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val

    # Last resort: concatenate all string values
    return " ".join(v for v in data.values() if isinstance(v, str))


def _load_bundle_file(path: str) -> list[str]:
    """
    Load a bundle JSON file that wraps multiple articles.
    Expected schema: ``{"source": "...", "articles": [{...}, ...]}``
    Extracts ``content`` (or ``description``) from each article entry.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    texts: list[str] = []
    for article in data.get("articles", []):
        for key in ("content", "full_text", "body", "text", "description"):
            val = article.get(key)
            if isinstance(val, str) and val.strip():
                texts.append(val)
                break
    return texts


def _norm_hash(text: str) -> str:
    """Normalized-content hash used to dedupe articles regardless of filename/topic."""
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


def load_corpus(articles_dir: str = ARTICLES_DIR) -> dict[str, list[str]]:
    """
    Walk the ``articles/`` directory tree and return a per-dialect text corpus.

    Supported layouts (both can coexist in the same topic folder):

    ① Per-article files:
       ``articles/<topic>/<outlet>/article_NNN.json``
       → ``{"full_text": "...", "country": "US", ...}``

    ② Bundle files (e.g. top_headlines):
       ``articles/<topic>/<outlet>.json``
       → ``{"source": "ap_us", "articles": [{...}]}``

    Dialect is determined by the outlet folder/file-stem name via
    ``OUTLET_TO_DIALECT``.  Unrecognised outlet names are skipped silently.

    Exact-duplicate articles (same normalized text re-fetched under a different
    filename or even a different topic — a known upstream fetch bug, see
    ``get_news.ipynb``'s ``exclude_urls``) are dropped, keeping only the first
    occurrence. Without this, duplicated articles double-count their POS n-grams
    and skew the per-dialect frequency distribution toward whatever happened to
    get re-fetched.

    Returns:
        ``{"US": ["text1", ...], "UK": [...], "AUS": [...]}``
    """
    corpus: dict[str, list[str]] = defaultdict(list)
    seen_hashes: set[str] = set()
    n_dupes = 0

    for topic in sorted(os.listdir(articles_dir)):
        topic_path = os.path.join(articles_dir, topic)
        if not os.path.isdir(topic_path) or topic.startswith("_"):
            continue

        for entry in sorted(os.listdir(topic_path)):
            entry_path = os.path.join(topic_path, entry)

            # ── Layout ①: outlet is a sub-directory ────────────────────────
            if os.path.isdir(entry_path):
                dialect = OUTLET_TO_DIALECT.get(entry)
                if dialect is None:
                    continue
                for fname in os.listdir(entry_path):
                    if not fname.endswith(".json"):
                        continue
                    text = _load_individual_file(os.path.join(entry_path, fname))
                    if not text.strip():
                        continue
                    h = _norm_hash(text)
                    if h in seen_hashes:
                        n_dupes += 1
                        continue
                    seen_hashes.add(h)
                    corpus[dialect].append(text)

            # ── Layout ②: outlet is a single bundle .json file ─────────────
            elif entry.endswith(".json"):
                stem    = entry[:-5]   # strip ".json"
                dialect = OUTLET_TO_DIALECT.get(stem)
                if dialect is None:
                    continue
                for text in _load_bundle_file(entry_path):
                    h = _norm_hash(text)
                    if h in seen_hashes:
                        n_dupes += 1
                        continue
                    seen_hashes.add(h)
                    corpus[dialect].append(text)

    if n_dupes:
        print(f"load_corpus: skipped {n_dupes} duplicate-content article(s)")

    return dict(corpus)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DISTRIBUTION BUILDING
# ──────────────────────────────────────────────────────────────────────────────

def build_distributions(corpus: dict[str, list[str]],
                         nlp: Language) -> dict[str, Counter]:
    """
    For each dialect, count every POS n-gram across all of its texts.

    Returns:
        ``{"US": Counter({("NN","VB"): 142, ...}), "UK": ..., "AUS": ...}``
    """
    distributions: dict[str, Counter] = {}

    for dialect in DIALECTS:
        texts = corpus.get(dialect, [])
        print(f"  [{dialect}] tagging {len(texts)} articles …", flush=True)
        counts: Counter = Counter()

        for text in texts:
            counts.update(text_to_pos_ngrams(text, nlp))

        distributions[dialect] = counts
        print(f"         → {len(counts):,} unique n-grams | "
              f"{sum(counts.values()):,} total occurrences")

    return distributions


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — LAPLACE-SMOOTHED LOG-LIKELIHOOD
# ──────────────────────────────────────────────────────────────────────────────

def _log_likelihood(ngrams: list[tuple],
                    dialect_counts: Counter,
                    vocab_size: int) -> float:
    """
    Compute the summed log-likelihood of an n-gram sequence under one dialect
    model using Laplace (add-1) smoothing.

    For each n-gram *g*:

        P(g | dialect)  =  (C(g, dialect) + 1) / (N_dialect + V)

    where:
        C(g, dialect)  = raw count in training data for this dialect
        N_dialect      = total n-gram tokens seen for this dialect
        V              = global vocabulary size (unique n-grams across ALL
                         dialects) — ensures the distribution sums to ≤ 1
                         even for unseen n-grams

    Summing in log-space prevents floating-point underflow over long texts.
    """
    N = sum(dialect_counts.values())
    log_prob = 0.0
    for ngram in ngrams:
        smoothed = dialect_counts.get(ngram, 0) + 1
        log_prob += math.log(smoothed / (N + vocab_size))
    return log_prob


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CLASSIFIER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class DialectClassifier:
    """
    Bayesian POS n-gram dialect classifier for US / UK / AUS English.

    Usage::

        clf = DialectClassifier(nlp).fit(corpus)
        scores = clf.evaluate_dialect("The organisation finalised the programme.")
        # → {"US": -42.1, "UK": -38.7, "AUS": -40.3}  (higher = more probable)
    """

    def __init__(self, nlp: Language, n_sizes: list[int] = N_GRAM_SIZES):
        self.nlp      = nlp
        self.n_sizes  = n_sizes

        # Populated by .fit()
        self.distributions: dict[str, Counter] = {}
        self.log_priors: dict[str, float]      = {}
        self.vocab_size: int                   = 0

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, corpus: dict[str, list[str]]) -> "DialectClassifier":
        """
        Build smoothed n-gram frequency models from the labelled corpus.

        Args:
            corpus: ``{dialect_label: [raw_text, raw_text, ...]}``

        Returns:
            *self* — allows ``DialectClassifier(nlp).fit(corpus)`` chaining.
        """
        print("\nBuilding POS n-gram distributions …")
        self.distributions = build_distributions(corpus, self.nlp)

        # Empirical log-priors: P(dialect) = articles_in_dialect / total_articles
        total_docs = sum(len(texts) for texts in corpus.values())
        self.log_priors = {
            dialect: math.log(max(len(corpus.get(dialect, [])), 1) / total_docs)
            for dialect in DIALECTS
        }

        # Global vocabulary = union of every n-gram seen across all dialects.
        # Used as the smoothing denominator so probabilities stay comparable.
        all_ngrams: set = set()
        for counts in self.distributions.values():
            all_ngrams.update(counts.keys())
        self.vocab_size = len(all_ngrams)

        print(f"\nTraining complete.")
        print(f"  Global vocabulary : {self.vocab_size:,} unique n-grams")
        print(f"  Log-priors        : { {d: round(v,3) for d,v in self.log_priors.items()} }")
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def evaluate_dialect(self, text: str) -> dict[str, float]:
        """
        Classify a raw input text and return log-posterior scores per dialect.

        Applies Bayes' Theorem in log-space:

            log P(dialect | text)  ∝  log P(text | dialect) + log P(dialect)

        The same quote-masking and POS-tagging pipeline used during training
        is applied to *text* before scoring.

        Args:
            text: Raw string — e.g. an LLM-generated summary.

        Returns:
            ``{"US": float, "UK": float, "AUS": float}``
            Higher (less negative) values indicate higher probability.
            All values are in natural-log scale.

        Raises:
            RuntimeError: if ``.fit()`` has not been called yet.
        """
        if not self.distributions:
            raise RuntimeError("Classifier not trained — call .fit() first.")

        ngrams = text_to_pos_ngrams(text, self.nlp, self.n_sizes)

        if not ngrams:
            return {d: float("-inf") for d in DIALECTS}

        scores: dict[str, float] = {}
        for dialect in DIALECTS:
            ll  = _log_likelihood(ngrams, self.distributions[dialect], self.vocab_size)
            scores[dialect] = ll + self.log_priors.get(dialect, 0.0)

        return scores


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MODULE-LEVEL CONVENIENCE API
# ──────────────────────────────────────────────────────────────────────────────

# Singleton set by build_classifier(); used by the standalone evaluate_dialect()
_classifier: DialectClassifier | None = None


def build_classifier(articles_dir: str = ARTICLES_DIR) -> DialectClassifier:
    """
    Full end-to-end pipeline: load data → build NLP → train classifier.

    Sets the module-level ``_classifier`` so ``evaluate_dialect()`` works
    without explicitly holding a reference to the returned object.

    Args:
        articles_dir: Path to the root ``articles/`` folder.

    Returns:
        A trained :class:`DialectClassifier`.
    """
    global _classifier

    print("Loading spaCy model (en_core_web_sm) …")
    nlp = build_nlp_pipeline()

    print(f"\nLoading corpus from '{articles_dir}' …")
    corpus = load_corpus(articles_dir)

    for dialect in DIALECTS:
        n = len(corpus.get(dialect, []))
        print(f"  {dialect}: {n} articles")

    _classifier = DialectClassifier(nlp).fit(corpus)
    return _classifier


def evaluate_dialect(text: str) -> dict[str, float]:
    """
    Public one-shot API: classify *text* against the trained dialect models.

    This wraps :py:meth:`DialectClassifier.evaluate_dialect` and requires
    :func:`build_classifier` to have been called first.

    Args:
        text: Raw input string (e.g. an LLM-generated news summary).

    Returns:
        ``{"US": log_prob, "UK": log_prob, "AUS": log_prob}``
        Higher (less negative) = more probable dialect.

    Raises:
        RuntimeError: if ``build_classifier()`` has not been called.
    """
    if _classifier is None:
        raise RuntimeError(
            "No trained classifier found.  Call build_classifier() first."
        )
    return _classifier.evaluate_dialect(text)


def scores_to_probs(log_scores: dict[str, float]) -> dict[str, float]:
    """
    Convert a dict of log-posterior scores to normalised probabilities.

    Uses the log-sum-exp trick to avoid floating-point overflow/underflow:

        prob(d) = exp(score_d - max_score) / sum(exp(score_i - max_score))

    Args:
        log_scores: Output of :func:`evaluate_dialect` or
                    :py:meth:`DialectClassifier.evaluate_dialect`.

    Returns:
        ``{"US": float, "UK": float, "AUS": float}`` where values sum to 1.0.
    """
    max_score = max(log_scores.values())
    exp_scores = {d: math.exp(s - max_score) for d, s in log_scores.items()}
    total = sum(exp_scores.values())
    return {d: v / total for d, v in exp_scores.items()}


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SHARED DEMO TEXTS
# ──────────────────────────────────────────────────────────────────────────────
# Hoisted to module level (rather than living inside __main__) so other
# scripts — e.g. bert_dialect_classifier.py — can import the same probe
# sentences and compare predictions across models on identical inputs.

DEMO_TEXTS: dict[str, str] = {
    "US-leaning": (
        'The organization announced it will finalize its program next fall. '
        'Officials noted "we are optimistic about the outcome" and said they '
        'would prioritize labor reforms and defense spending.'
    ),
    "UK-leaning": (
        'The organisation announced it will finalise its programme next autumn. '
        'Officials noted "we are optimistic about the outcome" and said they '
        'would prioritise labour reforms and defence spending.'
    ),
    "AUS-leaning": (
        'The organisation said it will finalise the programme by arvo. '
        'Officials noted "she\'ll be right" and pledged to prioritise '
        'labour reforms and defence spending across the states.'
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — ENTRY POINT & DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Train ─────────────────────────────────────────────────────────────────
    classifier = build_classifier()

    print("\n" + "=" * 64)
    print("DIALECT EVALUATION DEMO")
    print("=" * 64)

    for label, text in DEMO_TEXTS.items():
        scores = evaluate_dialect(text)
        probs  = scores_to_probs(scores)
        predicted = max(scores, key=scores.get)

        print(f"\n[{label}]")
        print(f"  Text : {text[:90]}…")
        print(f"  {'Dialect':<8}  {'Log-posterior':>14}  {'Probability':>12}")
        print(f"  {'-'*8}  {'-'*14}  {'-'*12}")
        for dialect, score in sorted(scores.items(), key=lambda x: -x[1]):
            marker = " ◄" if dialect == predicted else ""
            print(f"  {dialect:<8}  {score:>14.4f}  {probs[dialect]:>11.2%}{marker}")
