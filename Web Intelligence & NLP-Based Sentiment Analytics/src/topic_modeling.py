"""
Web Intelligence & NLP
======================

Keyword and Topic Modeling Module.

This module provides reusable functions for:

- Word frequency analysis
- TF-IDF keyword extraction
- Document-level keyword extraction
- N-gram analysis
- LDA topic modeling
- NMF topic modeling
- Dominant topic assignment
- Topic summaries
- Topic × sentiment analysis
- Topic model persistence
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import joblib
import numpy as np
import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF,
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 0.95
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 500


# ============================================================
# TEXT VALIDATION
# ============================================================

def validate_text_column(
    df: pd.DataFrame,
    text_column: str = "processed_text",
) -> None:
    """
    Validate that the required text column exists.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    text_column : str
        Text column name.

    Raises
    ------
    ValueError
        If the column does not exist.
    """

    if text_column not in df.columns:

        raise ValueError(
            f"Required column '{text_column}' "
            "was not found."
        )


def prepare_text_series(
    df: pd.DataFrame,
    text_column: str = "processed_text",
) -> pd.Series:
    """
    Prepare a clean text Series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    text_column : str
        Text column.

    Returns
    -------
    pandas.Series
        Clean text series.
    """

    validate_text_column(
        df,
        text_column,
    )

    text_series = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return text_series


# ============================================================
# WORD FREQUENCY
# ============================================================

def word_frequency(
    text_series: Iterable[str],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate corpus-level word frequency.

    Parameters
    ----------
    text_series : iterable
        Collection of processed documents.

    top_n : int, optional
        Number of top words to return.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - keyword
        - frequency
    """

    counter = Counter()

    for text in text_series:

        if not isinstance(text, str):
            continue

        counter.update(
            text.split()
        )

    result = pd.DataFrame(
        counter.items(),
        columns=[
            "keyword",
            "frequency",
        ],
    )

    if result.empty:

        return pd.DataFrame(
            columns=[
                "keyword",
                "frequency",
            ]
        )

    result = (
        result
        .sort_values(
            "frequency",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if top_n is not None:

        result = result.head(
            top_n
        )

    return result


# ============================================================
# TF-IDF VECTORIZER
# ============================================================

def create_tfidf_vectorizer(
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer.

    Parameters
    ----------
    ngram_range : tuple
        N-gram range.

    min_df : int
        Minimum document frequency.

    max_df : float
        Maximum document frequency.

    Returns
    -------
    TfidfVectorizer
        Configured vectorizer.
    """

    return TfidfVectorizer(
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


# ============================================================
# TF-IDF KEYWORDS
# ============================================================

def extract_tfidf_keywords(
    text_series: Iterable[str],
    top_n: Optional[int] = None,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Extract corpus-level TF-IDF keywords.

    Parameters
    ----------
    text_series : iterable
        Processed documents.

    top_n : int, optional
        Number of keywords to return.

    ngram_range : tuple
        N-gram range.

    min_df : int
        Minimum document frequency.

    max_df : float
        Maximum document frequency.

    Returns
    -------
    tuple
        Keyword DataFrame and fitted TF-IDF vectorizer.
    """

    text_series = list(text_series)

    vectorizer = create_tfidf_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    matrix = vectorizer.fit_transform(
        text_series
    )

    features = np.array(
        vectorizer.get_feature_names_out()
    )

    mean_scores = np.asarray(
        matrix.mean(axis=0)
    ).ravel()

    result = pd.DataFrame(
        {
            "keyword": features,
            "mean_tfidf": mean_scores,
        }
    )

    result = (
        result
        .sort_values(
            "mean_tfidf",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if top_n is not None:

        result = result.head(
            top_n
        )

    return result, vectorizer


# ============================================================
# DOCUMENT-LEVEL TF-IDF KEYWORDS
# ============================================================

def document_keywords(
    text_series: Iterable[str],
    top_k: int = 8,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> tuple[list[list[tuple[str, float]]], TfidfVectorizer]:
    """
    Extract top TF-IDF keywords for every document.

    Parameters
    ----------
    text_series : iterable
        Processed text documents.

    top_k : int
        Number of keywords per document.

    vectorizer : TfidfVectorizer, optional
        Existing fitted vectorizer.

    Returns
    -------
    tuple
        List of document keywords and fitted vectorizer.
    """

    text_series = list(text_series)

    if vectorizer is None:

        vectorizer = create_tfidf_vectorizer()

        matrix = vectorizer.fit_transform(
            text_series
        )

    else:

        matrix = vectorizer.transform(
            text_series
        )

    features = np.array(
        vectorizer.get_feature_names_out()
    )

    all_keywords = []

    for row_index in range(
        matrix.shape[0]
    ):

        scores = (
            matrix[row_index]
            .toarray()
            .ravel()
        )

        top_indices = scores.argsort()[
            ::-1
        ][:top_k]

        keywords = [
            (
                features[index],
                round(
                    float(scores[index]),
                    4,
                ),
            )
            for index in top_indices
            if scores[index] > 0
        ]

        all_keywords.append(
            keywords
        )

    return (
        all_keywords,
        vectorizer,
    )


# ============================================================
# N-GRAM GENERATION
# ============================================================

def generate_ngrams(
    tokens: Iterable[str],
    n: int = 2,
) -> list[tuple[str, ...]]:
    """
    Generate n-grams from tokens.

    Parameters
    ----------
    tokens : iterable
        Input tokens.

    n : int
        N-gram size.

    Returns
    -------
    list of tuples
        Generated n-grams.
    """

    if n < 1:

        raise ValueError(
            "n must be >= 1."
        )

    tokens = list(tokens)

    return [
        tuple(
            tokens[index:index + n]
        )
        for index in range(
            len(tokens) - n + 1
        )
    ]


def ngram_frequency(
    text_series: Iterable[str],
    n: int = 2,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate n-gram frequency.

    Parameters
    ----------
    text_series : iterable
        Processed documents.

    n : int
        N-gram size.

    top_n : int, optional
        Number of results.

    Returns
    -------
    pandas.DataFrame
        N-gram frequency table.
    """

    counter = Counter()

    for text in text_series:

        if not isinstance(text, str):
            continue

        tokens = text.split()

        ngrams = generate_ngrams(
            tokens,
            n=n,
        )

        counter.update(
            ngrams
        )

    result = pd.DataFrame(
        [
            {
                "ngram": " ".join(
                    ngram
                ),
                "frequency": frequency,
            }
            for ngram, frequency
            in counter.items()
        ]
    )

    if result.empty:

        return pd.DataFrame(
            columns=[
                "ngram",
                "frequency",
            ]
        )

    result = (
        result
        .sort_values(
            "frequency",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if top_n is not None:

        result = result.head(
            top_n
        )

    return result


# ============================================================
# COUNT VECTOR
# ============================================================

def create_count_vectorizer(
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
) -> CountVectorizer:
    """
    Create a CountVectorizer for topic modeling.
    """

    return CountVectorizer(
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )


# ============================================================
# LDA MODEL
# ============================================================

def create_lda_model(
    n_topics: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = 50,
) -> LatentDirichletAllocation:
    """
    Create an LDA topic model.

    Parameters
    ----------
    n_topics : int
        Number of topics.

    random_state : int
        Reproducibility seed.

    max_iter : int
        Maximum iterations.

    Returns
    -------
    LatentDirichletAllocation
        Configured LDA model.
    """

    if n_topics < 2:

        raise ValueError(
            "n_topics must be at least 2."
        )

    return LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=max_iter,
    )


def fit_lda(
    text_series: Iterable[str],
    n_topics: int = 3,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = 50,
):
    """
    Fit an LDA model.

    Returns
    -------
    tuple
        LDA model, CountVectorizer, document-topic matrix.
    """

    text_series = list(text_series)

    vectorizer = create_count_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    matrix = vectorizer.fit_transform(
        text_series
    )

    model = create_lda_model(
        n_topics=n_topics,
        random_state=random_state,
        max_iter=max_iter,
    )

    document_topic_matrix = (
        model.fit_transform(matrix)
    )

    return (
        model,
        vectorizer,
        document_topic_matrix,
    )


# ============================================================
# NMF MODEL
# ============================================================

def create_nmf_model(
    n_topics: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> NMF:
    """
    Create an NMF topic model.
    """

    if n_topics < 2:

        raise ValueError(
            "n_topics must be at least 2."
        )

    return NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=max_iter,
    )


def fit_nmf(
    text_series: Iterable[str],
    n_topics: int = 3,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
):
    """
    Fit an NMF topic model.

    Returns
    -------
    tuple
        NMF model, TF-IDF vectorizer, document-topic matrix.
    """

    text_series = list(text_series)

    vectorizer = create_tfidf_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    matrix = vectorizer.fit_transform(
        text_series
    )

    model = create_nmf_model(
        n_topics=n_topics,
        random_state=random_state,
        max_iter=max_iter,
    )

    document_topic_matrix = (
        model.fit_transform(matrix)
    )

    return (
        model,
        vectorizer,
        document_topic_matrix,
    )


# ============================================================
# TOPIC WORDS
# ============================================================

def get_topic_words(
    model,
    feature_names: Iterable[str],
    top_words: int = 10,
) -> pd.DataFrame:
    """
    Extract the strongest words for each topic.

    Parameters
    ----------
    model
        Fitted LDA or NMF model.

    feature_names : iterable
        Vectorizer feature names.

    top_words : int
        Number of words per topic.

    Returns
    -------
    pandas.DataFrame
        Topic and top-word table.
    """

    feature_names = np.array(
        list(feature_names)
    )

    rows = []

    for topic_index, topic in enumerate(
        model.components_
    ):

        top_indices = topic.argsort()[
            ::-1
        ][:top_words]

        words = [
            feature_names[index]
            for index in top_indices
        ]

        rows.append(
            {
                "topic": topic_index + 1,
                "top_words": ", ".join(words),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DOMINANT TOPIC
# ============================================================

def assign_dominant_topic(
    document_topic_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Assign the dominant topic to each document.

    Parameters
    ----------
    document_topic_matrix : ndarray
        Document-topic weight matrix.

    Returns
    -------
    pandas.DataFrame
        Dominant topic and topic score.
    """

    document_topic_matrix = np.asarray(
        document_topic_matrix
    )

    dominant_indices = (
        document_topic_matrix.argmax(
            axis=1
        )
    )

    dominant_scores = (
        document_topic_matrix.max(
            axis=1
        )
    )

    return pd.DataFrame(
        {
            "dominant_topic": (
                dominant_indices + 1
            ),
            "topic_score": (
                dominant_scores
            ),
        }
    )


# ============================================================
# ADD TOPIC ASSIGNMENTS
# ============================================================

def add_topic_assignments(
    df: pd.DataFrame,
    document_topic_matrix: np.ndarray,
    topic_prefix: str = "lda",
) -> pd.DataFrame:
    """
    Add dominant topic and topic score to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    document_topic_matrix : ndarray
        Document-topic matrix.

    topic_prefix : str
        Prefix such as 'lda' or 'nmf'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with topic assignments.
    """

    if len(df) != len(
        document_topic_matrix
    ):

        raise ValueError(
            "Number of documents does not match "
            "document-topic matrix."
        )

    assignments = assign_dominant_topic(
        document_topic_matrix
    )

    result = df.copy()

    result[
        f"{topic_prefix}_dominant_topic"
    ] = assignments[
        "dominant_topic"
    ].values

    result[
        f"{topic_prefix}_topic_score"
    ] = assignments[
        "topic_score"
    ].values

    return result


# ============================================================
# TOPIC DISTRIBUTION
# ============================================================

def topic_distribution(
    df: pd.DataFrame,
    topic_column: str = "lda_dominant_topic",
) -> pd.DataFrame:
    """
    Calculate topic distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic-assigned DataFrame.

    topic_column : str
        Dominant topic column.

    Returns
    -------
    pandas.DataFrame
        Topic counts and percentages.
    """

    if topic_column not in df.columns:

        raise ValueError(
            f"Column '{topic_column}' "
            "was not found."
        )

    result = (
        df[topic_column]
        .value_counts()
        .sort_index()
        .rename_axis("topic")
        .reset_index(
            name="document_count"
        )
    )

    result["percentage"] = (
        result["document_count"]
        / len(df)
        * 100
    ).round(2)

    return result


# ============================================================
# TOPIC × SENTIMENT
# ============================================================

def topic_sentiment_analysis(
    df: pd.DataFrame,
    topic_column: str = "lda_dominant_topic",
    sentiment_column: str = "sentiment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate topic × sentiment counts and percentages.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic and sentiment dataset.

    topic_column : str
        Topic column.

    sentiment_column : str
        Sentiment column.

    Returns
    -------
    tuple
        Counts and row-normalized percentages.
    """

    required = [
        topic_column,
        sentiment_column,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    counts = pd.crosstab(
        df[topic_column],
        df[sentiment_column],
    )

    percentages = pd.crosstab(
        df[topic_column],
        df[sentiment_column],
        normalize="index",
    ).mul(
        100
    ).round(2)

    return (
        counts,
        percentages,
    )


# ============================================================
# TOPIC SUMMARY
# ============================================================

def create_topic_summary(
    df: pd.DataFrame,
    topic_words: Optional[pd.DataFrame] = None,
    topic_column: str = "lda_dominant_topic",
    score_column: str = "lda_topic_score",
) -> pd.DataFrame:
    """
    Create a reusable topic summary table.

    Parameters
    ----------
    df : pandas.DataFrame
        Topic-assigned DataFrame.

    topic_words : pandas.DataFrame, optional
        Topic → top words table.

    topic_column : str
        Dominant topic column.

    score_column : str
        Topic score column.

    Returns
    -------
    pandas.DataFrame
        Topic summary.
    """

    if topic_column not in df.columns:

        raise ValueError(
            f"Column '{topic_column}' "
            "was not found."
        )

    aggregation = {
        "document_count": (
            topic_column,
            "count",
        )
    }

    grouped = (
        df.groupby(topic_column)
        .agg(
            document_count=(
                topic_column,
                "count",
            )
        )
        .reset_index()
        .rename(
            columns={
                topic_column: "topic"
            }
        )
    )

    if score_column in df.columns:

        score_summary = (
            df.groupby(topic_column)[
                score_column
            ]
            .mean()
            .reset_index()
            .rename(
                columns={
                    topic_column: "topic",
                    score_column: (
                        "avg_topic_score"
                    ),
                }
            )
        )

        grouped = grouped.merge(
            score_summary,
            on="topic",
            how="left",
        )

    grouped["document_percentage"] = (
        grouped["document_count"]
        / len(df)
        * 100
    ).round(2)

    if topic_words is not None:

        grouped = grouped.merge(
            topic_words,
            on="topic",
            how="left",
        )

    return grouped


# ============================================================
# TOPIC MODELING PIPELINE
# ============================================================

def run_topic_modeling_pipeline(
    df: pd.DataFrame,
    text_column: str = "processed_text",
    n_topics: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_words: int = 10,
) -> dict:
    """
    Run the complete keyword and topic-modeling workflow.

    Workflow
    --------
    Processed Text
        ↓
    Word Frequency
        ↓
    TF-IDF Keywords
        ↓
    N-Grams
        ↓
    LDA
        ↓
    NMF
        ↓
    Topic Assignment
        ↓
    Topic Summary
        ↓
    Topic × Sentiment

    Parameters
    ----------
    df : pandas.DataFrame
        NLP-ready DataFrame.

    text_column : str
        Processed text column.

    n_topics : int
        Number of latent topics.

    random_state : int
        Reproducibility seed.

    top_words : int
        Number of topic words.

    Returns
    -------
    dict
        All pipeline outputs.
    """

    text_series = prepare_text_series(
        df,
        text_column,
    )

    # --------------------------------------------------------
    # WORD FREQUENCY
    # --------------------------------------------------------

    word_frequency_df = word_frequency(
        text_series
    )

    # --------------------------------------------------------
    # TF-IDF KEYWORDS
    # --------------------------------------------------------

    tfidf_keywords_df, tfidf_vectorizer = (
        extract_tfidf_keywords(
            text_series
        )
    )

    # --------------------------------------------------------
    # DOCUMENT KEYWORDS
    # --------------------------------------------------------

    document_keyword_list, _ = (
        document_keywords(
            text_series,
            vectorizer=tfidf_vectorizer,
        )
    )

    # --------------------------------------------------------
    # N-GRAMS
    # --------------------------------------------------------

    bigrams_df = ngram_frequency(
        text_series,
        n=2,
    )

    trigrams_df = ngram_frequency(
        text_series,
        n=3,
    )

    # --------------------------------------------------------
    # LDA
    # --------------------------------------------------------

    (
        lda_model,
        lda_vectorizer,
        lda_document_topic,
    ) = fit_lda(
        text_series,
        n_topics=n_topics,
        random_state=random_state,
    )

    lda_features = (
        lda_vectorizer
        .get_feature_names_out()
    )

    lda_topics = get_topic_words(
        model=lda_model,
        feature_names=lda_features,
        top_words=top_words,
    )

    # --------------------------------------------------------
    # NMF
    # --------------------------------------------------------

    (
        nmf_model,
        nmf_vectorizer,
        nmf_document_topic,
    ) = fit_nmf(
        text_series,
        n_topics=n_topics,
        random_state=random_state,
    )

    nmf_features = (
        nmf_vectorizer
        .get_feature_names_out()
    )

    nmf_topics = get_topic_words(
        model=nmf_model,
        feature_names=nmf_features,
        top_words=top_words,
    )

    # --------------------------------------------------------
    # ASSIGN TOPICS
    # --------------------------------------------------------

    result_df = add_topic_assignments(
        df,
        lda_document_topic,
        topic_prefix="lda",
    )

    result_df = add_topic_assignments(
        result_df,
        nmf_document_topic,
        topic_prefix="nmf",
    )

    # --------------------------------------------------------
    # TOPIC DISTRIBUTION
    # --------------------------------------------------------

    lda_distribution = topic_distribution(
        result_df,
        topic_column="lda_dominant_topic",
    )

    # --------------------------------------------------------
    # TOPIC SUMMARY
    # --------------------------------------------------------

    topic_summary = create_topic_summary(
        result_df,
        topic_words=lda_topics,
        topic_column="lda_dominant_topic",
        score_column="lda_topic_score",
    )

    # --------------------------------------------------------
    # TOPIC × SENTIMENT
    # --------------------------------------------------------

    if "sentiment" in result_df.columns:

        (
            topic_sentiment_counts,
            topic_sentiment_percentages,
        ) = topic_sentiment_analysis(
            result_df,
            topic_column="lda_dominant_topic",
            sentiment_column="sentiment",
        )

    else:

        topic_sentiment_counts = None
        topic_sentiment_percentages = None

    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {
        "data": result_df,
        "word_frequency": word_frequency_df,
        "tfidf_keywords": tfidf_keywords_df,
        "tfidf_vectorizer": tfidf_vectorizer,
        "document_keywords": document_keyword_list,
        "bigrams": bigrams_df,
        "trigrams": trigrams_df,
        "lda_model": lda_model,
        "lda_vectorizer": lda_vectorizer,
        "lda_document_topic": lda_document_topic,
        "lda_topics": lda_topics,
        "lda_distribution": lda_distribution,
        "nmf_model": nmf_model,
        "nmf_vectorizer": nmf_vectorizer,
        "nmf_document_topic": nmf_document_topic,
        "nmf_topics": nmf_topics,
        "topic_summary": topic_summary,
        "topic_sentiment_counts": (
            topic_sentiment_counts
        ),
        "topic_sentiment_percentages": (
            topic_sentiment_percentages
        ),
    }


# ============================================================
# MODEL PERSISTENCE
# ============================================================

def save_topic_model(
    model,
    output_path: str | Path,
) -> Path:
    """
    Save a topic model using joblib.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )

    return output_path


def load_topic_model(
    model_path: str | Path,
):
    """
    Load a saved topic model.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Topic model not found: "
            f"{model_path}"
        )

    return joblib.load(
        model_path
    )


def save_vectorizer(
    vectorizer,
    output_path: str | Path,
) -> Path:
    """
    Save a fitted vectorizer.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        vectorizer,
        output_path,
    )

    return output_path


def load_vectorizer(
    vectorizer_path: str | Path,
):
    """
    Load a saved vectorizer.
    """

    vectorizer_path = Path(
        vectorizer_path
    )

    if not vectorizer_path.exists():

        raise FileNotFoundError(
            f"Vectorizer not found: "
            f"{vectorizer_path}"
        )

    return joblib.load(
        vectorizer_path
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    sample_documents = [
        "machine learning artificial intelligence",
        "deep learning neural network model",
        "data science machine learning",
        "artificial intelligence neural network",
        "python data analysis visualization",
        "data science analytics python",
    ]

    sample_df = pd.DataFrame(
        {
            "processed_text": sample_documents,
            "sentiment": [
                "Positive",
                "Positive",
                "Neutral",
                "Positive",
                "Neutral",
                "Positive",
            ],
        }
    )

    results = run_topic_modeling_pipeline(
        sample_df,
        text_column="processed_text",
        n_topics=2,
        random_state=42,
        top_words=5,
    )

    print("=" * 70)
    print("WORD FREQUENCY")
    print("=" * 70)

    print(
        results["word_frequency"]
        .head()
    )

    print()
    print("=" * 70)
    print("TF-IDF KEYWORDS")
    print("=" * 70)

    print(
        results["tfidf_keywords"]
        .head()
    )

    print()
    print("=" * 70)
    print("LDA TOPICS")
    print("=" * 70)

    print(
        results["lda_topics"]
    )

    print()
    print("=" * 70)
    print("NMF TOPICS")
    print("=" * 70)

    print(
        results["nmf_topics"]
    )

    print()
    print(
        "Topic modeling module loaded successfully."
    )