"""
Web Intelligence & NLP
======================

Text preprocessing module.

This module provides reusable functions for:
- Text cleaning
- Lowercasing
- URL and HTML removal
- Punctuation/noise removal
- Tokenization
- Stopword removal
- Lemmatization
- Stemming
- N-gram generation
- Text statistics
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Optional

import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ============================================================
# NLTK RESOURCE SETUP
# ============================================================

def download_nltk_resources() -> None:
    """
    Download required NLTK resources if they are not available.

    Required resources:
    - punkt
    - punkt_tab
    - stopwords
    - wordnet
    - omw-1.4
    """

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, resource_name in resources:

        try:
            nltk.data.find(resource_path)

        except LookupError:
            nltk.download(
                resource_name,
                quiet=True,
            )


# Initialize required NLTK resources.
download_nltk_resources()


# ============================================================
# NLP OBJECTS
# ============================================================

LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

STOP_WORDS = set(
    stopwords.words("english")
)


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(
    text: object,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """
    Clean raw text for NLP processing.

    Parameters
    ----------
    text : object
        Input text.

    lowercase : bool, default=True
        Convert text to lowercase.

    remove_urls : bool, default=True
        Remove HTTP/HTTPS URLs.

    remove_html : bool, default=True
        Remove HTML tags.

    remove_punctuation : bool, default=True
        Remove punctuation characters.

    Returns
    -------
    str
        Cleaned text.
    """

    if text is None:
        return ""

    text = str(text)

    # Normalize whitespace.
    text = text.strip()

    # Remove URLs.
    if remove_urls:
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # Remove HTML tags.
    if remove_html:
        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

    # Lowercase.
    if lowercase:
        text = text.lower()

    # Remove punctuation.
    if remove_punctuation:
        text = text.translate(
            str.maketrans(
                "",
                "",
                string.punctuation,
            )
        )

    # Keep alphabetic characters and whitespace.
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize multiple spaces.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(
    text: str,
) -> list[str]:
    """
    Tokenize cleaned text into individual words.

    Parameters
    ----------
    text : str
        Cleaned text.

    Returns
    -------
    list[str]
        List of tokens.
    """

    if not text:
        return []

    return word_tokenize(text)


# ============================================================
# STOPWORD REMOVAL
# ============================================================

def remove_stopwords(
    tokens: Iterable[str],
    stop_words: Optional[set[str]] = None,
) -> list[str]:
    """
    Remove English stopwords from tokens.

    Parameters
    ----------
    tokens : iterable
        Input token sequence.

    stop_words : set, optional
        Custom stopword set.

    Returns
    -------
    list[str]
        Tokens after stopword removal.
    """

    active_stop_words = (
        stop_words
        if stop_words is not None
        else STOP_WORDS
    )

    return [
        token
        for token in tokens
        if token.lower() not in active_stop_words
    ]


# ============================================================
# LEMMATIZATION
# ============================================================

def lemmatize_tokens(
    tokens: Iterable[str],
) -> list[str]:
    """
    Lemmatize tokens using WordNet.

    Parameters
    ----------
    tokens : iterable
        Input tokens.

    Returns
    -------
    list[str]
        Lemmatized tokens.
    """

    return [
        LEMMATIZER.lemmatize(
            token
        )
        for token in tokens
    ]


# ============================================================
# STEMMING
# ============================================================

def stem_tokens(
    tokens: Iterable[str],
) -> list[str]:
    """
    Stem tokens using Porter Stemmer.

    Stemming is provided as an optional alternative to
    lemmatization.

    Parameters
    ----------
    tokens : iterable
        Input tokens.

    Returns
    -------
    list[str]
        Stemmed tokens.
    """

    return [
        STEMMER.stem(token)
        for token in tokens
    ]


# ============================================================
# COMPLETE NLP PIPELINE
# ============================================================

def preprocess_text(
    text: object,
    remove_stop_words: bool = True,
    lemmatize: bool = True,
) -> str:
    """
    Execute the complete NLP preprocessing pipeline.

    Workflow
    --------
    Raw Text
        ↓
    Cleaning
        ↓
    Lowercase
        ↓
    Tokenization
        ↓
    Stopword Removal
        ↓
    Lemmatization
        ↓
    Processed Text

    Parameters
    ----------
    text : object
        Raw input text.

    remove_stop_words : bool, default=True
        Remove English stopwords.

    lemmatize : bool, default=True
        Apply WordNet lemmatization.

    Returns
    -------
    str
        NLP-ready processed text.
    """

    cleaned = clean_text(text)

    if not cleaned:
        return ""

    tokens = tokenize_text(cleaned)

    if remove_stop_words:
        tokens = remove_stopwords(tokens)

    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


# ============================================================
# STEMMING PIPELINE
# ============================================================

def preprocess_with_stemming(
    text: object,
    remove_stop_words: bool = True,
) -> str:
    """
    Alternative preprocessing pipeline using stemming.

    Parameters
    ----------
    text : object
        Raw input text.

    remove_stop_words : bool, default=True
        Remove English stopwords.

    Returns
    -------
    str
        Stemmed NLP-ready text.
    """

    cleaned = clean_text(text)

    if not cleaned:
        return ""

    tokens = tokenize_text(cleaned)

    if remove_stop_words:
        tokens = remove_stopwords(tokens)

    tokens = stem_tokens(tokens)

    return " ".join(tokens)


# ============================================================
# DATAFRAME PREPROCESSING
# ============================================================

def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "quote_text",
    output_column: str = "processed_text",
) -> pd.DataFrame:
    """
    Apply NLP preprocessing to a DataFrame text column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    text_column : str, default="quote_text"
        Name of the raw text column.

    output_column : str, default="processed_text"
        Name of the processed text column.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing processed text.

    Raises
    ------
    ValueError
        If the text column does not exist.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Required text column '{text_column}' "
            "was not found."
        )

    result = df.copy()

    result["original_text"] = (
        result[text_column]
        .fillna("")
        .astype(str)
    )

    result[output_column] = (
        result[text_column]
        .fillna("")
        .apply(preprocess_text)
    )

    return result


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_statistics(
    text: object,
) -> dict:
    """
    Calculate basic statistics for a text document.

    Parameters
    ----------
    text : object
        Input text.

    Returns
    -------
    dict
        Character count, word count, and unique word count.
    """

    text = "" if text is None else str(text)

    words = text.split()

    return {
        "character_count": len(text),
        "word_count": len(words),
        "unique_word_count": len(set(words)),
    }


def add_text_statistics(
    df: pd.DataFrame,
    text_column: str = "processed_text",
) -> pd.DataFrame:
    """
    Add text statistics to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    text_column : str, default="processed_text"
        Text column to analyze.

    Returns
    -------
    pandas.DataFrame
        DataFrame with text statistics.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found."
        )

    result = df.copy()

    statistics = result[text_column].apply(
        text_statistics
    )

    statistics_df = pd.DataFrame(
        statistics.tolist(),
        index=result.index,
    )

    return pd.concat(
        [result, statistics_df],
        axis=1,
    )


# ============================================================
# WORD FREQUENCY
# ============================================================

def get_word_frequency(
    text_series: Iterable[str],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate corpus-level word frequency.

    Parameters
    ----------
    text_series : iterable
        Collection of processed text documents.

    top_n : int, optional
        Return only the top N words.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - word
        - frequency
    """

    counter = Counter()

    for text in text_series:

        if not isinstance(text, str):
            continue

        counter.update(
            text.split()
        )

    frequency_df = (
        pd.DataFrame(
            counter.items(),
            columns=[
                "word",
                "frequency",
            ],
        )
        .sort_values(
            "frequency",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if top_n is not None:
        frequency_df = frequency_df.head(
            top_n
        )

    return frequency_df


# ============================================================
# N-GRAM GENERATION
# ============================================================

def generate_ngrams(
    tokens: Iterable[str],
    n: int = 2,
) -> list[tuple]:
    """
    Generate n-grams from a token sequence.

    Parameters
    ----------
    tokens : iterable
        Input tokens.

    n : int, default=2
        N-gram size.

    Returns
    -------
    list[tuple]
        Generated n-grams.

    Raises
    ------
    ValueError
        If n is less than 1.
    """

    if n < 1:
        raise ValueError(
            "n must be greater than or equal to 1."
        )

    tokens = list(tokens)

    return [
        tuple(tokens[i:i + n])
        for i in range(
            len(tokens) - n + 1
        )
    ]


def get_ngram_frequency(
    text_series: Iterable[str],
    n: int = 2,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate corpus-level n-gram frequency.

    Parameters
    ----------
    text_series : iterable
        Collection of processed text documents.

    n : int, default=2
        N-gram size.

    top_n : int, optional
        Return only top N n-grams.

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

        counter.update(ngrams)

    ngram_df = pd.DataFrame(
        [
            {
                "ngram": " ".join(ngram),
                "frequency": frequency,
            }
            for ngram, frequency
            in counter.items()
        ]
    )

    if ngram_df.empty:
        return pd.DataFrame(
            columns=[
                "ngram",
                "frequency",
            ]
        )

    ngram_df = (
        ngram_df
        .sort_values(
            "frequency",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if top_n is not None:
        ngram_df = ngram_df.head(
            top_n
        )

    return ngram_df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_processed_text(
    df: pd.DataFrame,
    text_column: str = "processed_text",
) -> dict:
    """
    Validate an NLP-processed DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed DataFrame.

    text_column : str
        Processed text column.

    Returns
    -------
    dict
        Validation results.
    """

    column_exists = (
        text_column in df.columns
    )

    if not column_exists:

        return {
            "is_valid": False,
            "row_count": len(df),
            "column_exists": False,
            "empty_texts": None,
        }

    empty_texts = int(
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    return {
        "is_valid": (
            not df.empty
            and empty_texts == 0
        ),
        "row_count": len(df),
        "column_exists": True,
        "empty_texts": empty_texts,
    }


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save processed NLP data to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed DataFrame.

    output_path : str or Path
        Destination path.

    Returns
    -------
    pathlib.Path
        Saved file path.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


# ============================================================
# END-TO-END PREPROCESSING PIPELINE
# ============================================================

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    text_column: str = "quote_text",
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Workflow
    --------
    Raw Text
        ↓
    Cleaning
        ↓
    Tokenization
        ↓
    Stopword Removal
        ↓
    Lemmatization
        ↓
    Text Statistics
        ↓
    Validation
        ↓
    Optional CSV Export

    Parameters
    ----------
    df : pandas.DataFrame
        Input raw/cleaned DataFrame.

    text_column : str, default="quote_text"
        Source text column.

    output_path : str or Path, optional
        Optional CSV output path.

    Returns
    -------
    pandas.DataFrame
        NLP-ready DataFrame.

    Raises
    ------
    ValueError
        If validation fails.
    """

    result = preprocess_dataframe(
        df=df,
        text_column=text_column,
        output_column="processed_text",
    )

    result = add_text_statistics(
        result,
        text_column="processed_text",
    )

    validation = validate_processed_text(
        result,
        text_column="processed_text",
    )

    if not validation["is_valid"]:
        raise ValueError(
            "Processed text validation failed: "
            f"{validation}"
        )

    if output_path is not None:

        saved_path = save_processed_data(
            result,
            output_path,
        )

        print(
            f"Processed dataset saved to: "
            f"{saved_path}"
        )

    return result


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    sample_text = (
        "This Product is AMAZING!!! "
        "Visit https://example.com today."
    )

    print("Original:")
    print(sample_text)

    print("\\nCleaned:")
    print(clean_text(sample_text))

    print("\\nProcessed:")
    print(preprocess_text(sample_text))

    print("\\nStemmed:")
    print(
        preprocess_with_stemming(
            sample_text
        )
    )

    print("\\nStatistics:")
    print(
        text_statistics(
            preprocess_text(sample_text)
        )
    )