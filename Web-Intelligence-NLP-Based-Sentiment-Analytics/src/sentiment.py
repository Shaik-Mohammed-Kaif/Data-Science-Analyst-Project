"""
Web Intelligence & NLP
======================

Sentiment analysis module.

This module provides reusable functions for:

- VADER sentiment scoring
- Weak / pseudo-label generation
- Train-test splitting
- TF-IDF vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Model evaluation
- Model comparison
- Best-model selection
- Sentiment prediction
- Prediction scoring
- Model persistence

Important
---------
The current project uses VADER-generated labels because the
Quotes to Scrape dataset does not contain human-annotated
sentiment labels.

Therefore, model evaluation represents agreement with
weak/pseudo labels rather than true ground-truth accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import nltk
import numpy as np
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# CONFIGURATION
# ============================================================

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42

DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 0.95


# ============================================================
# NLTK RESOURCE
# ============================================================

def download_vader_resource() -> None:
    """
    Download the VADER lexicon if it is unavailable.
    """

    try:
        nltk.data.find(
            "sentiment/vader_lexicon"
        )

    except LookupError:

        nltk.download(
            "vader_lexicon",
            quiet=True,
        )


download_vader_resource()


# ============================================================
# SENTIMENT ANALYZER
# ============================================================

ANALYZER = SentimentIntensityAnalyzer()


# ============================================================
# VADER SENTIMENT SCORES
# ============================================================

def vader_scores(
    text: object,
) -> dict:
    """
    Calculate VADER sentiment scores.

    Parameters
    ----------
    text : object
        Input text.

    Returns
    -------
    dict
        VADER sentiment components:

        - negative_score
        - neutral_score
        - positive_score
        - compound_score
    """

    if text is None:
        text = ""

    text = str(text)

    scores = ANALYZER.polarity_scores(
        text
    )

    return {
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "positive_score": scores["pos"],
        "compound_score": scores["compound"],
    }


# ============================================================
# VADER SENTIMENT LABEL
# ============================================================

def vader_label(
    compound_score: float,
    positive_threshold: float = POSITIVE_THRESHOLD,
    negative_threshold: float = NEGATIVE_THRESHOLD,
) -> str:
    """
    Convert a VADER compound score into a sentiment label.

    Parameters
    ----------
    compound_score : float
        VADER compound score.

    positive_threshold : float
        Score at or above this value becomes Positive.

    negative_threshold : float
        Score at or below this value becomes Negative.

    Returns
    -------
    str
        Positive, Neutral, or Negative.
    """

    if compound_score >= positive_threshold:
        return "Positive"

    if compound_score <= negative_threshold:
        return "Negative"

    return "Neutral"


# ============================================================
# ADD VADER SENTIMENT
# ============================================================

def add_vader_sentiment(
    df: pd.DataFrame,
    text_column: str = "processed_text",
) -> pd.DataFrame:
    """
    Add VADER scores and weak sentiment labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    text_column : str
        Text column to analyze.

    Returns
    -------
    pandas.DataFrame
        DataFrame with VADER scores and sentiment labels.

    Raises
    ------
    ValueError
        If the text column does not exist.
    """

    if text_column not in df.columns:

        raise ValueError(
            f"Required column '{text_column}' "
            "was not found."
        )

    result = df.copy()

    score_df = (
        result[text_column]
        .fillna("")
        .apply(vader_scores)
        .apply(pd.Series)
    )

    result = pd.concat(
        [
            result.reset_index(drop=True),
            score_df.reset_index(drop=True),
        ],
        axis=1,
    )

    result["sentiment"] = result[
        "compound_score"
    ].apply(
        vader_label
    )

    return result


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    df: pd.DataFrame,
    text_column: str = "processed_text",
    target_column: str = "sentiment",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Split text data into training and testing sets.

    Stratification is used when every class has enough
    observations for a stratified split.

    Parameters
    ----------
    df : pandas.DataFrame
        Input labeled dataset.

    text_column : str
        Text column.

    target_column : str
        Sentiment label column.

    test_size : float
        Fraction used for testing.

    random_state : int
        Reproducibility seed.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    required_columns = [
        text_column,
        target_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    X = (
        df[text_column]
        .fillna("")
        .astype(str)
    )

    y = (
        df[target_column]
        .astype(str)
    )

    class_counts = y.value_counts()

    use_stratify = (
        len(class_counts) > 1
        and class_counts.min() >= 2
    )

    stratify = y if use_stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


# ============================================================
# TF-IDF VECTORIZE
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
        Range of n-grams.

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
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


def fit_transform_tfidf(
    X_train,
    X_test,
    vectorizer: Optional[TfidfVectorizer] = None,
):
    """
    Fit TF-IDF only on training data and transform
    both training and testing data.

    This prevents data leakage from the test set.

    Parameters
    ----------
    X_train : iterable
        Training text.

    X_test : iterable
        Testing text.

    vectorizer : TfidfVectorizer, optional
        Existing vectorizer.

    Returns
    -------
    tuple
        vectorizer, X_train_tfidf, X_test_tfidf
    """

    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    return (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_models(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """
    Create the baseline sentiment classification models.

    Models
    ------
    Logistic Regression
    Multinomial Naive Bayes
    Linear SVM

    Returns
    -------
    dict
        Dictionary of model name → estimator.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),

        "Multinomial Naive Bayes": MultinomialNB(),

        "Linear SVM": LinearSVC(
            random_state=random_state,
        ),
    }


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    models: dict,
    X_train,
    y_train,
) -> dict:
    """
    Train all supplied models.

    Parameters
    ----------
    models : dict
        Model dictionary.

    X_train
        TF-IDF training matrix.

    y_train
        Training labels.

    Returns
    -------
    dict
        Trained models.
    """

    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        trained_models[name] = model

    return trained_models


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
) -> dict:
    """
    Evaluate one trained model.

    Metrics
    -------
    Accuracy
    Precision
    Recall
    F1-score

    Parameters
    ----------
    model
        Trained classifier.

    X_test
        Test feature matrix.

    y_test
        Test labels.

    Returns
    -------
    dict
        Evaluation metrics and predictions.
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "Negative",
            "Neutral",
            "Positive",
        ],
    )

    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
    }


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_models(
    models: dict,
    X_test,
    y_test,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate multiple trained models.

    Parameters
    ----------
    models : dict
        Trained models.

    X_test
        Test feature matrix.

    y_test
        Test labels.

    Returns
    -------
    tuple
        Comparison DataFrame and detailed results dictionary.
    """

    results = []
    detailed_results = {}

    for name, model in models.items():

        evaluation = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        detailed_results[name] = evaluation

        results.append(
            {
                "model": name,
                "accuracy": evaluation["accuracy"],
                "precision_weighted": evaluation[
                    "precision_weighted"
                ],
                "recall_weighted": evaluation[
                    "recall_weighted"
                ],
                "f1_weighted": evaluation[
                    "f1_weighted"
                ],
            }
        )

    comparison_df = (
        pd.DataFrame(results)
        .sort_values(
            "f1_weighted",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        comparison_df,
        detailed_results,
    )


# ============================================================
# BEST MODEL
# ============================================================

def select_best_model(
    models: dict,
    comparison_df: pd.DataFrame,
    metric: str = "f1_weighted",
):
    """
    Select the best model based on a chosen metric.

    Parameters
    ----------
    models : dict
        Trained models.

    comparison_df : pandas.DataFrame
        Model comparison table.

    metric : str
        Metric used for selection.

    Returns
    -------
    tuple
        Best model name and trained estimator.
    """

    if comparison_df.empty:
        raise ValueError(
            "Model comparison table is empty."
        )

    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found."
        )

    best_name = comparison_df.iloc[0]["model"]

    return (
        best_name,
        models[best_name],
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(
    model,
    vectorizer: TfidfVectorizer,
    texts,
) -> np.ndarray:
    """
    Predict sentiment labels for new text.

    Parameters
    ----------
    model
        Trained sentiment model.

    vectorizer
        Fitted TF-IDF vectorizer.

    texts : iterable
        Input text documents.

    Returns
    -------
    numpy.ndarray
        Predicted sentiment labels.
    """

    cleaned_texts = [
        "" if text is None else str(text)
        for text in texts
    ]

    matrix = vectorizer.transform(
        cleaned_texts
    )

    return model.predict(
        matrix
    )


# ============================================================
# PREDICTION SCORE
# ============================================================

def prediction_scores(
    model,
    vectorizer: TfidfVectorizer,
    texts,
) -> np.ndarray:
    """
    Generate model prediction scores.

    For probabilistic models, the highest class probability
    is returned.

    For LinearSVC, the highest decision-function value is
    returned as a confidence-like score.

    Important
    ---------
    LinearSVC decision scores are NOT calibrated probabilities.
    """

    cleaned_texts = [
        "" if text is None else str(text)
        for text in texts
    ]

    matrix = vectorizer.transform(
        cleaned_texts
    )

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = model.predict_proba(
            matrix
        )

        return probabilities.max(
            axis=1
        )

    if hasattr(
        model,
        "decision_function",
    ):

        decision_values = model.decision_function(
            matrix
        )

        if decision_values.ndim == 1:

            # Convert binary decision values to
            # a positive magnitude score.
            return np.abs(
                decision_values
            )

        return decision_values.max(
            axis=1
        )

    return np.full(
        shape=len(cleaned_texts),
        fill_value=np.nan,
    )


# ============================================================
# ADD MODEL PREDICTIONS TO DATAFRAME
# ============================================================

def add_predictions(
    df: pd.DataFrame,
    model,
    vectorizer: TfidfVectorizer,
    text_column: str = "processed_text",
) -> pd.DataFrame:
    """
    Add model predictions and prediction scores to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    model
        Trained sentiment model.

    vectorizer
        Fitted TF-IDF vectorizer.

    text_column : str
        Text column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with prediction fields.
    """

    if text_column not in df.columns:

        raise ValueError(
            f"Column '{text_column}' was not found."
        )

    result = df.copy()

    result["predicted_sentiment"] = (
        predict_sentiment(
            model=model,
            vectorizer=vectorizer,
            texts=result[text_column],
        )
    )

    result["prediction_score"] = (
        prediction_scores(
            model=model,
            vectorizer=vectorizer,
            texts=result[text_column],
        )
    )

    return result


# ============================================================
# CONFUSION MATRIX
# ============================================================

def get_confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Create a labeled confusion matrix DataFrame.

    Parameters
    ----------
    y_true
        True / weak labels.

    y_pred
        Model predictions.

    labels : list[str], optional
        Class order.

    Returns
    -------
    pandas.DataFrame
        Confusion matrix.
    """

    if labels is None:

        labels = [
            "Negative",
            "Neutral",
            "Positive",
        ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return pd.DataFrame(
        matrix,
        index=[
            f"Actual_{label}"
            for label in labels
        ],
        columns=[
            f"Predicted_{label}"
            for label in labels
        ],
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def get_prediction_errors(
    df: pd.DataFrame,
    actual_column: str = "sentiment",
    predicted_column: str = "predicted_sentiment",
) -> pd.DataFrame:
    """
    Extract documents where the model disagrees with
    the weak sentiment labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction DataFrame.

    actual_column : str
        Weak-label column.

    predicted_column : str
        Prediction column.

    Returns
    -------
    pandas.DataFrame
        Misclassified / disagreement rows.
    """

    required = [
        actual_column,
        predicted_column,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df[
        df[actual_column]
        != df[predicted_column]
    ].copy()


# ============================================================
# MODEL PERSISTENCE
# ============================================================

def save_model(
    model,
    output_path: str | Path,
) -> Path:
    """
    Save a trained model using joblib.

    Parameters
    ----------
    model
        Trained estimator.

    output_path : str or Path
        Destination path.

    Returns
    -------
    pathlib.Path
        Saved model path.
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


def load_model(
    model_path: str | Path,
):
    """
    Load a trained model.

    Parameters
    ----------
    model_path : str or Path
        Model file.

    Returns
    -------
    object
        Loaded estimator.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


# ============================================================
# VECTORIZER PERSISTENCE
# ============================================================

def save_vectorizer(
    vectorizer: TfidfVectorizer,
    output_path: str | Path,
) -> Path:
    """
    Save a fitted TF-IDF vectorizer.
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
) -> TfidfVectorizer:
    """
    Load a saved TF-IDF vectorizer.
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
# COMPLETE SENTIMENT PIPELINE
# ============================================================

def run_sentiment_pipeline(
    df: pd.DataFrame,
    text_column: str = "processed_text",
    model_output_path: Optional[str | Path] = None,
    vectorizer_output_path: Optional[str | Path] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Run the complete sentiment-analysis workflow.

    Workflow
    --------
    Processed Text
        ↓
    VADER Scores
        ↓
    Weak Labels
        ↓
    Train/Test Split
        ↓
    TF-IDF
        ↓
    Model Training
        ↓
    Model Evaluation
        ↓
    Best Model
        ↓
    Final Predictions

    Parameters
    ----------
    df : pandas.DataFrame
        NLP-ready dataset.

    text_column : str
        Processed text column.

    model_output_path : str or Path, optional
        Path for saving best model.

    vectorizer_output_path : str or Path, optional
        Path for saving fitted vectorizer.

    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict
        Pipeline results.
    """

    # --------------------------------------------------------
    # STEP 1 — VADER
    # --------------------------------------------------------

    sentiment_df = add_vader_sentiment(
        df=df,
        text_column=text_column,
    )

    # --------------------------------------------------------
    # STEP 2 — SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        split_data(
            sentiment_df,
            text_column=text_column,
            target_column="sentiment",
            random_state=random_state,
        )
    )

    # --------------------------------------------------------
    # STEP 3 — TF-IDF
    # --------------------------------------------------------

    vectorizer = create_tfidf_vectorizer()

    (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
    ) = fit_transform_tfidf(
        X_train,
        X_test,
        vectorizer,
    )

    # --------------------------------------------------------
    # STEP 4 — MODELS
    # --------------------------------------------------------

    models = create_models(
        random_state=random_state
    )

    trained_models = train_models(
        models=models,
        X_train=X_train_tfidf,
        y_train=y_train,
    )

    # --------------------------------------------------------
    # STEP 5 — EVALUATION
    # --------------------------------------------------------

    comparison_df, detailed_results = (
        evaluate_models(
            models=trained_models,
            X_test=X_test_tfidf,
            y_test=y_test,
        )
    )

    # --------------------------------------------------------
    # STEP 6 — BEST MODEL
    # --------------------------------------------------------

    best_model_name, best_model = (
        select_best_model(
            models=trained_models,
            comparison_df=comparison_df,
            metric="f1_weighted",
        )
    )

    # --------------------------------------------------------
    # STEP 7 — FINAL PREDICTIONS
    # --------------------------------------------------------

    final_df = add_predictions(
        df=sentiment_df,
        model=best_model,
        vectorizer=vectorizer,
        text_column=text_column,
    )

    # --------------------------------------------------------
    # STEP 8 — SAVE MODEL
    # --------------------------------------------------------

    if model_output_path is not None:

        save_model(
            model=best_model,
            output_path=model_output_path,
        )

    if vectorizer_output_path is not None:

        save_vectorizer(
            vectorizer=vectorizer,
            output_path=vectorizer_output_path,
        )

    return {
        "data": final_df,
        "models": trained_models,
        "vectorizer": vectorizer,
        "comparison": comparison_df,
        "detailed_results": detailed_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    sample_texts = [
        "This is an amazing and wonderful experience.",
        "The experience was okay and average.",
        "This is terrible and disappointing.",
    ]

    print("=" * 60)
    print("VADER SENTIMENT TEST")
    print("=" * 60)

    for text in sample_texts:

        scores = vader_scores(
            text
        )

        label = vader_label(
            scores["compound_score"]
        )

        print()
        print("Text:", text)
        print("Scores:", scores)
        print("Label:", label)

    print()
    print("Sentiment module loaded successfully.")