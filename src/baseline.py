from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


LABEL_MAP = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
    "none": 3,
}

LABEL_NAMES = ["negative", "positive", "neutral", "none"]

# Reference scores from the trained DistilBERT model (src/training.py)
DISTILBERT_F1 = 0.9046
DISTILBERT_ACC = 0.9054


@dataclass
class DataSplit:
    """Stratified 80/10/10 split of the gold sentiment dataset.

    Attributes:
        - x_train (list[str]): Training set headlines.
        - x_val (list[str]): Validation set headlines.
        - x_test (list[str]): Test set headlines.
        - y_train (list[int]): Training set integer labels.
        - y_val (list[int]): Validation set integer labels.
        - y_test (list[int]): Test set integer labels.
    """

    x_train: list[str]
    x_val: list[str]
    x_test: list[str]
    y_train: list[int]
    y_val: list[int]
    y_test: list[int]


def load_and_split(csv_path: str) -> DataSplit:
    """Loads the gold dataset and splits it 80/10/10 into train, validation, and test sets.

    Uses stratified sampling to preserve class proportions, mirroring the approach in
    training.py. A fixed random_state makes the baseline split reproducible; note that
    training.py does not fix its seed, so the exact examples in each split will differ.

    Args:
        - csv_path (str): Path to the raw CSV file.

    Returns:
        - split (DataSplit): The three stratified headline/label splits.
    """
    data = pd.read_csv(filepath_or_buffer=csv_path)
    data["label"] = data["Price Sentiment"].map(func=LABEL_MAP)

    x_train, x_temp, y_train, y_temp = train_test_split(
        data["News"].tolist(),
        data["label"].tolist(),
        test_size=0.2,
        stratify=data["label"],
        random_state=42,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    split = DataSplit(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    return split


def evaluate(
    fitted_pipeline: Pipeline,
    x_test: list[str],
    y_test: list[int],
    model_name: str,
) -> None:
    """Evaluates a fitted sklearn Pipeline on the test set and prints a results table.

    Args:
        - fitted_pipeline (Pipeline): A fitted sklearn Pipeline ending in a classifier.
        - x_test (list[str]): Test set headlines.
        - y_test (list[int]): True integer labels for the test set.
        - model_name (str): Human-readable name printed in the results header.

    Returns:
        - None
    """
    predictions = fitted_pipeline.predict(X=x_test)
    f1 = f1_score(y_true=y_test, y_pred=predictions, average="weighted")
    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)

    print(f"=== {model_name} ===")
    print(f"  Weighted F1 : {f1:.4f}  (DistilBERT: {DISTILBERT_F1:.4f})")
    print(f"  Accuracy    : {accuracy:.4f}  (DistilBERT: {DISTILBERT_ACC:.4f})")
    print()
    print(
        classification_report(
            y_true=y_test,
            y_pred=predictions,
            target_names=LABEL_NAMES,
        )
    )


def run_baselines(csv_path: str) -> None:
    """Trains and evaluates TF-IDF + Logistic Regression and TF-IDF + Naive Bayes baselines.

    Both models use unigrams and bigrams with a 50,000-feature vocabulary. Results are
    printed alongside DistilBERT reference scores for direct comparison.

    Args:
        - csv_path (str): Path to the raw CSV file.

    Returns:
        - None
    """
    split = load_and_split(csv_path=csv_path)

    print(
        f"Split — Train: {len(split.x_train)}, "
        f"Val: {len(split.x_val)}, "
        f"Test: {len(split.x_test)}\n"
    )

    lr_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ],
    )

    lr_pipeline.fit(X=split.x_train, y=split.y_train)

    evaluate(
        fitted_pipeline=lr_pipeline,
        x_test=split.x_test,
        y_test=split.y_test,
        model_name="TF-IDF + Logistic Regression",
    )

    nb_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)),
            ("clf", MultinomialNB()),
        ],
    )

    nb_pipeline.fit(X=split.x_train, y=split.y_train)

    evaluate(
        fitted_pipeline=nb_pipeline,
        x_test=split.x_test,
        y_test=split.y_test,
        model_name="TF-IDF + Naive Bayes",
    )


if __name__ == "__main__":
    run_baselines(csv_path="./data/gold-dataset-sinha-khandait.csv")
