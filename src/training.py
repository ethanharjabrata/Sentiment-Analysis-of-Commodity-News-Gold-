import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    BatchEncoding,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from torch.utils.data import Dataset
from torch import tensor


class GoldDataset(Dataset):
    """PyTorch Dataset wrapping tokenized headline encodings and integer sentiment labels."""

    def __init__(self, encodings: BatchEncoding, labels: list[int]) -> None:
        """Stores tokenizer output and corresponding labels for use by the Trainer.

        Args:
            - encodings (BatchEncoding): Tokenizer output for a list of headlines.
            - labels (list[int]): Integer sentiment labels aligned with the encodings.

        Returns:
            - None
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns the tensor dictionary for a single example.

        Args:
            - idx (int): Index of the example to retrieve.

        Returns:
            - item (dict[str, torch.Tensor]): Tensor dict with tokenizer output keys
                plus a 'labels' key.
        """
        item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """Returns the number of examples in the dataset.

        Returns:
            - length (int): Total number of examples.
        """
        return len(self.labels)


def get_f1_and_acc(eval_pred: EvalPrediction) -> dict[str, float]:
    """Computes weighted F1 and accuracy from a Trainer EvalPrediction object.

    Args:
        - eval_pred (EvalPrediction): Prediction object from the HuggingFace Trainer,
            containing raw logits and ground-truth labels.

    Returns:
        - metrics (dict[str, float]): Dict with keys 'f1' and 'accuracy'.
    """
    logits, labels = eval_pred

    # Some models return a tuple for logits, uncomment this to pull out first element
    # if isinstance(logits, tuple):
    #     logits = logits[0]

    predictions = np.argmax(a=logits, axis=-1)

    # average='weighted' accounts for class imbalance (positive makes up over 40% of examples)
    f1 = f1_score(y_true=labels, y_pred=predictions, average="weighted")
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)

    metrics = {"f1": f1, "accuracy": accuracy}
    return metrics


def train_classifier(device: torch.device) -> None:
    """Trains a DistilBERT sequence classifier on the gold sentiment dataset and saves it.

    Splits data 80/10/10 (train/val/test), fine-tunes distilbert-base-uncased for up to
    10 epochs with early stopping (patience=2 on weighted F1), then saves the best
    checkpoint to ./models/gold-sentiment-classifier.

    Args:
        - device (torch.device): The device (CPU or CUDA) on which to run training.

    Returns:
        - None
    """
    data = pd.read_csv(filepath_or_buffer="./data/gold-dataset-sinha-khandait.csv")
    # print(data.isna().sum())
    # Nothing missing

    # Categories are positive, negative, neutral (Price not changing), and none (implying no useful information)
    label_map = {
        "negative": 0,
        "positive": 1,
        "neutral": 2,
        "none": 3,
    }

    data["label"] = data["Price Sentiment"].map(func=label_map)

    # Apparently Bert Tokenizer needs the data to be passed in as a list
    x_train, x_temp, y_train, y_temp = train_test_split(
        data["News"].tolist(),
        data["label"].tolist(),
        test_size=0.2,
        stratify=data["label"],
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        pretrained_model_name_or_path="distilbert-base-uncased",
    )

    train_tokens = tokenizer(text=x_train, padding=True, truncation=True)
    val_tokens = tokenizer(text=x_val, padding=True, truncation=True)
    test_tokens = tokenizer(text=x_test, padding=True, truncation=True)

    train = GoldDataset(encodings=train_tokens, labels=y_train)
    val = GoldDataset(encodings=val_tokens, labels=y_val)
    test = GoldDataset(encodings=test_tokens, labels=y_test)

    model = DistilBertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="distilbert-base-uncased",
        num_labels=4,
    )

    model.to(device=device)

    args = TrainingArguments(
        num_train_epochs=10,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=2e-5,
        warmup_steps=0.1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=get_f1_and_acc,
        # should probably jack up patience if you want to make sure you're not in some local max
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    test_results = trainer.predict(test_dataset=test)

    # trainer.predict automatically runs compute_metrics, prefixed with "test_"
    print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results.metrics['test_f1']:.4f}\n")

    # Save the trained model weights
    trainer.save_model(output_dir="./models/gold-sentiment-classifier")

    # Save the tokenizer
    tokenizer.save_pretrained(save_directory="./models/gold-sentiment-classifier")
