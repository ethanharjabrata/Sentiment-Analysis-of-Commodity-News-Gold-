import torch
from transformers import pipeline


def get_classification(news_headline: str, device: torch.device) -> str:
    """Classifies a gold market news headline into one of four sentiment categories.

    Args:
        - news_headline (str): A raw news headline about the gold commodity market.
        - device (torch.device): The device (CPU or CUDA) on which to run inference.

    Returns:
        - prediction (str): A human-readable sentiment label, one of:
            'Gold prices expected to Fall', 'Gold prices expected to Rise',
            'Gold prices unlikely to Change', or 'No useful information provided'.
    """
    if device == torch.device("cuda"):
        device_id = 0

    else:
        device_id = -1

    sentiment_analyzer = pipeline(
        task="text-classification",
        model="./models/gold-sentiment-classifier",
        tokenizer="./models/gold-sentiment-classifier",
        device=device_id,
    )

    result = sentiment_analyzer(inputs=news_headline)[0]

    if result["label"] == "LABEL_0":
        prediction = "Gold prices expected to Fall"

    elif result["label"] == "LABEL_1":
        prediction = "Gold prices expected to Rise"

    elif result["label"] == "LABEL_2":
        prediction = "Gold prices unlikely to Change"

    elif result["label"] == "LABEL_3":
        prediction = "No useful information provided"

    else:
        prediction = "Unknown label"

    return prediction
