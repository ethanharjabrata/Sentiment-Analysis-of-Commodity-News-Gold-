import os
from dotenv import load_dotenv

import torch
from huggingface_hub import login

from training import train_classifier
from inference import get_classification


def check_device() -> torch.device:
    """Checks whether a CUDA GPU is available and returns the appropriate torch device.

    Returns:
        - device (torch.device): The CUDA device if available, otherwise the CPU device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    return device


if __name__ == "__main__":
    # log in to hugging face for faster downloads
    # or honestly just comment this line out for convenience
    load_dotenv()
    access_token = os.getenv("access_token")

    if access_token:
        login(token=access_token)

    device = check_device()
    print(f"Using {device}")
    train_classifier(device=device)

    # Weird case (I have genuinely no clue why the classifier can't recognize this as gold price rising)
    print(
        get_classification(
            news_headline="Investors turn to gold, not bonds, as haven from war in Iran",
            device=device,
        )
    )

    # Expected
    print(
        get_classification(
            news_headline="Gold raced close to a record high on Monday",
            device=device,
        )
    )

    print(
        get_classification(
            news_headline="I stubbed my toe",
            device=device,
        )
    )
