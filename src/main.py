from training import train_classifier
from inference import get_classification
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv
def check_device():
    '''Literally just checks if your computer has a GPU to use'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

if __name__ == "__main__":
    #log in to hugging face for faster downloads
    # or honestly just comment this line out for convenience
    load_dotenv()
    access_token=os.getenv("access_token")
    login(token=access_token)

    device=check_device()
    print(f"Using {device}")
    train_classifier(device)

    #Weird case (I have genuinely no clue why the classifier can't recognize this as gold price rising)
    print(get_classification("Investors turn to gold, not bonds, as haven from war in Iran", device))

    #Expected
    print(get_classification("Gold raced close to a record high on Monday", device))
    print(get_classification("I stubbed my toe", device))
