from training import train_classifier
import torch

def check_device():
    '''Literally just checks if your computer has a GPU to use'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

if __name__ == "__main__":
    device=check_device()
    print(f"Using {device}")
    train_classifier(device)
