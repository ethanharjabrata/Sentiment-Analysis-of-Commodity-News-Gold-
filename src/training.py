import pandas as pd

# from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import Dataset
from torch import tensor
from sklearn.metrics import f1_score, accuracy_score
import pickle
class GoldDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_f1_and_acc(eval_pred):
    """
    Takes an EvalPrediction object from the Trainer and returns a dictionary of f1 and accuracy
    Dictionary has keys "f1", and "accuracy". 
    """
    logits, labels = eval_pred
    
    # Some models return a tuple for logits, uncomment this to pull out first element
    # if isinstance(logits, tuple):
    #     logits = logits[0]
        
    # Convert the logits (raw model outputs) into final class predictions (0, 1, 2, or 3)
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate the multi-class F1 score
    # We use average='weighted' since some classes show up a lot more than others (positive makes up over 40% of the results)
    f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "f1": f1,
        "accuracy": accuracy
    }
def train_classifier(device:str):
    '''
    Contains all the code required to train a classifier model.
    Device should be the text passed in from main.py checking if a GPU is available
    '''
    data= pd.read_csv("./data/gold-dataset-sinha-khandait.csv")
    # print(data.isna().sum())
    # Nothing missing

    # Categories are positive, negative, neutral (Price not changing), and none (implying no useful information)
    label_map = {"negative": 0, "positive": 1, "neutral": 2, "none": 3}
    data["label"] = data["Price Sentiment"].map(label_map)
    
    # Apparently Bert Tokenizer needs the data to be passed in as a list
    x_train, x_test, y_train, y_test = train_test_split(
        data['News'].tolist(), 
        data['label'].tolist(), 
        test_size=0.2, 
        stratify=data['label']
    )
    tokenizer= DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_tokens=tokenizer(x_train, padding=True, truncate=True)
    test_tokens=tokenizer(x_test, padding=True, truncate=True)
    train=GoldDataset(train_tokens, y_train)
    test=GoldDataset(test_tokens, y_test)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
    model.to(device)

    args=TrainingArguments(
        num_train_epochs=10,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    trainer=Trainer(
        model, 
        args, 
        train_dataset=train, 
        eval_dataset=test, 
        compute_metrics=get_f1_and_acc,
        #should probably jack up patience if you want to make sure you're not in some local max
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    # Save the trained model weights
    trainer.save_model("./models/gold-sentiment-classifier")

    # Save the tokenizer (crucial so you can process new text later)
    tokenizer.save_pretrained("./models/gold-sentiment-classifier")
    
    



