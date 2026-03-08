import pandas as pd
#so the typehints don't bug out
from pandas import DataFrame

from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from torch.utils.data import random_split, Dataset
class GoldNewsDataset(Dataset):
    def __init__(self, dataframe:DataFrame):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        '''
        Returns a dictionary'''
def train_classifier(device:str):
    '''
    Contains all the code required to train a classifier model.
    Device should be the text passed in from main.py checking if a GPU is available
    '''
    data= pd.read_csv("./data/gold-dataset-sinha-khandait.csv")
    enc=OrdinalEncoder()
    #print(enc.categories_)

