# Purpose
Contains all the scripts for training NLP models for sentiment analysis using the Kaggle gold dataset
# Set up
Download uv \
Run 
```
uv sync
```
This should install all dependencies required. 
# Running the code
Make a .env file in the root of this repo and put your Access token in the file 
```
access_token=(PUT YOUR HUGGING FACE ACCESS TOKEN HERE)
```
# Data Sources
All data taken from [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold)

## To do?
Currently debating making a seperate script for training a regressor outputting a scale from -1 to 1 for negative to positive. \
Need to figure out how I can differentiate between an inconclusive article and an article that implies 0 change in gold. 

