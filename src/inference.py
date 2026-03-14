from transformers import pipeline
def get_classification(news_headline:str, device:str):
    '''
    Takes in a news_headline and uses the classifier model to perform sentiment analysis
    Device should be the same string outputted in main.py
    '''
    if device=="cuda":
        device=0
    else:
        device=-1
    sentiment_analyzer = pipeline(
        "text-classification", 
        model="./models/gold-sentiment-classifier", 
        tokenizer="./models/gold-sentiment-classifier",
        device=device
    )
    #pull out the dict
    prediction=sentiment_analyzer(news_headline)[0]
    #Convert into human readable text
    if prediction['label']=="LABEL_0":
        prediction="Gold prices expected to Fall"
    elif prediction['label']=="LABEL_1":
        prediction="Gold prices expected to Rise"
    elif prediction['label']=="LABEL_2":
        prediction="Gold prices unlikely to Change"
    elif prediction['label']=="LABEL_3":
        prediction="No useful information provided"
    return prediction

