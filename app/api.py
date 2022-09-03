
try:
    from fastapi import FastAPI
    from fastapi.openapi.utils import get_openapi
    
    import numpy as np
    import pandas as pd 
    
    from pydantic import BaseModel
    
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    import emoji

    
    import re
    
    import tensorflow as tf
    import tensorflow
    
    from nltk.stem import PorterStemmer
    
    import pickle
    
    import json
except:
    print("Error importing packages!")
    
    
app = FastAPI()

def tweet_preprocess(tweet):
    stemmer = PorterStemmer()
    new_tweet = tweet.lower()
    new_tweet = re.sub(r'@\w+', '', new_tweet)
    new_tweet = re.sub(r'#', '', new_tweet)
    new_tweet = re.sub(r':', ' ', emoji.demojize(new_tweet))
    new_tweet = re.sub(r'http\S+','', new_tweet)
    new_tweet = re.sub(r'\$\S+', 'dollar', new_tweet) # Change dollar amounts to dollar
    new_tweet = re.sub(r'[^a-z0-9\s]', '', new_tweet) # Remove punctuation
    new_tweet = re.sub(r'[0-9]+', 'number', new_tweet) # Change number values to number
    new_tweet = new_tweet.split(" ")
    new_tweet =  list(map(lambda x: stemmer.stem(x), new_tweet))
    new_tweet =  list(map(lambda x: x.strip(), new_tweet))
    if '' in new_tweet:
        new_tweet.remove('')
     
    return new_tweet


def text_preprocess(data):
 
    data = [tweet_preprocess(data)]
    
    with open('config.json', 'r') as openfile:
        conf = json.load(openfile)
        
    max_seq_length = conf["max_seq_length"]
    
    with open(r'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    sequences = tokenizer.texts_to_sequences(data)
    
    model_inputs = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    
    return model_inputs

class Sentiment(BaseModel):
    text : str
    
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    
    
@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


@app.post('/predict', response_model= SentimentResponse)
async def predict_sentiment(data: Sentiment):
    text = data.dict()
    model = load_model("model")
    data_in = text_preprocess(text["text"])
    probability = model(data_in)[0][0]
    prediction = 'positive' if probability>0.5  else 'negative'
    confidence = 100 - probability if prediction=='negative' else probability
    
    return SentimentResponse(
        sentiment=prediction, confidence=confidence
    )
    
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sentiment Predictor",
        version="2.5.0",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

    