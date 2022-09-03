try:
    from model import GRU_model
    import numpy as np
    import pandas as pd
    
    import os
    import re
    import emoji
    from nltk.stem import PorterStemmer
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import tensorflow as tf
    
    from model_config import CONFIG
    import pickle
    
    import json
    
except:
    print("Error importing packages!")
    
class Train(object):
    def __init__(self, stemmer, embedding_dim, epochs, batch_size):
        self.stemmer = stemmer
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 
        self.vocab_length = None
        self.max_seq_length = None
        self.df = pd.read_csv('airline_sentiment_analysis.csv').drop(columns='Unnamed: 0')
    

    def tweet_preprocess(self, tweet):
        new_tweet = tweet.lower()
        new_tweet = re.sub(r'@\w+', '', new_tweet)
        new_tweet = re.sub(r'#', '', new_tweet)
        new_tweet = re.sub(r':', ' ', emoji.demojize(new_tweet))
        new_tweet = re.sub(r'http\S+','', new_tweet)
        new_tweet = re.sub(r'\$\S+', 'dollar', new_tweet) # Change dollar amounts to dollar
        new_tweet = re.sub(r'[^a-z0-9\s]', '', new_tweet) # Remove punctuation
        new_tweet = re.sub(r'[0-9]+', 'number', new_tweet) # Change number values to number
        new_tweet = new_tweet.split(" ")
        new_tweet =  list(map(lambda x: self.stemmer.stem(x), new_tweet))
        new_tweet =  list(map(lambda x: x.strip(), new_tweet))
        if '' in new_tweet:
            new_tweet.remove('')
         
        return new_tweet
    
    def get_train_params(self, tweets):
        vocabulary = set()

        for tweet in tweets:
            for word in tweet:
                if word not in vocabulary:
                    vocabulary.add(word)
        
        vocab_length = len(vocabulary)
        
        # Get max length of a sequence
        max_seq_length = 0
        
        for tweet in tweets:
            if len(tweet) > max_seq_length:
                max_seq_length = len(tweet)
        
        dictionary = {
            "vocab_length": vocab_length,
             "max_seq_length": max_seq_length,
        }
        
        with open("config.json", "w") as outfile:
            json.dump(dictionary, outfile)
                
        return vocab_length, max_seq_length

    
    def get_data(self):
        tweets = self.df['text'].apply(self.tweet_preprocess)
        labels = np.array(self.df['airline_sentiment'])
        vocab_length, max_seq_length = self.get_train_params(tweets=tweets)
        
        tokenizer = Tokenizer(num_words = vocab_length)
        tokenizer.fit_on_texts(tweets)
        
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        sequences = tokenizer.texts_to_sequences(tweets)
        
        word_index = tokenizer.word_index
        
        model_inputs = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
        
        le = LabelEncoder()
        
        labels = le.fit_transform(labels)
        X_train, X_test, y_train, y_test  = train_test_split(model_inputs, labels, train_size=0.7, random_state = 22, stratify=labels)
        
        return X_train, X_test, y_train, y_test, vocab_length, max_seq_length
    
    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.vocab_length, self.max_seq_length = self.get_data()
        
        model = GRU_model(self.max_seq_length, self.embedding_dim, self.vocab_length)
        
        model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
        
        if not os.path.exists(CONFIG.checkpoint_dir):
            os.mkdir(CONFIG.checkpoint_dir)
            
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CONFIG.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        history = model.fit(self.X_train, self.y_train, validation_split = 0.2, epochs=self.epochs,
                   batch_size = self.batch_size,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                   monitor = 'val_loss',
                   patience = 3,
                   restore_best_weights = True,
                   verbose = 1),
                   cp_callback,
                   tf.keras.callbacks.ReduceLROnPlateau()])
        
        
    def evaluate(self):
        model = GRU_model(self.max_seq_length, self.embedding_dim, self.vocab_length)
        model.load_weights(CONFIG.checkpoint_path)

        model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
        
        results = model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test Loss: {}'.format(results[0]))
        print('Test Accuracy: {}'.format(results[1]))
        model.save("model")
  
if __name__=='__main__':

    ps = PorterStemmer()
    model_train = Train(ps, CONFIG.embedding_dim, CONFIG.epochs, CONFIG.batch_size)
    model_train.train()
    model_train.evaluate()