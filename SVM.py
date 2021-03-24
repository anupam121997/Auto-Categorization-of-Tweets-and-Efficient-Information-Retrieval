#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
import wordsegment as ws 
ws.load() 
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn import svm
from sklearn.metrics import accuracy_score,recall_score
import joblib


# In[2]:


def clean_tweets(text):  
    text = text.lower()
    text_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]|))+')
    text = text_pattern.sub('', text)
    emoji = re.compile("["
                        u"\U0001F600-\U0001FFFF"   #EMOJIS
                        u"\U0001F300-\U0001F5FF"   #Symbols and Pictographs
                        u"\U0001F680-\U0001F6FF"   #map symbols
                        u"\U0001F1E0-\U0001F1FF"   #flags
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    text = emoji.sub(r'', text)
    text = re.sub(r'@[a-z0-9]+','',text)
   # text = re.sub(r'#', '', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"this's", "this is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'m", "am", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r'rt[\s]+', '', text)

    text = re.sub(r"[,.\"!#$%^&*(){}?/;'~:<>+=-]", "", text)
    
    tokens = word_tokenize(text)
    
    text = [ws.clean(w) for w in text if len(ws.clean(w)) > 0]
    text = [ws.segment(w)[0] for w in text]   #segmentation
    
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]

    words = [w for w in words if not w in stop_words]
    words = ' '.join(words)
        
    return words


# In[3]:


df=pd.read_csv("tweets_cleaned.csv")

X_train, X_val, y_train, y_val = train_test_split(df['text_cleaned'], df['class'] , test_size=0.2, stratify = df['class'],random_state=0)

model=make_pipeline(TfidfVectorizer(),svm.SVC())

model.fit(X_train, y_train)

#joblib.dump(model,"svm_model.sav")


# In[4]:


load_model = joblib.load("svm_model.sav")

labels_train = load_model.predict(X_train)
labels_val = load_model.predict(X_val)

print("Training Accuracy:",accuracy_score(y_train, labels_train)*100)
print("Validation Accuracy:",accuracy_score(y_val, labels_val)*100)
print(recall_score(y_train, labels_train, average='weighted'))
print(recall_score(y_val, labels_val, average='weighted'))

def predict(s,model = load_model):
    print(model.predict([s]))
    
text = clean_tweets("yogi") 
print(text)
predict(text)


# In[ ]:




