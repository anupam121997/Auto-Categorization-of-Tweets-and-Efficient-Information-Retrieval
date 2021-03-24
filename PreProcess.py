# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:49:42 2021

@author: LENOVO
"""

import pandas as pd

import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import wordsegment as ws 
ws.load()
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

"""tweet_df = pd.read_csv("/content/gdrive/MyDrive/IR Project/Tweet_dataset.csv");
tweet_df
"""

#nltk.download('punkt')

#nltk.download('stopwords')



def clean_tweets(df):
  all_tweets = []
  i = 1
  tweets = df['text'].values.tolist();
  for text in tweets:
    print(i)
    i += 1  
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
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    
    words = [w for w in words if not w in stop_words]
    sentence = []
    for w in words:
      sentence.append(lemmatizer.lemmatize(w))
    
    sentence = ' '.join(sentence)
    
    all_tweets.append(sentence)
  return all_tweets

nan_value = float("NaN")

tweet_df = pd.read_csv("Tweet_dataset_new.csv")

Cleaned_tweets = clean_tweets(tweet_df)

tweet_df['text_cleaned'] = Cleaned_tweets

tweet_df_cleaned = tweet_df[['id','class', 'text_cleaned']]

#
tweet_df_cleaned.replace("", nan_value, inplace=True)
tweet_df_cleaned.dropna(subset=['text_cleaned'], inplace=True)

tweet_df_cleaned = tweet_df_cleaned.sample(frac = 1)

tweet_df_cleaned.to_csv('tweets_cleaned_new.csv',index=False)