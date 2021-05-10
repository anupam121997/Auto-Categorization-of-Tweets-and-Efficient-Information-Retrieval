import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
import wordsegment as ws 
ws.load() 
from numpy.linalg import norm
import numpy as np
import math

import joblib
from rank_bm25 import BM25Okapi

import ast

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

cosine_similarity = lambda a, b: np.inner(a, b) / (norm(a) * norm(b)) if norm(a) != 0.0 and norm(b) != 0.0 else 0.0

euclidean_similarity = lambda a, b: np.linalg.norm(a - b)

dot_product = lambda a,b: np.dot(a,b)

def predict(s,model):
    return model.predict([s])


def TfIdf(query_vec, tweet, tf_idf, n=10):

    result = tf_idf.apply(lambda row: dot_product(query_vec, row), 
                          axis='columns').sort_values(ascending=False)
    
    tweetsIDs = [i for i in range(len(tweet))]
    
    tweets = tweet['text']
    
    res = []
    for i in range(0, n):
        res.append(tweets[tweetsIDs.index(result.index[i])])
    
    return res

def Cosine(query_vec, tweet, tf_idf, n=10):

    result = tf_idf.apply(lambda row: cosine_similarity(query_vec, row), 
                          axis='columns').sort_values(ascending=False)
    
    tweetsIDs = [i for i in range(len(tweet))]
    
    tweets = tweet['text']
    
    res = []
    for i in range(0, n):
        res.append(tweets[tweetsIDs.index(result.index[i])])
        
    return res
          
def Euclidean(query_vec, tweet, tf_idf, n=10):

    result = tf_idf.apply(lambda row: euclidean_similarity(query_vec, row), 
                          axis='columns').sort_values(ascending=False)
    
    tweetsIDs = [i for i in range(len(tweet))]
    
    tweets = tweet['text']
    
    res = []
    for i in range(0, n):
        res.append(tweets[tweetsIDs.index(result.index[i])])
        
    return res
    
def jaccard(tweets, tokens, data_1, data_2):
    
    query=set(tokens)

    tem_dic={}
    for i in data_2.keys():
        set_1=set(data_2[i])
        inter=set_1.intersection(query)
        union=set_1.union(query)
        score=len(inter)/len(union)
        tem_dic[i]=score
    
    sorted_keys = sorted(tem_dic, key=tem_dic.get)  # [1, 3, 2]
    sorted_keys.reverse()
    
    res = []
    for i in range(10):
        for j in range(len(tweets)):
            if data_1[sorted_keys[i]] == tweets.iloc[j]['id']:
                res.append(tweets.iloc[j]['text'])
                
    return res
    
def bm(query_vec, tweets, data, data_1):
    res = {}
    for j in range(1,len(tweets)+1):
        k = str(j)
        t=data.loc[:,k]
        res[j] = dot_product(query_vec, t)
        
    sorted_keys = sorted(res, key=res.get)
    sorted_keys.reverse()
    
    res = []
    for i in range(10):
        for j in range(len(tweets)):
            if data_1[sorted_keys[i]] == tweets.iloc[j]['id']:
                res.append(tweets.iloc[j]['text'])
     
    return res
    
def return_params(query, model):

    query = clean_tweets(query)
    
    load_model = joblib.load("svm_model.sav")
    tweet_class = predict(query,load_model)
    
    print(tweet_class)
    print(query)
    
    list1 = []
            
    if tweet_class[0] == 'sports':
        tweets = pd.read_csv('tweets_cleaned_sports.csv')
        tf_idf = joblib.load("tf_idf_vector_sports1.sav")
        data = pd.read_csv('BM25_sports.csv')
        
        with open('document_id_name_decionary_sports.txt',errors="ignore",encoding ="utf-8") as f: 
            data_1 = f.read()
    
        with open('docid_tokens_sports.txt',errors="ignore",encoding ="utf-8") as f: 
            data_2 = f.read()
            
        data_1 = ast.literal_eval(data_1)
        data_2 = ast.literal_eval(data_2)
        
        terms=list(tf_idf.columns)
        terms1=list(data['word'])
        
        tokens=word_tokenize(query)
        
        query_vec=[]
        k=0
        for i in terms:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec.append(1)
            else:
                query_vec.append(0)
        
        query_vec1=[]
        k=0
        for i in terms1:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec1.append(1)
            else:
                query_vec1.append(0)
        
        if model == 'tfidf':
            print(model)
            list1 = TfIdf(query_vec, tweets, tf_idf)
            
        elif model == 'cosine':
            print(model)
            list1 = Cosine(query_vec, tweets, tf_idf)
            
        elif model == 'bm25':
            print(model)
            list1 = bm(query_vec1, tweets, data, data_1)
            
        elif model == 'euclidean':
            print(model)
            list1 = Euclidean(query_vec, tweets, tf_idf)
            
        elif model == 'jaccard':
            print(model)
            list1 = jaccard(tweets, tokens, data_1, data_2)
            
    elif tweet_class[0] == 'education':
        tf_idf = joblib.load("tf_idf_vector_education1.sav")
        tweets = pd.read_csv('tweets_cleaned_education.csv')
        data = pd.read_csv('BM25_education.csv')
        
        with open('document_id_name_decionary_education.txt',errors="ignore",encoding ="utf-8") as f: 
            data_1 = f.read()
    
        with open('docid_tokens_education.txt',errors="ignore",encoding ="utf-8") as f: 
            data_2 = f.read()
            
        data_1 = ast.literal_eval(data_1)
        data_2 = ast.literal_eval(data_2)
        
        terms=list(tf_idf.columns)
        terms1=list(data['word'])
        
        tokens=word_tokenize(query)
        
        query_vec=[]
        k=0
        for i in terms:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec.append(1)
            else:
                query_vec.append(0)
        
        query_vec1=[]
        k=0
        for i in terms1:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec1.append(1)
            else:
                query_vec1.append(0)
        
        if model == 'tfidf':
            print(model)
            list1 = TfIdf(query_vec, tweets, tf_idf)
            
        elif model == 'cosine':
            print(model)
            list1 = Cosine(query_vec, tweets, tf_idf)
            
        elif model == 'bm25':
            print(model)
            list1 = bm(query_vec1, tweets, data, data_1)
            
        elif model == 'euclidean':
            print(model)
            list1 = Euclidean(query_vec, tweets, tf_idf)
            
        elif model == 'jaccard':
            print(model)
            list1 = jaccard(tweets, tokens, data_1, data_2)
            
    elif tweet_class[0] == 'politics':
        tf_idf = joblib.load("tf_idf_vector_politics1.sav")
        tweets = pd.read_csv('tweets_cleaned_politics.csv')
        data = pd.read_csv('BM25_politics.csv')
        
        with open('document_id_name_decionary_politics.txt',errors="ignore",encoding ="utf-8") as f: 
            data_1 = f.read()
    
        with open('docid_tokens_politics.txt',errors="ignore",encoding ="utf-8") as f: 
            data_2 = f.read()
            
        data_1 = ast.literal_eval(data_1)
        data_2 = ast.literal_eval(data_2)
        
        terms=list(tf_idf.columns)
        terms1=list(data['word'])
        
        tokens=word_tokenize(query)
        
        query_vec=[]
        k=0
        for i in terms:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec.append(1)
            else:
                query_vec.append(0)
        
        query_vec1=[]
        k=0
        for i in terms1:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec1.append(1)
            else:
                query_vec1.append(0)
        
        if model == 'tfidf':
            print(model)
            list1 = TfIdf(query_vec, tweets, tf_idf)
            
        elif model == 'cosine':
            print(model)
            list1 = Cosine(query_vec, tweets, tf_idf)
            
        elif model == 'bm25':
            print(model)
            list1 = bm(query_vec1, tweets, data, data_1)
            
        elif model == 'euclidean':
            print(model)
            list1 = Euclidean(query_vec, tweets, tf_idf)
            
        elif model == 'jaccard':
            print(model)
            list1 = jaccard(tweets, tokens, data_1, data_2)
            
    elif tweet_class[0] == 'entertainment':
        tf_idf = joblib.load("tf_idf_vector_entertainment1.sav")
        tweets = pd.read_csv('tweets_cleaned_entertainment.csv')
        data = pd.read_csv('BM25_entertainment.csv')
        
        with open('document_id_name_decionary_entertainment.txt',errors="ignore",encoding ="utf-8") as f: 
            data_1 = f.read()
    
        with open('docid_tokens_entertainment.txt',errors="ignore",encoding ="utf-8") as f: 
            data_2 = f.read()
            
        data_1 = ast.literal_eval(data_1)
        data_2 = ast.literal_eval(data_2)
        
        terms=list(tf_idf.columns)
        terms1=list(data['word'])
        
        tokens=word_tokenize(query)
        
        query_vec=[]
        k=0
        for i in terms:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec.append(1)
            else:
                query_vec.append(0)
        
        query_vec1=[]
        k=0
        for i in terms1:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec1.append(1)
            else:
                query_vec1.append(0)
        
        if model == 'tfidf':
            print(model)
            list1 = TfIdf(query_vec, tweets, tf_idf)
            
        elif model == 'cosine':
            print(model)
            list1 = Cosine(query_vec, tweets, tf_idf)
            
        elif model == 'bm25':
            print(model)
            list1 = bm(query_vec1, tweets, data, data_1)
            
        elif model == 'euclidean':
            print(model)
            list1 = Euclidean(query_vec, tweets, tf_idf)
            
        elif model == 'jaccard':
            print(model)
            list1 = jaccard(tweets, tokens, data_1, data_2)
            
            
    elif tweet_class[0] == 'technology':
        tf_idf = joblib.load("tf_idf_vector_technology1.sav")
        tweets = pd.read_csv('tweets_cleaned_technology.csv')
        data = pd.read_csv('BM25_technology.csv')
        
        with open('document_id_name_decionary_technology.txt',errors="ignore",encoding ="utf-8") as f: 
            data_1 = f.read()
    
        with open('docid_tokens_technology.txt',errors="ignore",encoding ="utf-8") as f: 
            data_2 = f.read()
            
        data_1 = ast.literal_eval(data_1)
        data_2 = ast.literal_eval(data_2)
        
        terms=list(tf_idf.columns)
        terms1=list(data['word'])
        
        tokens=word_tokenize(query)
        
        query_vec=[]
        k=0
        for i in terms:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec.append(1)
            else:
                query_vec.append(0)
        
        query_vec1=[]
        k=0
        for i in terms1:
            flag=0
            for j in tokens:
                if(i==j):
                    flag=1
            if(flag==1):
                query_vec1.append(1)
            else:
                query_vec1.append(0)
        
        if model == 'tfidf':
            print(model)
            list1 = TfIdf(query_vec, tweets, tf_idf)
            
        elif model == 'cosine':
            print(model)
            list1 = Cosine(query_vec, tweets, tf_idf)
            
        elif model == 'bm25':
            print(model)
            list1 = bm(query_vec1, tweets, data, data_1)
            
        elif model == 'euclidean':
            print(model)
            list1 = Euclidean(query_vec, tweets, tf_idf)
            
        elif model == 'jaccard':
            print(model)
            list1 = jaccard(tweets, tokens, data_1, data_2)
                
    return (tweet_class[0],list1)