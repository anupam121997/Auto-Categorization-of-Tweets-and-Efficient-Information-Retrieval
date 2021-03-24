# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:22:57 2021

@author: LENOVO
"""

import pandas as pd
data_technology = pd.read_json('tweets_technology.json', lines = True)
data_technology.sort_values('text',inplace=True)
data_technology.drop_duplicates(subset='text',keep = False, inplace=True)

data_sports = pd.read_json('tweets_sports.json', lines = True)
data_sports.sort_values('text',inplace=True)
data_sports.drop_duplicates(subset='text',keep = False, inplace=True)

data_entertainment = pd.read_json('tweets_entertainment.json', lines = True)
data_entertainment.sort_values('text',inplace=True)
data_entertainment.drop_duplicates(subset='text',keep = False, inplace=True)

data_education = pd.read_json('tweets_education.json', lines = True)
data_education.sort_values('text',inplace=True)
data_education.drop_duplicates(subset='text',keep = False, inplace=True)

data_politics = pd.read_json('tweets_politics.json', lines = True)
data_politics.sort_values('text',inplace=True)
data_politics.drop_duplicates(subset='text',keep = False, inplace=True)

df_technology = pd.DataFrame()
df_technology['id'] = data_technology['id']
df_technology['class'] = 'technology'
df_technology['text'] = data_technology['text']

df_sports = pd.DataFrame()
df_sports['id'] = data_sports['id']
df_sports['class'] = 'sports'
df_sports['text'] = data_sports['text']

df_entertainment = pd.DataFrame()
df_entertainment['id'] = data_entertainment['id']
df_entertainment['class'] = 'entertainment'
df_entertainment['text'] = data_entertainment['text']

df_education = pd.DataFrame()
df_education['id'] = data_education['id']
df_education['class'] = 'education'
df_education['text'] = data_education['text']

df_politics = pd.DataFrame()
df_politics['id'] = data_politics['id']
df_politics['class'] = 'politics'
df_politics['text'] = data_politics['text']
#df_politics = df_politics.sample(frac=0.8)


frames = [df_politics, df_technology, df_education, df_entertainment, df_sports]

df = pd.concat(frames)

df = df.sample(frac = 1)

df.to_csv('Tweet_dataset_new.csv',index=False)
#df.to_csv(r'Tweet_dataset_new.csv',header = None, index=None, sep = ' ', mode = 'a')