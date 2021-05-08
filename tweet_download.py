# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:18:29 2021

@author: LENOVO
"""

import time
import json
import tweepy
from tweepy import OAuthHandler
    
from tweepy import Stream
from tweepy.streaming import StreamListener

access_token = "773003693797040128-jUHoBg5eHTs2uvt4EJPzQC5SSCF1XQt"
access_token_secret = "rk3YLgJQzgtFmCotfTOBb0YV1W1TRvcToBOI6qUfWFchM"

api_key = "Ul2v0hXS7V2ffEuqLQyatVkYg"
api_secret = "dE62A3PzAlgX3RTyCesQgkVkXFr6z1ULTqCWuZBsjDYgbHQdAf"

auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

print ("Getting Tweets...")

class MyListener(StreamListener):
 
    def on_data(self, data):
        print ("...")
        try:
            with open('tweets_politics.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        
        return True
 
    def on_error(self, status):
        print(status)
        return False
 
twitter_stream = Stream(auth, MyListener())

twitter_stream.filter(track=['politics'],languages = ["en"], encoding='utf8')

print ("TWEETS SUCCESFULLY DOWNLOADED IN 'tweet.txt'...")