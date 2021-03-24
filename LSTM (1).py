#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model,save_model
import tensorflow as tf
from numpy import array
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras.utils import np_utils
Encoder = LabelEncoder()
nltk.download('stopwords')


# In[2]:


data = pd.read_csv("tweets_cleaned.csv")


# In[3]:


data


# In[4]:


data.text_cleaned.apply(lambda x: len(x.split(" "))).max()


# In[5]:


text = data['text_cleaned'].tolist()
y = data['class']


# In[6]:


y = Encoder.fit_transform(y)
y = np_utils.to_categorical(y,5)


# In[7]:


y


# In[8]:


token = Tokenizer()
token.fit_on_texts(text)


# In[9]:


vocab_size  = len(token.word_index) + 1


# In[10]:


encoded_text = token.texts_to_sequences(text)


# In[11]:


max_length = 30
X = pad_sequences(encoded_text, maxlen=max_length, padding='post')


# In[12]:


glove_vectors = dict()


# In[13]:


file = open('glove.twitter.27B.200d.txt', encoding='utf-8')

for line in file:
    values = line.split()
    word = values[0]
    #storing the word in the variable
    vectors = np.asarray(values[1: ])
    #storing the vector representation of the respective word in the dictionary
    glove_vectors[word] = vectors
file.close()


# In[14]:


keys = glove_vectors.keys()


# In[15]:


word_vector_matrix = np.zeros((vocab_size, 200))
for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    if vector is not None:
        word_vector_matrix[index] = vector


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3, stratify = y)


# In[17]:


model =  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 200, input_length=max_length, weights = [word_vector_matrix], trainable = False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5)
])


# In[18]:


model.summary()


# In[19]:


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
             optimizer=tf.keras.optimizers.Adam(1e-4), 
             metrics=['accuracy'])


# In[20]:


history = model.fit(x = X_train, y = y_train, epochs = 25, validation_data = (X_test, y_test))

model.save('LSTM.h5')

model.save_weights('LSTM_weights.h5')


# In[21]:


y = model.evaluate(x = X_test, y = y_test)
y[1]


# In[22]:


import matplotlib.pyplot as plt
train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
epochs = range(1,len(train_acc)+1)
plt.plot(epochs, train_acc, 'r', label='Training accuracy')
plt.plot(epochs, test_acc, 'b', label='Testing accuracy')
plt.title('Training  and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

train_loss = history.history['loss']
test_loss = history.history['val_loss']
plt.plot(epochs, train_loss, 'r', label='Training Loss')
plt.plot(epochs, test_loss, 'b', label='Testing Loss')
plt.title('Training  and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc=0)
plt.figure()
plt.show()

