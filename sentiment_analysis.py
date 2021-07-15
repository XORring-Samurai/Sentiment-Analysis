# -*- coding: utf-8 -*-
"""Sentiment Analysis

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/XORring-Samurai/Sentiment-Analysis/blob/main/Sentiment_Analysis.ipynb
"""

!pwd

from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)

import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

df = pd.read_csv('/content/gdrive/MyDrive/IMDB Dataset.csv')

df['sentiment'].value_counts()
# to check for imbalance in the data

df_positive = df[df['sentiment']=='positive'][:2500]
df_negative = df[df['sentiment']=='negative'][:2500]

df_mini = pd.concat([df_positive, df_negative])

train, test = train_test_split(df_mini, test_size = 0.30, random_state = 0)

train_x = train['review']
train_y = train['sentiment']
test_x = test['review']
test_y = test['sentiment']

train_y

'''
Box of words: We need to convert text into numerical data
so as to be able to build a model on it

TF-IDF (Term Frequency, Inverse Document Frequency)
Suppose there is some word, which is frequent in all the
documents, then if the dataset is balanced, then this word
won't be significant to any of the documents and would be 
assigned less weight.
There is some another word, which is frequent in only few
of the documents and not in others, then this word is significant
to these docs and would be assigned higher weights.
This is what this method does, for some word, it calculates the 
overall frequency and multiplies it by the inverse frequency of 
the word overall documents.
'''

# stop_words = 'english' : ignores common english words.
tfidf = TfidfVectorizer(stop_words = 'english')
train_x_vector = tfidf.fit_transform(train_x)
train_x_vector

test_x_vector = tfidf.transform(test_x)

svc = SVC(C = 1, kernel = 'linear')
svc.fit(train_x_vector, train_y)

print(svc.predict(tfidf.transform(['A good movie'])))
print(svc.predict(tfidf.transform(['An excellent movie'])))
print(svc.predict(tfidf.transform(['I did not like this movie at all'])))

svc.score(test_x_vector, test_y)

'''
[[true positive, false positive]
 [false negative, true negative]]
'''
conf_mat = confusion_matrix(test_y, 
                            svc.predict(test_x_vector), 
                            labels=['positive', 'negative'])

print(conf_mat)

'''
for some class A:
precision: predicted correct labels of class A / total predicted labels of class A
recall: predicted correct labels of class A / actual correct labels of class A
f1 score: harmonic mean of precision and recall.
'''
f1_score(test_y, svc.predict(test_x_vector),
         labels=['positive', 'negative'],
         average=None)

