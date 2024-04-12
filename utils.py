import nltk
import numpy as np
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
nltk.download('stopwords')

def trim (tweet, hyperlink=True, retweet=True, hash=True):
    '''
    arguments
        tweet: Tweet to be processed
        hyperlink: boolean, True if you wish to remove the hyperlinks
        retweet: boolean, True if you wish to remove the phrase 'RT' from the starting
        hash: boolean, True if you wish to remove the hash symbols.
    returns:
        updated tweet after doing the trimming
    '''
    if hyperlink:
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    if retweet:
        tweet = re.sub(r'^RT[\s]', '', tweet)
    if hash:
        tweet = re.sub(r'#', '', tweet)
    return tweet

def tokenize (tweet):
    '''
    arguments:
        tweet: the tweet to be tokenized
    returns:
        tweet_tokens: the list of tokens in tweet
    '''
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    return tweet_tokens

def removeStopwordsAndPunctuation (tweet_tokens):
    '''
    arguments:
        tweet_tokens: the tweet in tokenized format
    returns:
        clean_tokens: tweet_tokens but w/o stopwords and punctuation
    '''
    stopwords_english = stopwords.words('english')
    clean_tokens = []
    for token in tweet_tokens:
        if token not in stopwords_english and token not in string.punctuation:
            clean_tokens.append(token)
    return clean_tokens

def stem (clean_tokens):
    '''
    arguments:
        clean_tokens: tweet in tokenized form without stopwords
    returns:
        stemmed_tokens: reduces tokens in clean_tokens to their corresponding stem
    '''
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in clean_tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)
    return stemmed_tokens

def sigmoid (z):
    '''
    arguments:
        z: can be a number or a numpy array
    returns:
        h: sigmoid(z)
    '''
    h = 1 / (1 + np.exp(-z))
    return h