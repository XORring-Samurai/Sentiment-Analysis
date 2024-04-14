import numpy as np 
import matplotlib.pyplot as plt
import nltk
import utils
from nltk.corpus import twitter_samples
import sklearn
import pandas as pd

nltk.download('twitter_samples')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
tweets = positive_tweets + negative_tweets

train_tweets = tweets[:4000] + tweets[5000:9000]
test_tweets = tweets[4000:5000] + tweets[9000:]
train_labels = np.append(np.ones(4000), np.zeros(4000))
test_labels = np.append(np.ones(1000), np.zeros(1000))

processed_tweets = []

for i, tweet in enumerate(train_tweets):
    tweet = utils.trim(tweet)
    tweet_tokens = utils.tokenize(tweet)
    clean_tokens = utils.removeStopwordsAndPunctuation(tweet_tokens)
    stemmed_tokens = utils.stem(clean_tokens)
    processed_tweets.append(stemmed_tokens)

freqs = {}

for i, tweet in enumerate(processed_tweets):
    for token in tweet:
        pair = (token, train_labels[i])
        freqs[(token, train_labels[i])] = freqs.get((token, train_labels[i]), 0) + 1

# ------------------- log-prior -------------------
D = len(processed_tweets)
D_pos = np.sum(train_labels).astype(int)
D_neg = D - D_pos
log_prior = np.log(D_pos) - np.log(D_neg)

# ------------------- log-likelihood -------------------
log_likelihood = {}
N_pos = N_neg = 0
vocab = set([tuple(tup)[0] for tup in freqs.keys()])
V = len(vocab)

for word in vocab:
    N_pos += freqs.get((word, 1), 0)
    N_neg += freqs.get((word, 0), 0)

for word in vocab:
    pos = freqs.get((word, 1), 0)
    neg = freqs.get((word, 0), 0)
    f_pos = (pos + 1) / (N_pos + V)
    f_neg = (neg + 1) / (N_neg + V)
    log_likelihood[word] = np.log(f_pos) - np.log(f_neg)

# ------------------ Confidence-ellipse -----------------
pos = []
neg = []
for i, tweet in enumerate(processed_tweets):
    log_pos = 0
    log_neg = 0
    for word in tweet:
        f_pos = (freqs.get((word, 1), 0) + 1) / (N_pos + V)
        f_neg = (freqs.get((word, 0), 0) + 1) / (N_neg + V)
        log_pos += np.log(f_pos)
        log_neg += np.log(f_neg)
    pos.append(log_pos)
    neg.append(log_neg)

data = pd.DataFrame({'tweet': processed_tweets, 
                     'positive': pos, 
                     'negative':neg, 
                     'sentiment':train_labels})
print(type(data.sentiment.iloc[0]))

fig, ax = plt.subplots(figsize=(10, 10))
color = ['r', 'g']
sentiment = ['negative', 'positive']
for i in range(2):
    ax.scatter(x=data[data.sentiment==i].positive, 
               y=data[data.sentiment==i].negative,
               color=color[i], label=sentiment[i])

plt.xlim([-200, 40])
plt.ylim([-200, 40])
plt.xlabel('positive')
plt.ylabel('negative')

edgecolor = ['black', 'orange']

for i in range(2):
    utils.confidence_ellipse(x=data[data.sentiment==i].positive,
                            y=data[data.sentiment==i].negative,
                            ax=ax,
                            n_std=2.0,
                            edgecolor=edgecolor[i])
    utils.confidence_ellipse(x=data[data.sentiment==i].positive,
                            y=data[data.sentiment==i].negative,
                            ax=ax,
                            n_std=3.0,
                            edgecolor=edgecolor[i])
plt.show()

# ------------------ on the Test-set --------------------

processed_test_tweets = []
for i, tweet in enumerate(test_tweets):
    tweet = utils.trim(tweet)
    tweet_tokens = utils.tokenize(tweet)
    clean_tokens = utils.removeStopwordsAndPunctuation(tweet_tokens)
    stemmed_tokens = utils.stem(clean_tokens)
    processed_test_tweets.append(stemmed_tokens)

Y_hat = np.zeros((len(test_labels)))
for i, tweet in enumerate(processed_test_tweets):
    score = log_prior
    for word in tweet:
        if word in log_likelihood:
            score += log_likelihood[word]
    Y_hat[i] = (score >= 0).astype(int)

accuracy = 1/len(test_labels) * np.sum(Y_hat == test_labels)
print(accuracy)