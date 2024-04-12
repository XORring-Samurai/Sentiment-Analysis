import numpy as np 
import matplotlib.pyplot as plt
import nltk
import utils
from nltk.corpus import twitter_samples
import sklearn

nltk.download('twitter_samples')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
tweets = positive_tweets + negative_tweets

processed_tweets = []

for i, tweet in enumerate(tweets):
    tweet = utils.trim(tweet)
    tweet_tokens = utils.tokenize(tweet)
    clean_tokens = utils.removeStopwordsAndPunctuation(tweet_tokens)
    stemmed_tokens = utils.stem(clean_tokens)
    processed_tweets.append(stemmed_tokens)

labels = np.append(np.ones(5000), np.zeros(5000))

freqs = {}

for i, tweet in enumerate(processed_tweets):
    for token in tweet:
        pair = (token, labels[i])
        freqs[(token, labels[i])] = freqs.get((token, labels[i]), 0) + 1

X = np.zeros((10000, 3))
Y = labels
Y = np.reshape(Y, (len(labels), 1))

for i, tweet in enumerate(processed_tweets):
    X[i, 0] = 1
    for token in tweet:
        X[i, 1] = X[i, 1] + freqs.get((token, 1), 0)
        X[i, 2] = X[i, 2] + freqs.get((token, 0), 0)
    X[i, 1] = X[i, 1] / 10000
    X[i, 2] = X[i, 2] / 10000

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, 
                                                                            Y, 
                                                                            test_size=0.2, 
                                                                            stratify=Y,
                                                                            random_state=69)

num_iters=1500
learning_rate=1e-4
cost=0
weights=np.zeros((3, 1))
m=X_train.shape[0]

for iter in range(num_iters):
    Z = np.dot(X_train, weights)
    Y_hat = utils.sigmoid(Z)
    cost = -1/m * (np.dot(Y_train.T, np.log(Y_hat)) + np.dot((1 - Y_train.T), np.log(1 - Y_hat)))
    weights = weights - learning_rate / m * (np.dot(X_train.T, Y_train - Y_hat))
print(cost, weights)

preds = np.dot(X_test, weights)
preds = (preds >= 0.5).astype('int')
accuracy = np.sum(preds == Y_test) / Y_test.shape[0]
print(accuracy)



        


