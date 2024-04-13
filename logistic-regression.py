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

X = np.zeros((8000, 3))
Y = train_labels
Y = np.reshape(Y, (len(train_labels), 1))

for i, tweet in enumerate(processed_tweets):
    X[i, 0] = 1
    for token in tweet:
        X[i, 1] = X[i, 1] + freqs.get((token, 1), 0)
        X[i, 2] = X[i, 2] + freqs.get((token, 0), 0)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
for i in range (1, 3):
    X[:, i] = (X[:, i] - mean[i]) / std[i]

fig = plt.figure(figsize=(10, 10))
x = X[:, 1]
y = X[:, 2]
colors = ['r', 'g']
plt.scatter(x, y, color=[colors[(int)(Y[i])] for i in range(Y.shape[0])])
# plt.show()
# This plot shows that you have a very clear-separation between the positive and the negative tweets.

num_iters=1500
learning_rate=1e-9
cost=0
weights=np.zeros((3, 1))
m=X.shape[0]
for iter in range(num_iters):
    Z = np.dot(X, weights)
    Y_hat = utils.sigmoid(Z)
    cost = -1/m * (np.dot(Y.T, np.log(Y_hat)) + np.dot((1 - Y.T), np.log(1 - Y_hat)))
    weights = weights - (learning_rate / m) * (np.dot(X.T, Y_hat - Y))
print(cost, weights)

# -----------------------------------------------------------------------------------

processed_test_tweets = []
for i, tweet in enumerate(test_tweets):
    tweet = utils.trim(tweet)
    tweet_tokens = utils.tokenize(tweet)
    clean_tokens = utils.removeStopwordsAndPunctuation(tweet_tokens)
    stemmed_tokens = utils.stem(clean_tokens)
    processed_test_tweets.append(stemmed_tokens)

X_t = np.zeros((2000, 3))
Y_t = test_labels
Y_t = np.reshape(Y_t, (len(test_labels), 1))
for i, tweet in enumerate(processed_test_tweets):
    X_t[i, 0] = 1
    for token in tweet:
        X_t[i, 1] = X_t[i, 1] + freqs.get((token, 1), 0)
        X_t[i, 2] = X_t[i, 2] + freqs.get((token, 0), 0)

for i in range (1, 3):
    X_t[:, i] = (X_t[:, i] - mean[i]) / std[i]

preds = utils.sigmoid(np.dot(X_t, weights))
preds = (preds >= 0.5).astype('int')
accuracy = np.sum(preds == Y_t) / Y_t.shape[0]
print(accuracy)



        


