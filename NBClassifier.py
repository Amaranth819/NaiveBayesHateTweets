import numpy as np
import json
# import nltk
# from nltk.corpus import stopwords
from retrieve_tweets import read_csv, tokenize, process_one_tweet

# nltk.download('stopwords')

class NaiveBayesClassifier(object):
    """
        Implementation of Naive-Bayes classifier.

        Assume a sentence consists of a sequence of independent words (f1, f2, ..., fn). Then the probability of this sentence belonging to class S is:

            P(S|f1, f2, ..., fn) = ( P(S) * P(f1, f2, ..., fn|S) ) / P(f1, f2, ..., fn) = ( P(S) * P(f1|S) * P(f2|S) * ... * P(fn|S) ) / ( P(f1) * P(f2) * ... * P(fn) )

        The probabilities P(f1), P(f2), ... P(fn) do not depend on the class, so just ignore them.

        Apply log to P(S|f1, f2, ..., fn) and get an objective function L:

            L = log( P(S|f1, f2, ..., fn) ) = log( P(S) ) + log( P(f1|S) ) + log( P(f2|S) ) + ... + log( P(fn|S) )

        The goal is to find the class that maximize L (see NaiveBayesClassifier.predict()).
    """
    def __init__(self):
        self.corpus = {} # vocabulary: [hate_frequency, counterhate_frequency, neutral_frequency, other_frequency, vocabulary_frequency]
        self.label_token_frequency = {0 : 0, 1 : 0, 2 : 0, 3 : 0} # The frequency of vocabulary in each class
        self.label_prob = {0 : 0, 1 : 0, 2 : 0, 3 : 0} # The probability of each class

    def save(self, path):
        """
            Save the model in json.
        """
        with open(path, 'w') as f:
            content = [self.label_token_frequency, self.label_prob, self.corpus]
            json.dump(content, f) 

    def load(self, path):
        """
            Read from a json model file.
        """
        with open(path, 'r') as f:
            tmp_label_token_frequency, tmp_label_prob, self.corpus = json.load(f)
            self.label_token_frequency = {0 : 0, 1 : 0, 2 : 0, 3 : 0}
            self.label_prob = {0 : 0, 1 : 0, 2 : 0, 3 : 0}

            # The function json.load() causes the key of the dictionary to be type string. Convert them to type int.
            for k, v in tmp_label_token_frequency.items():
                self.label_token_frequency[int(k)] = v

            for k, v in tmp_label_prob.items():
                self.label_prob[int(k)] = v

    def train(self, csv_path):
        """
            Train the model.
        """
        # 1. Read the dataset.
        dataset = read_csv(csv_path)

        # 2. Count the frequency of tokens in each class.
        for label, tweet in dataset:
            if tweet.strip() != '':
                for token in tokenize(tweet):
                    if token != '':
                        label = int(label)
                        try:
                            self.corpus[token][label] += 1
                            self.corpus[token][-1] += 1
                        except KeyError:
                            self.corpus[token] = [0, 0, 0, 0, 0]
                            self.corpus[token][label] += 1
                            self.corpus[token][-1] += 1
                        self.label_token_frequency[label] += 1
                self.label_prob[label] += 1

        # 3. Calculate the probability of each class.
        total_label_frequency = sum([v for _, v in self.label_prob.items()])
        for label, f in self.label_prob.items():
            self.label_prob[label] = f / total_label_frequency

        # # Removing stopwords doesn't significantly improve the performance.
        # for w in stopwords.words('english'):
        #     try:
        #         self.corpus.pop(w)
        #     except:
        #         continue

    def prior_log_prob(self, label, token):
        """
            Compute the log prior probability P(fi | S) of the given token. 
            Add-1 smoothing.

                P(fi | S) = (The frequency of fi in class S + 1) / (The frequency of all tokens in class S + 1)
        """
        try:
            return np.log((self.corpus[token][label] + 1) / (self.label_token_frequency[label] + 1))
        except:
            # log((0 + 1) / (0 + 1))
            return 0

    def predict(self, tweet):
        """
            Predict the label of a given tweet.
        """
        # 1. Preprocess and tokenize.
        tweet = process_one_tweet(tweet)
        tokens = tokenize(tweet)

        # 2. Compute the objective function L for each class.
        res = np.zeros(4, dtype = np.float32)
        for label in [0, 1, 2, 3]:
            res[label] += np.log(self.label_prob[label])

        for token in tokens:
            for label in [0, 1, 2, 3]:
                res[label] += self.prior_log_prob(label, token)

        # 3. Return the class resulting in the maximum value.
        return np.argmax(res)


if __name__ == '__main__':
    model = NaiveBayesClassifier()
    
    # Train the model.
    # model.train('train.csv')
    # model.save('model.json')

    # Load the model.
    model.load('model.json')

    # Predict the tweets in the testing set and compute the accuracy.
    data = read_csv('test.csv')
    correct, total = 0, 0
    for label, tweet in data:
        pred = model.predict(tweet)
        correct += (pred == int(label))
        total += 1

    # Not filtering stopwords: 42.29%
    # After filtering stopwords: 42.78%
    print('Accuracy = %.4f' % (correct / total))