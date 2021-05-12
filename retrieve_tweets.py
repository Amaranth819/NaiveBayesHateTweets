import csv
import re
import langdetect as ld
import contractions
import tweepy
import string

'''
    Tweets processing
'''
def substitution(pattern, replace, tweet):
    """
        Use regular expression to substitute specific strings. For example:
        1. Remove non-letters: pattern = r"[^a-z]"
        2. Remove websites: pattern = r"http\S+"
        3. Remove tags starting with specific characters : pattern = r"[@#]\S+"
    """
    return re.sub(pattern, replace, tweet)

def remove_punctuations(tweet):
    """
        Remove all punctuations in a given tweet.
    """
    return tweet.translate(str.maketrans('', '', string.punctuation))

def is_english(tweet):
    """
        Check whether the tweet is in English. Use try-except block to avoid "No features" error.
    """
    try:
        return ld.detect(tweet) == 'en'
    except:
        return False

def remove_spaces(tweet):
    """
        Remove all spaces inside the given tweet (after processing).
    """
    return ' '.join(filter(lambda k: k != '', tweet.strip().split(' ')))

def to_lowercase(tweet):
    """
        Convert the letters in the tweet to lowercase.
    """
    return tweet.lower()

def expand_contractions(tweet):
    """
        Expand the contractions. 
        However, the expansions are not always correct. For example, when the input is "he's been looking for ...", the output will be "he is been looking for ...".
    """
    return contractions.fix(tweet)

def tokenize(tweet):
    """
        Tokenize a given sentence. For example, "How are you" -> ["How", "are", "you"].
    """
    return tweet.strip().split(' ')

def batch_generator(batch_size, *attrs):
    """
        Create an iterator, which returns a batch of data every time.
    """
    length = min(len(a) for a in attrs)
    curr_start = 0
    while curr_start < length:
        curr_end = min(curr_start + batch_size, length)
        yield (a[curr_start:curr_end] for a in attrs)
        curr_start = curr_end

def process_one_tweet(tweet):
    """
        The steps for processing a tweet. It includes:
        1. Expand the contractions.
        2. Remove the urls.
        3. Remove the tags.
        4. Remove the punctuations.
        5. Convert the letters in the tweet to lowercase.
        6. Remove the non-letter characters.
        7. Remove the spaces.
    """
    tweet = expand_contractions(tweet)
    tweet = substitution(r"http\S+", " ", tweet)
    tweet = substitution(r"[@#]\S+", " ", tweet)
    tweet = remove_punctuations(tweet)
    tweet = to_lowercase(tweet)
    tweet = substitution(r"[^a-z]", " ", tweet)
    tweet = remove_spaces(tweet)
    return tweet

'''
    CSV file I/O
'''
def read_raw_csv(csv_path):
    """
        Read the Covid-HATE tweet data file from http://claws.cc.gatech.edu/covid/#dataset.
        The tweets are clssified into: hate, counterhate, neutral, and other.
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        label_idx = {'hate' : 0, 'counterhate' : 1, 'neutral' : 2, 'other' : 3}
        tweet_ids, labels = [], []
        for item in reader:
            # Ignore the name of attributes.
            if reader.line_num == 1:
                continue

            tweet_ids.append(int(item[0])) # The first attribute is tweet_id.
            labels.append(label_idx[item[-1].lower()]) # The last attribute is label.

        return tweet_ids, labels

def write_csv(csv_path, data_list, overwrite = False):
    """
        A function for writing a list into a csv file.
        The format of data_list: [[attr0, attr1, ...], [attr0, attr1, ...], ...]
    """
    with open(csv_path, 'w' if overwrite else 'a', newline = '', encoding='UTF-8') as f:
        writer = csv.writer(f)
        for attr in data_list:
            writer.writerow([*attr])

def read_csv(csv_path):
    """
        A function for reading from a csv file.
        The format of output: [[attr0, attr1, ...], [attr0, attr1, ...], ...]
    """
    with open(csv_path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        return [[*item] for item in reader]

'''
    Twitter
'''
def authenticate():
    """
        Authentication.
    """
    CONSUMER_KEY = 'F0QvlupLAm5gnOXM8TFfrazTz'
    CONSUMER_SECRET = '7w4BiQWCQEQpWOgmFuBimByHlsssU6keOQWFYZRzFDPz3Te1c9'
    OAUTH_TOKEN = '962795342096773120-8pegTEGJOiR9afoPXR3MaOvrt7me12r'
    OAUTH_TOKEN_SECRET = '6ZbcPCPn3QhuzIBlcl3h9RwI8F150QkS4U3MhBikxYzSg'

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

    return api

def crawler(api, tweet_ids):
    """
        According to given tweet ids, retrieve the content of tweets.
    """
    status = api.statuses_lookup(tweet_ids)
    return [str(s.text) for s in status]


if __name__ == '__main__':
    """
        Retrieve the tweets.
    """
    api = authenticate()

    tweet_ids, labels = read_raw_csv('hate_Jul8_Aug7_with_date.csv')
    batch_size = 100
    for i, (ids, ls) in enumerate(batch_generator(batch_size, tweet_ids, labels)):
        tweets = crawler(api, ids)
        tweet_label_pairs = list(zip(ls, tweets))
        english_tweet_label_pairs = list(filter(lambda x: is_english(x[-1]), tweet_label_pairs))
        write_csv('english.csv', english_tweet_label_pairs)
        print('Finish processing %d tweets!' % ((i + 1) * batch_size))

    """
        Split the tweets into a training set and a testing set. The ratio is 9 : 1.
    """
    data = read_csv('english.csv')
    labels, tweets = list(zip(*data))
    processed_tweets = list(map(process_one_tweet, tweets))
    trainset_size = int(len(labels) * 0.9)
    write_csv('train.csv', list(zip(labels[:trainset_size], processed_tweets[:trainset_size])))
    write_csv('test.csv', list(zip(labels[trainset_size:], processed_tweets[trainset_size:])))