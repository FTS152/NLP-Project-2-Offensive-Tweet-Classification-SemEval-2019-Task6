
#Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

import re
import nltk
nltk.download('punkt', 'stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#%%
def _take_data_to_shower(tweet):
    noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']

    for noise in noises:
        tweet = tweet.replace(noise, '')

    return re.sub(r'[^a-zA-Z]', ' ', tweet)


def _tokenize(tweet):
    lower_tweet = tweet.lower()
    return word_tokenize(lower_tweet)

def _remove_stop_words(tokens):
    clean_tokens = []
    stopWords = set(stopwords.words('english'))
    for token in tokens:
        if token not in stopWords:
            if token.replace(' ', '') != '':
                if len(token) > 1:
                    clean_tokens.append(token)
    return clean_tokens

def _stem_and_lem(tokens):
    clean_tokens = []
    for token in tokens:
        token = wordnet_lemmatizer.lemmatize(token)
        token = lancaster_stemmer.stem(token)
        if len(token) > 1:
            clean_tokens.append(token)
    return clean_tokens


def _get_vectors(vectors, labels, keyword):
    if len(vectors) != len(labels):
        print("Unmatching sizes!")
        return
    result = list()
    for vector, label in zip(vectors, labels):
        if label == keyword:
            result.append(vector)
    return result

def Load(subtask = 'a'):
    print("Reading Dataset...")
    train_data = pd.read_csv('./project2_data/olid-training-v1.0.tsv', sep='\t', header=0)
    test_data = pd.read_csv('./project2_data/join_Test.csv', header=0)
 
    train_tweets = train_data[["tweet"]]
    train_subtask_a_labels = train_data[["subtask_a"]]
    train_subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
    train_subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["subtask_c"]]
    train_tweets = train_tweets.assign(tag='train')

    test_tweets = test_data[["tweet"]]
    test_subtask_a_labels = test_data[["subtask_a"]]
    test_subtask_b_labels = test_data.query("subtask_a == 'OFF'")[["subtask_b"]]
    test_subtask_c_labels = test_data.query("subtask_b == 'TIN'")[["subtask_c"]]
    test_tweets = test_tweets.assign(tag='test')

    tweets = train_tweets.append(test_tweets)
    clean_tweets = copy.deepcopy(tweets)

    tqdm.pandas(desc="Cleaning Data Phase I...")
    clean_tweets['tweet'] = tweets['tweet'].progress_apply(_take_data_to_shower)

    tqdm.pandas(desc="Tokenizing Data...")
    clean_tweets['tokens'] = clean_tweets['tweet'].progress_apply(_tokenize)

    tqdm.pandas(desc="Cleaning Data Phase II...")
    clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(_remove_stop_words)

    tqdm.pandas(desc="Stemming And Lemmatizing")
    clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(_stem_and_lem)

    train_vectors_a = clean_tweets.query("tag == 'train'")['tweet'].tolist() # Numerical Vectors A
    train_labels_a = train_subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

    test_vectors_a = clean_tweets.query("tag == 'test'")['tweet'].tolist() # Numerical Vectors A
    test_labels_a = test_subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

    if subtask == 'a':
        return (pd.DataFrame(train_vectors_a, columns=['tweet']).assign(label=train_labels_a), pd.DataFrame(test_vectors_a, columns=['tweet']).assign(label = test_labels_a))

    train_vectors_b = _get_vectors(train_vectors_a, train_labels_a, "OFF") # Numerical Vectors B
    train_labels_b = train_subtask_b_labels['subtask_b'].values.tolist() # Subtask A Labels

    test_vectors_b = _get_vectors(test_vectors_a, test_labels_a, "OFF") # Numerical Vectors B
    test_labels_b = test_subtask_b_labels['subtask_b'].values.tolist() # Subtask A Labels

    if subtask == 'b':
        return (pd.DataFrame(train_vectors_b, columns=['tweet']).assign(label=train_labels_b), pd.DataFrame(test_vectors_b, columns=['tweet']).assign(label = test_labels_b))

    train_vectors_c = _get_vectors(train_vectors_b, train_labels_b, "TIN") # Numerical Vectors C
    train_labels_c = train_subtask_c_labels['subtask_c'].values.tolist() # Subtask A Labels

    test_vectors_c = _get_vectors(test_vectors_b, test_labels_b, "TIN") # Numerical Vectors C
    test_labels_c = test_subtask_c_labels['subtask_c'].values.tolist() # Subtask A Labels

    if subtask == 'c':
        return (pd.DataFrame(train_vectors_c, columns=['tweet']).assign(label=train_labels_c), pd.DataFrame(test_vectors_c, columns=['tweet']).assign(label = test_labels_c))

    raise ValueError(f"Incorrect subtask type {subtask}")