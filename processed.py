import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

stop_words = set(stopwords.words('english'))
lematizer = WordNetLemmatizer()


def clean(tweet):
    tweet = tweet.lower()
    tweet = re.sub("\d+", " ", tweet)
    tweet = word_tokenize(tweet)
    tweet = [i for i in tweet if not i in stop_words]
    tweet = [lematizer.lemmatize(i) for i in tweet]
    return ' '.join(tweet)

def clean_cased(tweet):
    tweet = re.sub("\d+", " ", tweet)
    tweet = word_tokenize(tweet)
    tweet = [i for i in tweet if not i in stop_words]
    tweet = [lematizer.lemmatize(i) for i in tweet]
    return ' '.join(tweet)

data = pd.read_csv('./project2_data/olid-training-v1.0.tsv', delimiter='\t')
data['tweet'] = data['tweet'].apply(clean)
data.to_csv("uncased_processed_train.csv")


data = pd.read_csv('./project2_data/olid-training-v1.0.tsv', delimiter='\t')
data['tweet'] = data['tweet'].apply(clean_cased)
data.to_csv("cased_processed_train.csv")


tests = ['a', 'b', 'c']
for test in tests:
    data = pd.read_csv(f'./project2_data/testset-level{test}.tsv', delimiter='\t')
    lab = pd.read_csv(f'./project2_data/labels-level{test}.csv', header= None)
    data[f'subtask_{test}'] = lab[1]

    data['tweet'] = data['tweet'].apply(clean)
    data.to_csv(f"uncased_processed_test{test}.csv")
