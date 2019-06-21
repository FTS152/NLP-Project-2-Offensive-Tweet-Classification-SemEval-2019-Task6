import pandas as pd
from nltk.stem import WordNetLemmatizer
lematizer = WordNetLemmatizer()

train = pd.read_csv('uncased_processed_train.csv')
words = {}
for i, d in train.iterrows():
    tweet = d['tweet']
    for w in set(tweet.split(' ')):
        if d['subtask_a'] == 'OFF':
            words[w] = words.get(w, 0) + 1
        else:
            words[w] = words.get(w, 0) - 1

keys = list(words.keys())

profane = set(lematizer.lemmatize(line.strip()) for line in open('./project2_data/full-profane.csv'))
for w in keys:
    if abs(words[w]) < 100 or w in profane:
        del words[w]

import pprint
print(pprint.pprint(words))        