import spacy
from spacy import displacy
from collections import Counter
import pandas as pd

nlp = spacy.load('en_core_web_sm')


data = pd.read_csv('./uncased_processed_train.csv')

correct = 0
incorrect = 0

orgs = set(['NORP', 'ORG'])
peps = set([])
for i, d in data.iterrows():
    if pd.isna(d['subtask_c']):
        continue
    tweet = nlp(d['tweet'])
    tag = 'IND'
    for w in tweet.doc.ents:
        if w.label_ in orgs:
            tag = 'GRP'

    if tag == d['subtask_c']:
        correct += 1
    else:
        incorrect += 1

        
    
print(f'Correct :\t{correct}\nIncorrect:\t{incorrect}\nPercentage:\t{correct / (correct + incorrect) * 100:.5f}%')

