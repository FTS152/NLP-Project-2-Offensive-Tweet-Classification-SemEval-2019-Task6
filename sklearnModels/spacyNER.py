import spacy
from spacy import displacy
from collections import Counter
import pandas as pd

from sklearn.metrics import f1_score
nlp = spacy.load('en_core_web_sm')


data = pd.read_csv('./uncased_processed_testc.csv')

correct = 0
incorrect = 0

orgs = set(['NORP', 'ORG'])

pred = []
for i, d in data.iterrows():
    if pd.isna(d['subtask_c']):
        continue
    tweet = nlp(d['tweet'])
    tag = 'IND'
    for w in tweet.doc.ents:
        if w.label_ in orgs:
            tag = 'GRP'

    pred.append(tag)

        
print(f1_score(data['subtask_c'], pred, average='macro'))    

