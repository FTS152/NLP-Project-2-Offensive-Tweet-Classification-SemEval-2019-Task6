from nltk.tag import StanfordNERTagger
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from loader import Load

train, test = Load('c')

ner = StanfordNERTagger('./stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz', './stanford-ner-2018-10-16/stanford-ner.jar')

data = train

data['tweet'] = ner.tag_sents(data['tweet'].str.split(' '))


pred = []

for i, d in data.iterrows():
    tweet = d['tweet']
    tag = 'IND'
    for w in tweet:
        if w[1] == 'ORGANIZATION':
            tag = 'GRP'
        # elif w[1] == 'PEOPLE':
        #     tag = 'IND'

    pred.append(tag)

print(confusion_matrix(data['label'], pred))
print(f1_score(data['label'], pred, average='macro'))

