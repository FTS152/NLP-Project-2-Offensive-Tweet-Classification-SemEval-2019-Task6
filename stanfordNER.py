from nltk.tag import StanfordNERTagger
import pandas as pd

data = pd.read_csv('./uncased_processed_train.csv')
ner = StanfordNERTagger('./stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz', './stanford-ner-2018-10-16/stanford-ner.jar')

correct = 0
incorrect = 0
data['tweet'] = ner.tag_sents(data['tweet'].str.split(' '))


for i, d in data.iterrows():
    if pd.isna(d['subtask_c']):
        continue
    tweet = d['tweet']
    tag = 'OTH'
    for w in tweet:
        if w[1] == 'ORGANIZATION':
            tag = 'GRP'
        elif w[1] == 'PEOPLE':
            tag = 'IND'

    if tag == d['subtask_c']:
        correct += 1
    else:
        incorrect += 1

        
    
print(f'Correct :\t{correct}\nIncorrect:\t{incorrect}\nPercentage:\t{correct / (correct + incorrect) * 100:.5f}%')

