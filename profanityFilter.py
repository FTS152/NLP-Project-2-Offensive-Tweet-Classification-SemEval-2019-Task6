import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from sklearnModels.loader import Load

stop_words = set(stopwords.words('english'))
lematizer = WordNetLemmatizer()
profane = set(lematizer.lemmatize(line.strip()) for line in open('./project2_data/full-profane.csv'))

'''
train, test = Load('a')
data = test
'''
# data = pd.read_csv('./uncased_processed_train.csv')
data = pd.read_csv('./uncased_processed_testa.csv')


pred = []
for d in data['tweet']:
    tweet = d.lower().split(' ')
    if not set(tweet).isdisjoint(profane):
        tag = 'OFF'
    else:
        tag = 'NOT'
    pred.append(tag)
    
# print(f1_score(data['label'], pred, average='macro'))
print(f1_score(data['subtask_a'], pred, average='macro'))
