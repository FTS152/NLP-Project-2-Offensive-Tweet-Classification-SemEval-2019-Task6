import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score

stop_words = set(stopwords.words('english'))
lematizer = WordNetLemmatizer()
profane = set(lematizer.lemmatize(line.strip()) for line in open('./project2_data/full-profane.csv'))

data = pd.read_csv('./uncased_processed_train.csv')
# data = pd.read_csv('./uncased_processed_testa.csv')

pred = []
orig = []
for i, d in data.iterrows():
    tweet = d['tweet'].split(' ')
    if not set(tweet).isdisjoint(profane):
        tag = 'OFF'
    else:
        tag = 'NOT'
    pred.append(tag)
    orig.append(d['subtask_a'])
    
print(f1_score(orig, pred, average='macro'))
