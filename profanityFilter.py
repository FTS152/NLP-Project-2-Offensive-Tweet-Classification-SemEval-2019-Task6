import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lematizer = WordNetLemmatizer()
profane = set(lematizer.lemmatize(line.strip()) for line in open('./project2_data/full-profane.csv'))

data = pd.read_csv('./uncased_processed_testa.csv')
correct = 0
incorrect = 0
for i, d in data.iterrows():
    tweet = d['tweet'].split(' ')
    if not set(tweet).isdisjoint(profane):
        tag = 'OFF'
    else:
        tag = 'NOT'

    if tag == d['subtask_a']:
        correct += 1
    else:
        incorrect += 1
        # if tag == 'NOT':
            # print(d['tweet'])

        
    
print(f'Correct :\t{correct}\nIncorrect:\t{incorrect}\nPercentage:\t{correct / (correct + incorrect) * 100:.5f}%')
