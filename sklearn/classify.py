
#%%

import warnings
warnings.filterwarnings('ignore')

#Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy


#%%
print("Reading Dataset...")
train_data = pd.read_csv('./project2_data/olid-training-v1.0.tsv', sep='\t', header=0)
test_data = pd.read_csv('./project2_data/join_Test.csv', header=0)
#%%
train_tweets = train_data[["tweet"]]
train_subtask_a_labels = train_data[["subtask_a"]]
train_subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
train_subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["subtask_c"]]
train_tweets = train_tweets.assign(tag='train')

train_size = train_tweets.shape[0]

test_tweets = test_data[["tweet"]]
test_subtask_a_labels = test_data[["subtask_a"]]
test_subtask_b_labels = test_data.query("subtask_a == 'OFF'")[["subtask_b"]]
test_subtask_c_labels = test_data.query("subtask_b == 'TIN'")[["subtask_c"]]
test_tweets = test_tweets.assign(tag='test')

tweets = train_tweets.append(test_tweets)
clean_tweets = copy.deepcopy(tweets)


#%%
##PREPROCESSING##


#%%
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
def take_data_to_shower(tweet):
    noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']

    for noise in noises:
        tweet = tweet.replace(noise, '')

    return re.sub(r'[^a-zA-Z]', ' ', tweet)


def tokenize(tweet):
    lower_tweet = tweet.lower()
    return word_tokenize(lower_tweet)


def remove_stop_words(tokens):
    clean_tokens = []
    stopWords = set(stopwords.words('english'))
    for token in tokens:
        if token not in stopWords:
            if token.replace(' ', '') != '':
                if len(token) > 1:
                    clean_tokens.append(token)
    return clean_tokens


def stem_and_lem(tokens):
    clean_tokens = []
    for token in tokens:
        token = wordnet_lemmatizer.lemmatize(token)
        token = lancaster_stemmer.stem(token)
        if len(token) > 1:
            clean_tokens.append(token)
    return clean_tokens


#%%
tqdm.pandas(desc="Cleaning Data Phase I...")
clean_tweets['tweet'] = tweets['tweet'].progress_apply(take_data_to_shower)

tqdm.pandas(desc="Tokenizing Data...")
clean_tweets['tokens'] = clean_tweets['tweet'].progress_apply(tokenize)

tqdm.pandas(desc="Cleaning Data Phase II...")
clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(remove_stop_words)

tqdm.pandas(desc="Stemming And Lemmatizing")
clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(stem_and_lem)

text_vector = clean_tweets['tokens'].tolist()


#%%
##EMBEDDING##


#%%
from sklearn.feature_extraction.text import TfidfVectorizer

def tfid(text_vector):
    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors
  
def get_vectors(vectors, labels, keyword):
    if len(vectors) != len(labels):
        print("Unmatching sizes!")
        return
    result = list()
    for vector, label in zip(vectors, labels):
        if label == keyword:
            result.append(vector)
    return result

clean_tweets['tweet'] = tfid(text_vector).tolist()
#%%
train_vectors_a = clean_tweets.query("tag == 'train'")['tweet'].tolist() # Numerical Vectors A
train_labels_a = train_subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

train_vectors_b = get_vectors(train_vectors_a, train_labels_a, "OFF") # Numerical Vectors B
train_labels_b = train_subtask_b_labels['subtask_b'].values.tolist() # Subtask A Labels

train_vectors_c = get_vectors(train_vectors_b, train_labels_b, "TIN") # Numerical Vectors C
train_labels_c = train_subtask_c_labels['subtask_c'].values.tolist() # Subtask A Labels

test_vectors_a = clean_tweets.query("tag == 'test'")['tweet'].tolist() # Numerical Vectors A
test_labels_a = test_subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

test_vectors_b = get_vectors(test_vectors_a, test_labels_a, "OFF") # Numerical Vectors B
test_labels_b = test_subtask_b_labels['subtask_b'].values.tolist() # Subtask A Labels

test_vectors_c = get_vectors(test_vectors_b, test_labels_b, "TIN") # Numerical Vectors C
test_labels_c = test_subtask_c_labels['subtask_c'].values.tolist() # Subtask A Labels




#%%
##CLASSIFING##


#%%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def classify(train_vectors, train_labels, test_vectors, test_labels, type="DT"):
    # Initialize Model
    classifier = None
    if(type=="MNB"):
        classifier = MultinomialNB(alpha=0.7)
        classifier.fit(train_vectors, train_labels)
    elif(type=="KNN"):
        classifier = KNeighborsClassifier(n_jobs=4)
        params = {'n_neighbors': [3,5,7,9], 'weights':['uniform', 'distance']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=-1, scoring='f1_macro')
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif(type=="SVM"):
        classifier = SVC(gamma='auto')
        classifier = GridSearchCV(classifier, {'C':[0.001, 0.01, 0.1, 1, 10]}, cv=3, n_jobs=-1, scoring='f1_macro')
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif(type=="DT"):
        classifier = DecisionTreeClassifier(max_depth=300, min_samples_split=5)
        params = {'criterion':['gini','entropy']}
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=-1, scoring='f1_macro')
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif(type=="RF"):
        classifier = RandomForestClassifier(max_depth=300, min_samples_split=5, n_jobs=-1)
        params = {'n_estimators': [n for n in range(150,300,50)], 'criterion':['gini','entropy'], }
        classifier = GridSearchCV(classifier, params, cv=3, n_jobs=-1, scoring='f1_macro')
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif(type=="LR"):
        classifier = LogisticRegression(multi_class='auto', solver='newton-cg',)
        classifier = GridSearchCV(classifier, {"C":np.logspace(-3,3,7), "penalty":["l2"]}, cv=3, n_jobs=-1, scoring='f1_macro')
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    else:
        print("Wrong Classifier Type!")
        return
    accuracy = f1_score(train_labels, classifier.predict(train_vectors), average='macro')
    print("Training Accuracy:", accuracy)
    test_predictions = classifier.predict(test_vectors)
    accuracy = f1_score(test_labels, test_predictions, average='macro')
    print("Test Accuracy:", accuracy)
    print("Confusion Matrix:", )
    print(confusion_matrix(test_labels, test_predictions))

#%%
for method, tasks in [\
    # 'SVM',\
    # ['MNB', [True, True, False]],\
    # ['KNN', [True, True, True]],\
    ['DT', [True, True, True]],\
    # ['RF', [True, True, True]],\
    ['LR', [True, True, True]]\
    ]:
    if tasks[0]:
        print(f"\nBuilding Model Subtask A with {method}...")
        classify(train_vectors_a, train_labels_a, test_vectors_a, test_labels_a, method) # {MNB, KNN, SVM, DT, RF, LR}

    if tasks[1]:
        print(f"\nBuilding Model Subtask B with {method}...")
        classify(train_vectors_b, train_labels_b, test_vectors_b, test_labels_b, method) # {MNB, KNN, SVM, DT, RF, LR}

    if tasks[2]:
        print(f"\nBuilding Model Subtask C with {method}...")
        classify(train_vectors_c, train_labels_c, test_vectors_c, test_labels_c, method) # {MNB, KNN, SVM, DT, RF, LR}


#%%


