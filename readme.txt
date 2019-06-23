required Python3 
To run this program, need to put training data and testing data in project2_data/
for BERT/ and stanfordNER.py, corresponding model file is required.

Preprocessing:
  joinTest.py: combine test data and test labels
  processed.py: generate cased and uncased data with lemmatization
  load.py: generate preprocessed data(tokenization, filter stopwords and non-alphanumeric symbols)
  removing_emoji.py: remove emojis

Domain_Knowledge_based:
  classify.py: used models: MultinomialNB, KNN, SVM, DecisionTree, RandomForest, LogisticRegression
  profanityFilter.py: detect offensive language by a list of common profanity words by Google
  spacyNER.py: NER tagger by SpaCy
  stanfordNER.py: NER tagger by stanford

NN_based:
  train model and do prediction by neural network methods
  used models: CNN, BiLSTM, BiGRU

BERT:
  training.py: train cased or uncased model
  testing.py: do prediction