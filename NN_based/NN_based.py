import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import keras
from keras import regularizers
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras import Input
from keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, Adagrad
from sklearn.metrics import f1_score
import keras.backend as K
import tensorflow as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

TRAIN_CSV_PATH = '../uncased_noemoji_train.csv'
TEST_CSV_PATH_A = '../uncased_noemoji_testa.csv'
TEST_CSV_PATH_B = '../uncased_noemoji_testb.csv'
TEST_CSV_PATH_C = '../uncased_noemoji_testc.csv'
TRAIN_LABEL_PATH = './label.npy'

TRAIN_CSV_PATH = '../uncased_noemoji_train.csv'
TEST_CSV_PATH = '../uncased_noemoji_testa.csv'
TRAIN_LABEL_PATH = './label.npy'

weight_decay = 1e-4
MAX_NUM_WORDS = 100000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 100
DROPOUT = 0.4
BATCH = 256
NUM_FILTERS = 16
# import data
df_traina = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
df_trainb = df_traina.dropna(subset=['subtask_b'])
df_trainc = df_traina.dropna(subset=['subtask_c'])
df_testa = pd.read_csv(TEST_CSV_PATH_A, index_col='id')
df_testb = pd.read_csv(TEST_CSV_PATH_B, index_col='id')
df_testc = pd.read_csv(TEST_CSV_PATH_C, index_col='id')
corpus = np.concatenate((df_traina['tweet'].as_matrix(), df_testa['tweet'].as_matrix()), axis=0)

def load_data():
    # convert data to np array and select right label columns
    X_traina = df_traina['tweet'].as_matrix()
    Y_traina = df_traina['subtask_a'].as_matrix()
    X_trainb = df_trainb['tweet'].as_matrix()
    Y_trainb = df_trainb['subtask_b'].as_matrix()
    X_trainc = df_trainc['tweet'].as_matrix()
    Y_trainc = df_trainc['subtask_c'].as_matrix()
    
    X_testa = df_testa['tweet'].as_matrix()
    Y_testa = df_testa['subtask_a'].as_matrix()
    X_testb = df_testb['tweet'].as_matrix()
    Y_testb = df_testb['subtask_b'].as_matrix()
    X_testc = df_testc['tweet'].as_matrix()
    Y_testc = df_testc['subtask_c'].as_matrix()
    return X_traina, Y_traina, X_trainb, Y_trainb, X_trainc, Y_trainc, X_testa, Y_testa, X_testb, Y_testb, X_testc, Y_testc


def data_preprocess(X_train, Y_train, X_test, Y_test, isSubtaskC):
    tokenizer = Tokenizer(num_words = MAX_NUM_WORDS, split=' ')
    tokenizer.fit_on_texts(corpus)
    # map sentences to sequences of numbers (word index)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    # Zero Padding
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    # Label One Hot Encoding
    if isSubtaskC == True:
        Y_train = to_categorical(Y_train)
    # split validation data
    VALID_SIZE = int (X_train.shape[0] / 10)
    X_train, X_valid = X_train[:-VALID_SIZE], X_train[-VALID_SIZE:]
    Y_train, Y_valid = Y_train[:-VALID_SIZE], Y_train[-VALID_SIZE:]
    return X_train, X_valid, Y_train, Y_valid, X_test, Y_test

def getThreshold(X_valid, Y_valid, model):
    # choose best threshold using validation data
    bestThreshold = 0
    bestScore = 0
    threshold = 0
    while threshold <= 1:
        pred = model.predict(X_valid)
        result = np.zeros(pred.shape[0])
        for i in range(pred.shape[0]):
            if pred[i] > threshold:
                result[i] = 1
            else:
                result[i] = 0
        if f1_score(Y_valid, result, average='macro') > bestScore:
            bestScore = f1_score(Y_valid, result, average='macro')
            bestThreshold = threshold
        threshold = threshold + 0.01
    return bestThreshold


def CNN(X_train, Y_train, X_valid, Y_valid, isSubtaskC):
    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Conv1D(NUM_FILTERS, 5, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    if isSubtaskC == True:
        model.add(Dense(units = 3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
        EPOCHS = 20
    else:
        model.add(Dense(units = 1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1])
        EPOCHS = 5
    model.summary()
    model.fit(X_train, Y_train,  validation_data=(X_valid, Y_valid), epochs=EPOCHS, batch_size=BATCH)
    return model

def BiLSTM(X_train, Y_train, X_valid, Y_valid, isSubtaskC):
    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Bidirectional(LSTM(64, return_sequences=False, dropout = DROPOUT)))
    model.add(Dense(32, activation='relu'))
    if isSubtaskC == True:
        model.add(Dense(units = 3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
        EPOCHS = 20
    else:
        model.add(Dense(units = 1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1])
        EPOCHS = 5
    model.summary()
    model.fit(X_train, Y_train,  validation_data=(X_valid, Y_valid), epochs=EPOCHS, batch_size=BATCH)
    return model

def BiGRU(X_train, Y_train, X_valid, Y_valid, isSubtaskC):
    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Bidirectional(GRU(64, return_sequences=False, dropout = DROPOUT)))
    model.add(Dense(32, activation='relu'))
    if isSubtaskC == True:
        model.add(Dense(units = 3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
        EPOCHS = 20
    else:
        model.add(Dense(units = 1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1])
        EPOCHS = 5
    model.summary()
    model.fit(X_train, Y_train,  validation_data=(X_valid, Y_valid), epochs=EPOCHS, batch_size=BATCH)
    return model
    
def predict(X_valid, Y_valid, X_test, Y_test, model, isSubtaskC):
    pred = model.predict(X_test)
    result = np.zeros(pred.shape[0])
    if isSubtaskC == True:
        for i in range(pred.shape[0]):
            result[i] = int(np.argmax(pred[i]))
    else:
        threshold = getThreshold(X_valid, Y_valid, model)
        print("threshold:", threshold)
        for i in range(pred.shape[0]):
            if pred[i] > threshold:
                result[i] = 1
            else:
                result[i] = 0
        
    print('test accuracy: ',f1_score(Y_test, result, average='macro'))
    return

    


def main():
    X_traina, Y_traina, X_trainb, Y_trainb, X_trainc, Y_trainc, X_testa, Y_testa, X_testb, Y_testb, X_testc, Y_testc = load_data()
    X_train, X_valid, Y_train, Y_valid, X_test, Y_test = data_preprocess(X_traina, Y_traina, X_testa, Y_testa, False)
    modelCNN = CNN(X_train, Y_train, X_valid, Y_valid, False)
    predict(X_valid, Y_valid, X_test, Y_test, modelCNN, False)
    modelBiLSTM = BiLSTM(X_train, Y_train, X_valid, Y_valid, False)
    predict(X_valid, Y_valid, X_test, Y_test, modelBiLSTM, False)
    modelBiGRU = BiGRU(X_train, Y_train, X_valid, Y_valid, False)
    predict(X_valid, Y_valid, X_test, Y_test, modelBiGRU, False)
    

    X_train, X_valid, Y_train, Y_valid, X_test, Y_test = data_preprocess(X_trainb, Y_trainb, X_testb, Y_testb, False)
    modelCNN = CNN(X_train, Y_train, X_valid, Y_valid, False)
    predict(X_valid, Y_valid, X_test, Y_test, modelCNN, False)
    modelBiLSTM = BiLSTM(X_train, Y_train, X_valid, Y_valid, False)
    predict(X_valid, Y_valid, X_test, Y_test, modelBiLSTM, False)
    modelBiGRU = BiGRU(X_train, Y_train, X_valid, Y_valid, False)
    predict(X_valid, Y_valid, X_test, Y_test, modelBiGRU, False)
    
    X_train, X_valid, Y_train, Y_valid, X_test, Y_test = data_preprocess(X_trainc, Y_trainc, X_testc, Y_testc, True)
    modelCNN = CNN(X_train, Y_train, X_valid, Y_valid, True)
    predict(X_valid, Y_valid, X_test, Y_test, modelCNN, True)
    modelBiLSTM = BiLSTM(X_train, Y_train, X_valid, Y_valid, True)
    predict(X_valid, Y_valid, X_test, Y_test, modelBiLSTM, True)
    modelBiGRU = BiGRU(X_train, Y_train, X_valid, Y_valid, True)
    predict(X_valid, Y_valid, X_test, Y_test, modelBiGRU, True)


if __name__ == "__main__":
    main()