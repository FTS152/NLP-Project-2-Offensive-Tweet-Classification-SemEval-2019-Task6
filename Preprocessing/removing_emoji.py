import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df_train = pd.read_csv('./uncased_processed_train.csv', index_col='id')
df_testa = pd.read_csv('./uncased_processed_testa.csv', index_col='id')
df_testb = pd.read_csv('./uncased_processed_testb.csv', index_col='id')
df_testc = pd.read_csv('./uncased_processed_testc.csv', index_col='id')

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_data_emoji(df):
    for index, row in df.iterrows():
        # print(remove_emoji(row['tweet']))
        df.loc[index, 'tweet'] = remove_emoji(row['tweet'])

def subtaska_label(df):
    for index, row in df.iterrows():
        if (row['subtask_a'] == 'OFF'):
            df.loc[index, 'subtask_a'] = 1
        else:
            df.loc[index, 'subtask_a'] = 0

def subtaskb_label(df):
    for index, row in df.iterrows():
        if (row['subtask_b'] == 'TIN'):
            df.loc[index, 'subtask_b'] = 1
        elif (row['subtask_b'] == 'UNT'):
            df.loc[index, 'subtask_b'] = 0

def subtaskc_label(df):
    for index, row in df.iterrows():
        if (row['subtask_c'] == 'IND'):
            df.loc[index, 'subtask_c'] = 0
        elif (row['subtask_c'] == 'GRP'):
            df.loc[index, 'subtask_c'] = 1
        elif (row['subtask_c'] == 'OTH'):
            df.loc[index, 'subtask_c'] = 2


remove_data_emoji(df_train)
subtaska_label(df_train)
subtaskb_label(df_train)
subtaskc_label(df_train)
df_train = df_train.drop('no', 1)

remove_data_emoji(df_testa)
subtaska_label(df_testa)
df_testa = df_testa.drop('no', 1)

remove_data_emoji(df_testb)
subtaskb_label(df_testb)
df_testb = df_testb.drop('no', 1)

remove_data_emoji(df_testc)
subtaskc_label(df_testc)
df_testc = df_testc.drop('no', 1)

df_train.to_csv('./uncased_noemoji_train.csv')
df_testa.to_csv('./uncased_noemoji_testa.csv')
df_testb.to_csv('./uncased_noemoji_testb.csv')
df_testc.to_csv('./uncased_noemoji_testc.csv')


