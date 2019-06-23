import pandas as pd

data = pd.read_csv(f'./project2_data/testset-levela.tsv', delimiter='\t', header=0)
for test in ['a', 'b', 'c']:
    lab = pd.read_csv(f'./project2_data/labels-level{test}.csv', header=None, names=['id', f'subtask_{test}'])
    data = data.join(lab.set_index('id'), on='id')

data.to_csv(f'./project2_data/join_test.csv')
