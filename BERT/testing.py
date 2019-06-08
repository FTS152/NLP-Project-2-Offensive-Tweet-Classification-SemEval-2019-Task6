import pandas as pd
import numpy as np
import torch
import os,sys
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from run_classifier import *


#predict test set
def predict(model, tokenizer, examples, label_list, eval_batch_size=64):
    max_seq_length = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    eval_examples = examples
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, output_mode='classification')
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    res = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        res.extend(logits.argmax(-1))
        nb_eval_steps += 1

    return res

if sys.argv[2] == 'cased':
    VOCAB = './bert_cased/vocab.txt'
    MODEL = './bert_cased'
else:
    VOCAB = './bert_uncased/vocab.txt'
    MODEL = './bert_uncased'

if sys.argv[1] == 'A':
    label = pd.read_csv('project2_data/labels-levela.csv',index_col=0,header=None)
    test = pd.read_csv('./project2_data/testset-levela.tsv', sep='\t',index_col='id')
    test_examples = [InputExample('test', row.tweet, label='OFF') for row in test.itertuples()]
    label_list = ['OFF', 'NOT']
    model = BertForSequenceClassification.from_pretrained(MODEL,
              cache_dir='output',
              num_labels = 2)
elif sys.argv[1] == 'B':
    label = pd.read_csv('project2_data/labels-levelb.csv',index_col=0,header=None)
    test = pd.read_csv('./project2_data/testset-levelb.tsv', sep='\t',index_col='id')
    test_examples = [InputExample('test', row.tweet, label='UNT') for row in test.itertuples()]
    label_list = ['UNT', 'TIN']
    model = BertForSequenceClassification.from_pretrained(MODEL,
              cache_dir='output',
              num_labels = 2)
elif sys.argv[1] == 'C':
    label = pd.read_csv('project2_data/labels-levelc.csv',index_col=0,header=None)
    test = pd.read_csv('./project2_data/testset-levelc.tsv', sep='\t',index_col='id')
    test_examples = [InputExample('test', row.tweet, label='IND') for row in test.itertuples()]
    label_list = ['IND', 'GRP', 'OTH']
    model = BertForSequenceClassification.from_pretrained(MODEL,
              cache_dir='output',
              num_labels = 3)

tokenizer = BertTokenizer.from_pretrained(VOCAB)
model.load_state_dict(torch.load('./BERT/bert_task'+str(sys.argv[1])+str(sys.argv[2])+'.pkl'))

res = predict(model, tokenizer, test_examples, label_list)


result = []
if sys.argv[1] == 'A':
    for i in res:
        if i == 0:
            result.append('OFF')
        else:
            result.append('NOT')
elif sys.argv[1] == 'B':
    for i in res:
        if i == 0:
            result.append('UNT')
        else:
            result.append('TIN')
elif sys.argv[1] == 'C':
    for i in res:
        if i == 0:
            result.append('IND')
        elif i == 1:
            result.append('GRP')
        else:
            result.append('OTH')

from sklearn.metrics import f1_score
print('test accuracy: ',f1_score(label[1], result, average='macro'))

submission = pd.DataFrame({'id':label.index,'label':result})
submission.to_csv('./BERT/submission_task'+str(sys.argv[1])+str(sys.argv[2])+'.csv', index=False)

