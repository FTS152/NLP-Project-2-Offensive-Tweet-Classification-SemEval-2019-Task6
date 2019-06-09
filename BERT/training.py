import pandas as pd
import numpy as np
import torch
import os,sys
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

if sys.argv[2] == 'cased':
    VOCAB = './bert_cased/vocab.txt'
    MODEL = './bert_cased'
else:
    VOCAB = './bert_uncased/vocab.txt'
    MODEL = './bert_uncased'

#convert model to pytorch_model
if os.path.exists(MODEL+'/pytorch_model.bin') == False:
    from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(MODEL+'/bert_model.ckpt', MODEL+'/bert_config.json', MODEL+'/pytorch_model.bin')

#load data
train_data = pd.read_csv('./project2_data/olid-training-v1.0.tsv', sep='\t',index_col='id')
if sys.argv[1] == 'A':
    test = pd.read_csv('./project2_data/testset-levela.tsv', sep='\t',index_col='id')
    train = pd.DataFrame({'tweet':train_data['tweet'],'label':train_data['subtask_a']})
elif sys.argv[1] == 'B':
    test = pd.read_csv('./project2_data/testset-levelb.tsv', sep='\t',index_col='id')
    train = pd.DataFrame({'tweet':train_data['tweet'],'label':train_data['subtask_b']})
elif sys.argv[1] == 'C':
    test = pd.read_csv('./project2_data/testset-levelc.tsv', sep='\t',index_col='id')
    train = pd.DataFrame({'tweet':train_data['tweet'],'label':train_data['subtask_c']})
train = train[train['label'].isna()==False]

#split validation set
from collections import Counter
Counter(train.label)
from sklearn.model_selection \
    import train_test_split
VALIDATION_RATIO = 0.1
RANDOM_STATE = 9527
train, val= \
    train_test_split(
        train, 
        test_size=VALIDATION_RATIO, 
        random_state=RANDOM_STATE
)

from run_classifier import *

train_examples = [InputExample('train', row.tweet, label=row.label) for row in train.itertuples()]
val_examples = [InputExample('val', row.tweet, label=row.label) for row in val.itertuples()]

orginal_total = len(train_examples)

#model parameter (for 16GB RAM settings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
gradient_accumulation_steps = 1
train_batch_size = 64
eval_batch_size = 64
train_batch_size = train_batch_size // gradient_accumulation_steps
#cased model
if sys.argv[2] == 'cased':
    do_lower=False
output_dir = 'output'
bert_model = 'bert-base-uncased'
num_train_epochs = 6
num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
cache_dir = "model"
learning_rate = 2e-5
warmup_proportion = 0.1
max_seq_length = 64

if sys.argv[1] == 'A':
    test_examples = [InputExample('test', row.tweet, label='OFF') for row in test.itertuples()]
    label_list = ['OFF', 'NOT']
if sys.argv[1] == 'B':
    test_examples = [InputExample('test', row.tweet, label='UNT') for row in test.itertuples()]
    label_list = ['UNT', 'TIN']
if sys.argv[1] == 'C':
    test_examples = [InputExample('test', row.tweet, label='IND') for row in test.itertuples()]
    label_list = ['IND', 'GRP', 'OTH']

tokenizer = BertTokenizer.from_pretrained(VOCAB)
if sys.argv[1] == 'C':
    model = BertForSequenceClassification.from_pretrained(MODEL,
                  cache_dir=cache_dir,
                  num_labels = 3)
else:
    model = BertForSequenceClassification.from_pretrained(MODEL,
                  cache_dir=cache_dir,
                  num_labels = 2)

model.load_state_dict(torch.load('./BERT/bert_task'+str(sys.argv[1])+str(sys.argv[2])+'.pkl'))

model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    
# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

train_features = convert_examples_to_features(
    train_examples, label_list, max_seq_length, tokenizer, output_mode = "classification")
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

model.train()

for _ in trange(int(num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_step = len(train_data) // train_batch_size
    ten_percent_step = total_step // 5
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
        if step % ten_percent_step == 0:
            print("Fininshed: {:.2f}% ({}/{})".format(step/total_step*100, step, total_step))


torch.save(model.state_dict(), './BERT/bert_task'+str(sys.argv[1])+str(sys.argv[2])+'.pkl')
