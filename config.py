import os
import torch
import time

data_dir = os.getcwd() + '/data/'
train_dir = data_dir + 'char_ner_train.npz'
test_dir = data_dir + 'evaluation_public.npz'
pretrain_dir = data_dir + 'char_ner_pretrain.npz'
files = ['train', 'test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
pretrain_model = 'pretrained_bert_models/bert-pretrain-chinese'
model_dir = os.getcwd() + '/experiments/clue/' + time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
os.mkdir(model_dir)
test_model = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + '/train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 16
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['address', 'book', 'company', 'game', 'government',
          'movie', 'name', 'organization', 'position', 'scene']


label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    "I-address": 11,
    "I-book": 12,
    "I-company": 13,
    'I-game': 14,
    'I-government': 15,
    'I-movie': 16,
    'I-name': 17,
    'I-organization': 18,
    'I-position': 19,
    'I-scene': 20,
    "S-address": 21,
    "S-book": 22,
    "S-company": 23,
    'S-game': 24,
    'S-government': 25,
    'S-movie': 26,
    'S-name': 27,
    'S-organization': 28,
    'S-position': 29,
    'S-scene': 30
}

labels = ['O', 'B-GPE', 'M-GPE', 'E-GPE', 'B-PER', 'M-PER', 'E-PER', 'B-LOC', 'M-LOC',\
     'E-LOC', 'B-ORG', 'M-ORG', 'E-ORG', 'S-GPE', 'S-LOC', 'S-PER', 'S-ORG']

label2id = {_label: _id for _id, _label in enumerate(labels)}
id2label = {_id: _label for _label, _id in list(label2id.items())}
