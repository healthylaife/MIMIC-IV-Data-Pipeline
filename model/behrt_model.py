import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import sys
import random
import pandas as pd
import pickle
import tqdm
import importlib

from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.gender_embeddings = nn.Embedding(config.gender_vocab_size, config.hidden_size)
        self.ethnicity_embeddings = nn.Embedding(config.ethni_vocab_size, config.hidden_size)
        self.ins_embeddings = nn.Embedding(config.ins_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, gender_ids=None, ethni_ids=None, ins_ids=None, seg_ids=None,
                posi_ids=None, age=True):

        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)
        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        gender_embed = self.gender_embeddings(gender_ids)
        ethnicity_embed = self.ethnicity_embeddings(ethni_ids)
        ins_embed = self.ins_embeddings(ins_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + gender_embed + ethnicity_embed + ins_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + gender_embed + ethnicity_embed + ins_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, gender_ids=None, ethni_ids=None, ins_ids=None, seg_ids=None,
                posi_ids=None, attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if gender_ids is None:
            gender_ids = torch.zeros_like(input_ids)
        if ethni_ids is None:
            ethni_ids = torch.zeros_like(input_ids)
        if ins_ids is None:
            ins_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, gender_ids, ethni_ids, ins_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForEHRPrediction(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForEHRPrediction, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, gender_ids=None, ethni_ids=None, ins_ids=None, seg_ids=None, posi_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, age_ids, gender_ids, ethni_ids, ins_ids, seg_ids, posi_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')
        self.gender_vocab_size = config.get('gender_vocab_size')
        self.ethni_vocab_size = config.get('ethni_vocab_size')
        self.ins_vocab_size = config.get('ins_vocab_size')
        self.number_output = config.get('number_output')

class TrainConfig(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        self.use_cuda = config.get('use_cuda')
        self.max_len_seq = config.get('max_len_seq')
        self.train_loader_workers = config.get('train_loader_workers')
        self.test_loader_workers = config.get('test_loader_workers')
        self.device = config.get('device')
        self.output_dir = config.get('output_dir')
        self.output_name = config.get('output_name')
        self.best_name = config.get('best_name')


class DataLoader(Dataset):
    def __init__(self, dataframe, max_len, code='code', age='age', labels='labels'):
        self.max_len = max_len
        self.code = dataframe[code]
        self.age = dataframe[age]
        self.labels = dataframe[labels]
        self.gender = dataframe["gender"]
        self.ethni = dataframe["ethni"]
        self.ins = dataframe["ins"]

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index]
        code = self.code[index]
        label = self.labels[index]
        gender = self.gender[index]
        ethni = self.ethni[index]
        ins = self.ins[index]

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0


        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len)
        gender = seq_padding(gender, self.max_len)
        ethni = seq_padding(ethni, self.max_len)
        ins = seq_padding(ins, self.max_len)

        # get position code and segment code
        code = seq_padding(code, self.max_len)
        position = position_idx(code)
        segment = index_seg(code)

        return torch.LongTensor(code), torch.LongTensor(age), torch.LongTensor(gender), torch.LongTensor(
            ethni), torch.LongTensor(ins), \
               torch.LongTensor(segment), torch.LongTensor(position), \
               torch.FloatTensor(mask), torch.FloatTensor(label)

    def __len__(self):
        return len(self.code)

SEP = 2
PAD = 0

def seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = PAD

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if i < token_len:
            seq.append(tokens[i])
        else:
            seq.append(symbol)
    return seq


def position_idx(tokens, symbol=SEP):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def index_seg(tokens, symbol=SEP):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
