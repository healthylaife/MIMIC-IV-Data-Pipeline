import pandas as pd
import pickle
import numpy as np
import tqdm
import os
import importlib
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

class BEHRT_models():
    def __init__(self,data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag):
        self.data_icu=data_icu
        if self.data_icu:
            self.id='stay_id'
        else:
            self.id='hadm_id'
        self.diag_flag,self.proc_flag,self.out_flag,self.chart_flag,self.med_flag,self.lab_flag=diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag
        #self.tokenization()
        
    def tokenize_dataset(self,labs_input, cond_input, demo_input, labels, vocab, demo_vocab, ins_vocab, gender_vocab):
        tokenized_src = []
        tokenized_gender = []
        tokenized_ethni = []
        tokenized_ins = []
        tokenized_age = []
        tokenized_labels = []
        idx = 0

        print("STARTING TOKENIZATION.")

        for patient, group in tqdm.tqdm(labs_input.groupby(self.id)):
            tokenized_src.append([])
            tokenized_src[idx].append(vocab["token2idx"]['CLS'])
            for row in cond_input[cond_input[self.id] == patient].itertuples(index=None):
                for key, value in row._asdict().items():
                    if value == '1':
                        tokenized_src[idx].append(vocab["token2idx"][key])
            tokenized_src[idx].append(vocab["token2idx"]['SEP'])
            for lab in group.itertuples(index=None):
                for col in lab:
                    if not isinstance(col, float):
                        tokenized_src[idx].append(vocab["token2idx"][col])
                tokenized_src[idx].append(vocab["token2idx"]['SEP'])
            tokenized_src[idx][-1] = vocab["token2idx"]['SEP']
            if len(tokenized_src[idx]) >= 512:
                tokenized_src.pop()
            else:
                gender = gender_vocab[demo_input[demo_input[self.id] == patient].iloc[0, 1]]
                ethnicity = demo_vocab[demo_input[demo_input[self.id] == patient].iloc[0, 2]]
                insurance = ins_vocab[demo_input[demo_input[self.id] == patient].iloc[0, 3]]
                age = demo_input[demo_input[self.id] == patient].iloc[0, 0]
                tokenized_gender.append([gender] * len(tokenized_src[idx]))
                tokenized_ethni.append([ethnicity] * len(tokenized_src[idx]))
                tokenized_ins.append([insurance] * len(tokenized_src[idx]))
                tokenized_age.append([age] * len(tokenized_src[idx]))
                tokenized_labels.append(labels[labels[self.id] == patient].iloc[0, 1])
                idx += 1

        print("FINISHED TOKENIZATION. \n")
        return pd.DataFrame(tokenized_src), pd.DataFrame(tokenized_gender), pd.DataFrame(tokenized_ethni), pd.DataFrame(tokenized_ins), pd.DataFrame(tokenized_age), pd.DataFrame(tokenized_labels)


    def tokenize(self):
        labs_list = []
        demo_list = []
        cond_list = []
        labels =  pd.read_csv('./data/csv/'+'labels.csv')
        first = True
        #labels = labels.iloc[:1, :]
        print("STARTING READING FILES.")
        for hadm in tqdm.tqdm(labels.itertuples(), total = labels.shape[0]):
            labs = pd.read_csv('./data/csv/' + str(hadm[1]) + '/dynamic.csv')
            demo = pd.read_csv('./data/csv/' + str(hadm[1]) + '/demo.csv')
            cond = pd.read_csv('./data/csv/' + str(hadm[1]) + '/static.csv')
            if first:
                condVocab_l = cond.iloc[0: , :].values.tolist()[0]
                first = False
            labs = labs.iloc[1: , :]
            cond = cond.iloc[1: , :]

            labs[self.id] = hadm[1]
            demo[self.id] = hadm[1]
            cond[self.id] = hadm[1]

            labs_list += labs.values.tolist()
            demo_list += demo.values.tolist()
            cond_list += cond.values.tolist()

        print("FINISHED READING FILES. \n")
        labs_list = pd.DataFrame(labs_list)
        demo_list = pd.DataFrame(demo_list)
        cond_list = pd.DataFrame(cond_list, columns=condVocab_l + [self.id])
        labs_list = labs_list.rename(columns={labs_list.columns.to_list()[-1]: self.id})
        demo_list = demo_list.rename(columns={demo_list.columns.to_list()[-1]: self.id})

        labs_list = pd.DataFrame(labs_list)
        demo_list = pd.DataFrame(demo_list)
        cond_list = pd.DataFrame(cond_list, columns=condVocab_l + [self.id])

        labs_list = labs_list.rename(columns={labs_list.columns.to_list()[-1]: self.id})
        demo_list = demo_list.rename(columns={demo_list.columns.to_list()[-1]: self.id})

        labs_list.replace(0, np.nan, inplace=True)

        '''    for col in labs_list.columns.to_list()[:-1]:
                if labs_list[col].nunique() < 2:
                    labs_list = labs_list.drop(columns=col)
        '''
        labs_codes = set()
        for col in labs_list.columns.to_list()[:-1]:
            labels_l = []
            if labs_list[col].nunique() > 1 :
                for i in range(len(pd.qcut(labs_list[col], 4, duplicates='drop', retbins=True)[1]) - 1):
                    labels_l.append(str(col) + "_" + str(i))
                labs_codes.update(labels_l)
                labs_list[col] = pd.qcut(labs_list[col], 4, labels=labels_l, duplicates='drop')
            elif labs_list[col].nunique() == 1 :
                labs_list.loc[labs_list[labs_list[col] > 0][col].index, col] = "dyn_" + str(col)
                labs_codes.add("dyn_" + str(col))
        ethVocab = {}
        insVocab = {}
        condVocab = {'token2idx': {}, 'idx2token': {0: 'PAD', 1: 'CLS', 2: 'SEP'}}
        with open('./data/dict/ethVocab', 'rb') as fp:
            ethVocab_l = pickle.load(fp)
            for i in range(len(ethVocab_l)):
                ethVocab[ethVocab_l[i]] = i

        with open('./data/dict/insVocab', 'rb') as fp:
            insVocab_l = pickle.load(fp)
            for i in range(len(insVocab_l)):
                insVocab[insVocab_l[i]] = i

        for v in condVocab_l:
            condVocab['idx2token'][max(condVocab['idx2token']) + 1] = v
        genderVocab = {'M': 0, 'F': 1}

        for new_code in labs_codes:
            condVocab['idx2token'][max(condVocab['idx2token']) + 1] = new_code

        condVocab['idx2token'][max(condVocab['idx2token']) + 1] = 'UNK'
        condVocab['token2idx'] = {v: k for k, v in condVocab['idx2token'].items()}
        cond_list = cond_list.sort_values(by=self.id)
        labs_list = labs_list.reset_index()
        labs_list = labs_list.sort_values(by=[self.id, 'index'])
        labs_list = labs_list.drop(columns=['index'])
        demo_list = demo_list.sort_values(by=self.id)

        tokenized_src, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_age, tokenized_labels = self.tokenize_dataset(
            labs_list, cond_list, demo_list, labels, condVocab, ethVocab, insVocab, genderVocab)

        print("FINAL COHORT STATISTICS: ")
        print(str(len(tokenized_labels[tokenized_labels[0] == 1])) + " Positive samples.")
        print(str(len(tokenized_labels[tokenized_labels[0] == 0])) + " Negative samples.\n")

        print(str(len(tokenized_gender[tokenized_gender[0] == 1])) + " Female samples.")
        print(str(len(tokenized_gender[tokenized_gender[0] == 0])) + " Male samples.\n")

        ethVocab_reversed = {v: k for k, v in ethVocab.items()}
        for i in range(len(ethVocab_reversed)):
            print(str(len(tokenized_ethni[tokenized_ethni[0] == i])) + " " + ethVocab_reversed[i] + " samples.")
        print("\n")

        insVocab_reversed = {v: k for k, v in insVocab.items()}
        for i in range(len(insVocab_reversed)):
            print(str(len(tokenized_ins[tokenized_ins[0] == i])) + " " + insVocab_reversed[i] + " samples.")

        return tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels
