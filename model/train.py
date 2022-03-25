#!/usr/bin/env python
# coding: utf-8

import pickle
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import numpy as np
import torch as T
import torch
import math
from sklearn import metrics
import torch.nn as nn
from torch import optim
import importlib
import torch.nn.functional as F
import import_ipynb
import model_utils
import evaluation
import parameters
from parameters import *
#import model as model
import mimic_model as model
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from pickle import dump,load
from sklearn.model_selection import train_test_split

#import torchvision.utils as utils
import argparse
from torch.autograd import Variable
from argparse import ArgumentParser
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

save_path = "saved_models/model.tar"
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

importlib.reload(model_utils)
import model_utils
importlib.reload(model)
import mimic_model as model
importlib.reload(parameters)
import parameters
from parameters import *
importlib.reload(evaluation)
import evaluation


class Model_Train():
    def __init__(self,diag_flag,proc_flag,out_flag,chart_flag,med_flag):
        self.nBatches,self.cond_vocab_size,self.proc_vocab_size,self.med_vocab_size,self.out_vocab_size,self.chart_vocab_size,self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size=model_utils.init(args.batch_size,diag_flag,proc_flag,out_flag,chart_flag,med_flag)
        print("[ DATA DIVIDED INTO BATCHES ]")
        
        self.train_test()
        
        print("[ TRAIN-TEST-VALIDATION SET CREATED ]")
        
        self.med_seq_len,self.cond_seq_len, self.proc_seq_len,self.out_seq_len,self.chart_seq_len=model_utils.get_meta()
    
        if torch.cuda.is_available():
            self.device='cuda:0'
        #self.device='cpu'
        self.create_model()
        print("[ MODEL CREATED ]")
        print(self.net)
        self.loss=evaluation.Loss(self.device,True,True,True,True,True,True,True,True,False,True,False)
        print("[ TRAINING STARTED ]")
        self.model_train()
        print("[ TRAINING COMPLETED ]")

        self.model_test()
        
        self.save_output()

    def train_test(self):        
        self.train_batch,self.test_batch=train_test_split(list(range(0,self.nBatches)),test_size=args.test_size, random_state=43)
        self.train_batch,self.val_batch=train_test_split(self.train_batch,test_size=args.val_size, random_state=53)
    
    def create_model(self):
#         self.net = model.LSTMAttn(self.device,
#                                self.cond_vocab_size,self.cond_seq_len, 
#                                self.proc_vocab_size,self.proc_seq_len,
#                                self.med_vocab_size,self.med_seq_len,
#                                self.out_vocab_size,self.out_seq_len,
#                                self.chart_vocab_size,self.chart_seq_len,
#                                embed_size=args.embedding_size,rnn_size=args.rnn_size, 
#                                batch_size=args.batch_size) 
        self.net = model.CNNBaseH(self.device,
                               self.cond_vocab_size,self.cond_seq_len, 
                               self.proc_vocab_size,self.proc_seq_len,
                               self.med_vocab_size,self.med_seq_len,
                               self.out_vocab_size,self.out_seq_len,
                               self.chart_vocab_size,self.chart_seq_len,
                               self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size,
                               embed_size=args.embedding_size,rnn_size=args.rnn_size, 
                               batch_size=args.batch_size) 
        self.loss=model.Loss(self.device)
        # define the loss and the optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        #criterion = nn.CrossEntropyLoss()
        self.net.to(self.device)
        
    def model_train(self):
        min_loss=100
        for epoch in range(5):
            print("======= EPOCH {:.1f} ========".format(epoch))
            # reinit the hidden and cell steates
            #net.init_hidden()
            train_prob=[]
            train_truth=[]
            self.net.train()
            for key, value in model_utils.get_batches().items():
                if key in self.train_batch:
                    meds,proc,conds,demo,labels=model_utils.get_batch_data(value)

                    meds=torch.tensor(meds)
                    #print(meds.shape)
                    meds=meds.type(torch.LongTensor)
                    
                    proc=torch.tensor(proc)
                    #print(proc.shape)
                    proc=proc.type(torch.LongTensor)

                    conds=torch.tensor(conds)
                    #print(conds.shape)
                    conds=conds.type(torch.LongTensor)
                    
                    demo=torch.tensor(demo)
                    #print(demo.shape)
                    demo=demo.type(torch.LongTensor)

                    labels=torch.tensor(labels)
                    labels=labels.type(torch.FloatTensor)

                    self.optimizer.zero_grad()

                    # get the output sequence from the input and the initial hidden and cell states
                    output = self.net(meds,proc,conds,demo,contrib=False)
                    output=output.squeeze()
                    
                    out_loss=self.loss(output,labels,True)
                    #print(out_loss)
                    # calculate the gradients
                    out_loss.backward()
                    # update the parameters of the model
                    self.optimizer.step()

                    train_prob.extend(output.data.cpu().numpy())
                    train_truth.extend(labels.data.cpu().numpy())
            
            self.loss(torch.tensor(train_prob),torch.tensor(train_truth),False)
            val_loss=self.model_val()
            if(val_loss<=min_loss):
                print("Updating Model")
                min_loss=val_loss
                T.save(self.net,save_path)
    
    def model_test(self):

        print("======= TESTING ========")

        self.prob=[]
        self.eth=[]
        self.gender=[]
        self.age=[]
        self.truth=[]
        self.net.train()
        for key, value in model_utils.get_batches().items():
            if key in self.test_batch:
                meds,proc,conds,demo,labels=model_utils.get_batch_data(value)

                meds=torch.tensor(meds)
                #print(meds.shape)
                meds=meds.type(torch.LongTensor)

                proc=torch.tensor(proc)
                #print(proc.shape)
                proc=proc.type(torch.LongTensor)

                conds=torch.tensor(conds)
                #print(conds.shape)
                conds=conds.type(torch.LongTensor)

                self.eth.extend(demo[0])
                self.gender.extend(demo[1])
                self.age.extend(demo[2])
                demo=torch.tensor(demo)
                #print(demo.shape)
                demo=demo.type(torch.LongTensor)

                labels=torch.tensor(labels)
                labels=labels.type(torch.FloatTensor)

                self.optimizer.zero_grad()

                # get the output sequence from the input and the initial hidden and cell states
                output = self.net(meds,proc,conds,demo,contrib=False)
                output=output.squeeze()

                out_loss=self.loss(output,labels,True)
                #print(out_loss)
                # calculate the gradients
                out_loss.backward()
                # update the parameters of the model
                self.optimizer.step()

                self.prob.extend(output.data.cpu().numpy())
                self.truth.extend(labels.data.cpu().numpy())

        self.loss(torch.tensor(self.prob),torch.tensor(self.truth),False)
            
    def model_val(self):
        
        print("======= VALIDATION ========")
        # reinit the hidden and cell steates
        #net.init_hidden()
        val_prob=[]
        val_truth=[]
        self.net.eval()
        for key, value in model_utils.get_batches().items():
            if key in self.val_batch:
                meds,proc,conds,demo,labels=model_utils.get_batch_data(value)

                meds=torch.tensor(meds)
                #print(meds.shape)
                meds=meds.type(torch.LongTensor)

                proc=torch.tensor(proc)
                #print(proc.shape)
                proc=proc.type(torch.LongTensor)

                conds=torch.tensor(conds)
                #print(conds.shape)
                conds=conds.type(torch.LongTensor)

                demo=torch.tensor(demo)
                #print(demo.shape)
                demo=demo.type(torch.LongTensor)

                labels=torch.tensor(labels)
                val_labels=labels.type(torch.FloatTensor)

                # get the output sequence from the input and the initial hidden and cell states
                output = self.net(meds,proc,conds,demo,contrib=False)
                output=output.squeeze()

                val_prob.extend(output.data.cpu().numpy())
                val_truth.extend(val_labels.data.cpu().numpy())
        
        self.loss(torch.tensor(val_prob),torch.tensor(val_truth),False)
        val_loss=self.loss(output,val_labels,True)
        return val_loss.item()
            
    def save_output(self):
        with open ('./data/dict/'+'ethVocabDict', 'rb') as fp:
            ethVocabDict = pickle.load(fp)
        with open ('./data/dict/'+'genderVocabDict', 'rb') as fp:
            genderVocabDict = pickle.load(fp)    
        with open ('./data/dict/'+'ageVocabDict', 'rb') as fp:
            ageVocabDict = pickle.load(fp)
            
        reversed_eth = {ethVocabDict[key]: key for key in ethVocabDict}
        reversed_gender = {genderVocabDict[key]: key for key in genderVocabDict}
        reversed_age = {ageVocabDict[key]: key for key in ageVocabDict}
        
        
        self.eth=list(pd.Series(self.eth).map(reversed_eth))
        self.gender=list(pd.Series(self.gender).map(reversed_gender))
        self.age=list(pd.Series(self.age).map(reversed_age))
        
        output_df=pd.DataFrame()
        output_df['Labels']=self.truth
        output_df['Prob']=self.prob
        output_df['ethnicity']=self.eth
        output_df['gender']=self.gender
        output_df['age']=self.age
        
        with open('./data/dict/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)

        
    

