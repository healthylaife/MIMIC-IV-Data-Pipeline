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

#save_path = "saved_models/model.tar"
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
    def __init__(self,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag,med_signal,lab_signal,model_type,model_name,train,init_batches):
        self.save_path="saved_models/"+model_name+".tar"
        self.diag_flag,self.proc_flag,self.out_flag,self.chart_flag,self.med_flag,self.lab_flag,self.med_signal,self.lab_signal=diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag,med_signal,lab_signal
            
        if train and init_batches: 
            self.nBatches,self.cond_vocab_size,self.proc_vocab_size,self.med_vocab_size,self.out_vocab_size,self.chart_vocab_size,self.lab_vocab_size,self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size=model_utils.init(args.batch_size,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag)
            print("[ DATA DIVIDED INTO BATCHES ]")
        else:
            self.nBatches,self.cond_vocab_size,self.proc_vocab_size,self.med_vocab_size,self.out_vocab_size,self.chart_vocab_size,self.lab_vocab_size,self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size=model_utils.init_read(args.batch_size,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag)
            
        
        #print(self.nBatches)
        self.train_test()
        print("[ TRAIN-TEST-VALIDATION SET CREATED ]")
        
        self.med_seq_len,self.cond_seq_len, self.proc_seq_len,self.out_seq_len,self.chart_seq_len,self.lab_seq_len=model_utils.get_meta()
    
        if torch.cuda.is_available():
            self.device='cuda:0'
        #self.device='cpu'
        if train:
            self.create_model(model_type)
            print("[ MODEL CREATED ]")
        else:
            self.net=torch.load(self.save_path)
            print("[ MODEL LOADED ]")
        print(self.net)
        self.loss=evaluation.Loss(self.device,True,True,True,True,True,True,True,True,True,True,True)
        if train: 
            print("[ TRAINING STARTED ]")
            self.model_train()
            print("[ TRAINING COMPLETED ]")

        self.model_test()
        
        self.save_output()

    def train_test(self):        
        self.train_batch,self.test_batch=train_test_split(list(range(0,self.nBatches)),test_size=args.test_size, random_state=43)
        self.train_batch,self.val_batch=train_test_split(self.train_batch,test_size=args.val_size, random_state=53)
    
    def create_model(self,model_type):
        if model_type=='Time-series LSTM':
            self.net = model.LSTMBase(self.device,
                               self.cond_vocab_size,self.cond_seq_len, 
                               self.proc_vocab_size,self.proc_seq_len,
                               self.med_vocab_size,self.med_seq_len,
                               self.out_vocab_size,self.out_seq_len,
                               self.chart_vocab_size,self.chart_seq_len,
                               self.lab_vocab_size,self.lab_seq_len,
                               self.med_signal,self.lab_signal,
                               embed_size=args.embedding_size,rnn_size=args.rnn_size,
                               batch_size=args.batch_size) 
        elif model_type=='Time-series CNN':
            self.net = model.CNNBase(self.device,
                               self.cond_vocab_size,self.cond_seq_len, 
                               self.proc_vocab_size,self.proc_seq_len,
                               self.med_vocab_size,self.med_seq_len,
                               self.out_vocab_size,self.out_seq_len,
                               self.chart_vocab_size,self.chart_seq_len,
                               self.lab_vocab_size,self.lab_seq_len,
                               self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size,
                               self.med_signal,self.lab_signal,
                               embed_size=args.embedding_size,rnn_size=args.rnn_size,
                               batch_size=args.batch_size) 
        elif model_type=='Hybrid LSTM':
            self.net = model.LSTMBaseH(self.device,
                               self.cond_vocab_size,self.cond_seq_len, 
                               self.proc_vocab_size,self.proc_seq_len,
                               self.med_vocab_size,self.med_seq_len,
                               self.out_vocab_size,self.out_seq_len,
                               self.chart_vocab_size,self.chart_seq_len,
                               self.lab_vocab_size,self.lab_seq_len,
                               self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size,
                               self.med_signal,self.lab_signal,
                               embed_size=args.embedding_size,rnn_size=args.rnn_size,
                               batch_size=args.batch_size) 
        elif model_type=='Hybrid CNN':
            self.net = model.CNNBaseH(self.device,
                               self.cond_vocab_size,self.cond_seq_len, 
                               self.proc_vocab_size,self.proc_seq_len,
                               self.med_vocab_size,self.med_seq_len,
                               self.out_vocab_size,self.out_seq_len,
                               self.chart_vocab_size,self.chart_seq_len,
                               self.lab_vocab_size,self.lab_seq_len,
                               self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size,
                               self.med_signal,self.lab_signal,
                               embed_size=args.embedding_size,rnn_size=args.rnn_size, 
                               batch_size=args.batch_size) 
        elif model_type=='LSTM with Attention':
            self.net = model.LSTMAttn(self.device,
                                   self.cond_vocab_size,self.cond_seq_len, 
                                   self.proc_vocab_size,self.proc_seq_len,
                                   self.med_vocab_size,self.med_seq_len,
                                   self.out_vocab_size,self.out_seq_len,
                                   self.chart_vocab_size,self.chart_seq_len,
                                   self.lab_vocab_size,self.lab_seq_len,
                                   self.eth_vocab_size,self.gender_vocab_size,self.age_vocab_size,
                                   self.med_signal,self.lab_signal,
                                   embed_size=args.embedding_size,rnn_size=args.rnn_size,
                                   batch_size=args.batch_size) 
        
        #self.loss=model.Loss(self.device)
        # define the loss and the optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        #criterion = nn.CrossEntropyLoss()
        self.net.to(self.device)
        
    def model_train(self):
        min_loss=100
        counter=0
        for epoch in range(args.num_epochs):
            if counter==args.patience:
                print("STOPPING THE TRAINING BECAUSE VALIDATION ERROR DID nOT IMPROVE FOR {:.1f} EPOCHS".format(args.patience))
                break
            print("======= EPOCH {:.1f} ========".format(epoch))
            # reinit the hidden and cell steates
            #net.init_hidden()
            train_prob=[]
            train_logits=[]
            train_truth=[]
            self.net.train()
            for key, value in model_utils.get_batches().items():
                if key in self.train_batch:
                    #print("new batch")
                    meds,proc,outs,charts,labs,conds,demo,labels=model_utils.get_batch_data(value,self.diag_flag,self.proc_flag,self.out_flag,self.chart_flag,self.med_flag,self.lab_flag)
                    
                    if len(meds[0]):
                        meds=torch.tensor(meds)
                        meds=meds.type(torch.LongTensor)

                    if len(proc):
                        proc=torch.tensor(proc)
                        #print(proc.shape)
                        proc=proc.type(torch.LongTensor)

                    if len(conds):
                        conds=torch.tensor(conds)
                        #print(conds.shape)
                        conds=conds.type(torch.LongTensor)

                    if len(outs):
                        outs=torch.tensor(outs)
                        #print(meds.shape)
                        outs=outs.type(torch.LongTensor)
                    
                    if len(charts[0]):
                        charts=torch.tensor(charts)
                        #print(proc.shape)
                        charts=charts.type(torch.LongTensor)

                    if len(labs[0]):
                        labs=torch.tensor(labs)
                        #print(conds.shape)
                        labs=labs.type(torch.LongTensor)

                    demo=torch.tensor(demo)
                    #print(demo.shape)
                    demo=demo.type(torch.LongTensor)

                    labels=torch.tensor(labels)
                    labels=labels.type(torch.FloatTensor)

                    self.optimizer.zero_grad()

                    # get the output sequence from the input and the initial hidden and cell states
                    output,logits = self.net(meds,proc,outs,charts,labs,conds,demo,contrib=False)
                    output=output.squeeze()
                    logits=logits.squeeze()

                    out_loss=self.loss(output,labels,logits,True)
                    #print(out_loss)
                    # calculate the gradients
                    out_loss.backward()
                    # update the parameters of the model
                    self.optimizer.step()

                    train_prob.extend(output.data.cpu().numpy())
                    train_truth.extend(labels.data.cpu().numpy())
                    train_logits.extend(logits.data.cpu().numpy())
            
            self.loss(torch.tensor(train_prob),torch.tensor(train_truth),torch.tensor(train_logits),False)
            val_loss=self.model_val()
            #print("Updating Model")
            #T.save(self.net,self.save_path)
            if(val_loss<=min_loss+0.3):
                print("Validation results improved")
                min_loss=val_loss
                print("Updating Model")
                T.save(self.net,self.save_path)
                counter=0
            else:
                print("No improvement in Validation results")
                counter=counter+1
    
    def model_test(self):

        print("======= TESTING ========")

        self.prob=[]
        self.eth=[]
        self.gender=[]
        self.age=[]
        self.truth=[]
        self.logits=[]
        self.net.eval()
        for key, value in model_utils.get_batches().items():
            if key in self.test_batch:
                meds,proc,outs,charts,labs,conds,demo,labels=model_utils.get_batch_data(value,self.diag_flag,self.proc_flag,self.out_flag,self.chart_flag,self.med_flag,self.lab_flag)
                    
                if len(meds[0]):
                    meds=torch.tensor(meds)
                    meds=meds.type(torch.LongTensor)

                if len(proc):
                    proc=torch.tensor(proc)
                    #print(proc.shape)
                    proc=proc.type(torch.LongTensor)

                if len(conds):
                    conds=torch.tensor(conds)
                    #print(conds.shape)
                    conds=conds.type(torch.LongTensor)

                if len(outs):
                    outs=torch.tensor(outs)
                    #print(meds.shape)
                    outs=outs.type(torch.LongTensor)

                if len(charts[0]):
                    charts=torch.tensor(charts)
                    #print(proc.shape)
                    charts=charts.type(torch.LongTensor)

                if len(labs[0]):
                    labs=torch.tensor(labs)
                    #print(conds.shape)
                    labs=labs.type(torch.LongTensor)

                self.eth.extend(demo[0])
                self.gender.extend(demo[1])
                self.age.extend(demo[2])
                demo=torch.tensor(demo)
                #print(demo.shape)
                demo=demo.type(torch.LongTensor)

                labels=torch.tensor(labels)
                labels=labels.type(torch.FloatTensor)

               

                # get the output sequence from the input and the initial hidden and cell states
                output,logits = self.net(meds,proc,outs,charts,labs,conds,demo,contrib=False)
                output=output.squeeze()
                logits=logits.squeeze()


                self.prob.extend(output.data.cpu().numpy())
                self.truth.extend(labels.data.cpu().numpy())
                self.logits.extend(logits.data.cpu().numpy())

        self.loss(torch.tensor(self.prob),torch.tensor(self.truth),torch.tensor(self.logits),False)
            
    def model_val(self):
        
        print("======= VALIDATION ========")
        # reinit the hidden and cell steates
        #net.init_hidden()
        val_prob=[]
        val_truth=[]
        val_logits=[]
        self.net.eval()
        for key, value in model_utils.get_batches().items():
            if key in self.val_batch:
                meds,proc,outs,charts,labs,conds,demo,labels=model_utils.get_batch_data(value,self.diag_flag,self.proc_flag,self.out_flag,self.chart_flag,self.med_flag,self.lab_flag)
                
                if meds:
                    meds=torch.tensor(meds)
                    #print(meds.shape)
                    meds=meds.type(torch.LongTensor)
                
                if proc:
                    proc=torch.tensor(proc)
                    #print(proc.shape)
                    proc=proc.type(torch.LongTensor)
                
                if conds:
                    conds=torch.tensor(conds)
                    #print(conds.shape)
                    conds=conds.type(torch.LongTensor)
                    
                if outs:
                    outs=torch.tensor(outs)
                    #print(meds.shape)
                    outs=outs.type(torch.LongTensor)
                
                if charts:
                    charts=torch.tensor(charts)
                    #print(proc.shape)
                    charts=charts.type(torch.LongTensor)
                
                if labs:
                    labs=torch.tensor(labs)
                    #print(conds.shape)
                    labs=labs.type(torch.LongTensor)

                demo=torch.tensor(demo)
                #print(demo.shape)
                demo=demo.type(torch.LongTensor)

                labels=torch.tensor(labels)
                val_labels=labels.type(torch.FloatTensor)

                # get the output sequence from the input and the initial hidden and cell states
                output,logits = self.net(meds,proc,outs,charts,labs,conds,demo,contrib=False)
                output=output.squeeze()
                logits=logits.squeeze()

                val_prob.extend(output.data.cpu().numpy())
                val_truth.extend(val_labels.data.cpu().numpy())
                val_logits.extend(logits.data.cpu().numpy())
        
        self.loss(torch.tensor(val_prob),torch.tensor(val_truth),torch.tensor(val_logits),False)
        val_loss=self.loss(torch.tensor(val_prob),torch.tensor(val_truth),torch.tensor(val_logits),True)
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
        output_df['Logits']=self.logits
        output_df['ethnicity']=self.eth
        output_df['gender']=self.gender
        output_df['age']=self.age
        
        with open('./data/dict/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)

        
    

