#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import sys
import torch
import math
from sklearn import metrics
import torch.nn as nn
from torch import optim
import importlib
import evaluation
importlib.reload(evaluation)
import evaluation


from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')


def callibrate(inputFile, outputFile):
    
    output_dict = pickle.load(open('./data/output/'+inputFile,"rb"))
   

    if torch.cuda.is_available():
        device='cuda:0'
    device='cpu'
    temperature = nn.Parameter(torch.ones(1).to(device))
    temperature=temperature.type(torch.FloatTensor)
    #temperature=temperature.to('cuda:0')
    args = {'temperature': temperature}
    optimizer = optim.LBFGS([temperature], lr=0.0001, max_iter=100000, line_search_fn='strong_wolfe')


    def T_scaling(logits, args):
        temperature = args#.get('temperature', None)
        return torch.div(logits, temperature)

    temps = []
    losses = []
    def _eval():
       # scaled=T_scaling(torch.tensor(output_dict['Prob']).cuda(), temperature)
        
        loss = criterion(T_scaling(torch.tensor(output_dict['Logits']).type(torch.FloatTensor).to(device), temperature), torch.tensor(output_dict['Labels']).type(torch.FloatTensor).to(device))
        #print(loss)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss.item())
        return loss

   
    criterion = nn.BCEWithLogitsLoss()

    optimizer.step(_eval)


    pred=T_scaling(torch.tensor(output_dict['Logits']).to(device), temperature)
    sm = nn.Sigmoid()
    prob=sm(pred)
    #print(prob)
    loss=evaluation.Loss(device,True,True,True,True,True,True,True,True,True,True,True)
    print("BEFORE CALLIBRATION")
    if 'Prob' in output_dict.columns:
        out_loss=loss(torch.tensor(output_dict['Prob']),torch.tensor(output_dict['Labels']),torch.tensor(output_dict['Logits']),False)
        output_dict['Prob']=prob.data.cpu().numpy()
    else:
        out_loss=loss(sm(torch.tensor(output_dict['Logits']).to(device)),torch.tensor(output_dict['Labels']),torch.tensor(output_dict['Logits']),False)
    
    print("AFTER CALLIBRATION")
    out_loss=loss(prob,torch.tensor(output_dict['Labels']),pred,False)   
    output_dict['Logits']=pred.data.cpu().numpy()
    
    
    output_dict.to_csv('./data/output/'+outputFile+'.csv',index=False)

