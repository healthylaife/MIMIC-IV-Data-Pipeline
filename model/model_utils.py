import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

MAX_LEN=12
MAX_COND_SEQ=56
MAX_PROC_SEQ=40
MAX_MED_SEQ=15#37
MAX_LAB_SEQ=899
MAX_BMI_SEQ=118



def create_vocab(file):
    with open ('./data/dict/'+file, 'rb') as fp:
        condVocab = pickle.load(fp)
    condVocabDict={}
    condVocabDict['<PAD>']=0
    for val in range(len(condVocab)):
        condVocabDict[condVocab[val]]= val+1    

    return condVocabDict

def gender_vocab():
    genderVocabDict={}
    genderVocabDict['<PAD>']=0
    genderVocabDict['M']=1
    genderVocabDict['F']=2

    return genderVocabDict

def create_batches(batch_size):
    with open ('./data/dict/'+'stayDic', 'rb') as fp:
        hids = pickle.load(fp)
    
    batchDict={}
    with open ('./data/dict/'+'dataDic', 'rb') as fp:
        dataDic = pickle.load(fp)

    batch_idx=0
    ids=range(0,len(hids))
    for i in range(0,int(len(hids)/batch_size)):
        rids=random.sample(ids, batch_size)
        ids=list(set(ids)-set(rids))
        batch_hids=hids[rids]
        batchDict[batch_idx]=dict((k, dataDic[k]) for k in batch_hids)
        batch_idx=batch_idx+1
        
    return batchDict


def init(batch_size,diag_flag,proc_flag,out_flag,chart_flag,med_flag):
        condVocabDict={}
        procVocabDict={}
        medVocabDict={}
        outVocabDict={}
        chartVocabDict={}
        ethVocabDict={}
        ageVocabDict={}
        genderVocabDict={}
        
        ethVocabDict=create_vocab('ethVocab')
        with open('./data/dict/'+'ethVocabDict', 'wb') as fp:
            pickle.dump(ethVocabDict, fp)
            
        ageVocabDict=create_vocab('ageVocab')
        with open('./data/dict/'+'ageVocabDict', 'wb') as fp:
            pickle.dump(ageVocabDict, fp)
        
        genderVocabDict=gender_vocab()
        with open('./data/dict/'+'genderVocabDict', 'wb') as fp:
            pickle.dump(genderVocabDict, fp)
        
        if diag_flag:
            condVocabDict=create_vocab('condVocab')
            with open('./data/dict/'+'condVocabDict', 'wb') as fp:
                pickle.dump(condVocabDict, fp)
        if proc_flag:
            procVocabDict=create_vocab('procVocab')
            with open('./data/dict/'+'procVocabDict', 'wb') as fp:
                pickle.dump(procVocabDict, fp)
        if med_flag:
            medVocabDict=create_vocab('medVocab')
            with open('./data/dict/'+'medVocabDict', 'wb') as fp:
                pickle.dump(medVocabDict, fp)
        if out_flag:
            outVocabDict=create_vocab('outVocab')
            with open('./data/dict/'+'outVocabDict', 'wb') as fp:
                pickle.dump(outVocabDict, fp)
        if chart_flag:
            chartVocabDict=create_vocab('chartVocab')
            with open('./data/dict/'+'chartVocabDict', 'wb') as fp:
                pickle.dump(chartVocabDict, fp)
        
        batchDict= create_batches(batch_size)

        with open('./data/dict/'+'batchDict', 'wb') as fp:
            pickle.dump(batchDict, fp)
        return len(batchDict),len(condVocabDict),len(procVocabDict),len(medVocabDict),len(outVocabDict),len(chartVocabDict),len(ethVocabDict),len(genderVocabDict),len(ageVocabDict)

def get_meta():
    with open ('./data/dict/'+'metaDic', 'rb') as fp:
        meta = pickle.load(fp)
    return meta['Med'],meta['Cond'],meta['Proc'],meta['Out'],meta['Chart']

def get_batches():
    with open ('./data/dict/'+'batchDict', 'rb') as fp:
        batchDict = pickle.load(fp)
    return batchDict


def get_batch_data(data): 
    conds=[]  
    procs=[]
    meds=[]  
    meds_rate=[]  
    meds_amount=[]  
    labels=[]
    eth=[]
    age=[]
    gender=[]
    
    
    with open ('./data/dict/'+'metaDic', 'rb') as fp:
        meta = pickle.load(fp)
    with open ('./data/dict/'+'condVocabDict', 'rb') as fp:
        condVocabDict = pickle.load(fp)
    with open ('./data/dict/'+'procVocabDict', 'rb') as fp:
        procVocabDict = pickle.load(fp)
    with open ('./data/dict/'+'medVocabDict', 'rb') as fp:
        medVocabDict = pickle.load(fp)
    
    with open ('./data/dict/'+'ethVocabDict', 'rb') as fp:
        ethVocabDict = pickle.load(fp)
    with open ('./data/dict/'+'genderVocabDict', 'rb') as fp:
        genderVocabDict = pickle.load(fp)    
    with open ('./data/dict/'+'ageVocabDict', 'rb') as fp:
        ageVocabDict = pickle.load(fp)


    for hid, hid_data in data.items():
        #print(hid)
        for feature, feat_data in hid_data.items():
            #print(feature)
            if feature=='Med':
                #print(list(feat_data.keys()))
                fids=list(map(medVocabDict.get, list(feat_data['signal'].keys())))
                fids_pad=list(np.zeros(meta['Med']))
                fids_pad[0:len(fids)]=fids
                #fids=list(pd.Series(feat_data.keys()).map(medVocabDict))
                #print(fids)
                #meds.append(fids_pad) 
                med_len=list(feat_data['signal'].values())
                med_pad=np.asarray(fids_pad)
                med_pad=np.expand_dims(med_pad, axis=1)
                #print(med_pad)
                zeros = [ [0] * meta['LOS'] for _ in range(meta['Med'])]
                zeros[0:len(med_len)]=med_len
                #print(zeros)
                zeros = med_pad * np.asarray(zeros)
                meds.append(zeros)
                
                
                rate_len=list(feat_data['rate'].values())
                rate_pad=np.asarray(fids_pad)
                rate_pad=np.expand_dims(rate_pad, axis=1)
                
                zeros = [ [0] * meta['LOS'] for _ in range(meta['Med'])]
                for i in range(0,len(rate_len )):
                    zeros[i][0:len(rate_len[i])]=rate_len[i]

                #zeros = rate_pad * np.asarray(zeros)
                meds_rate.append(zeros)
                
                
                amount_len=list(feat_data['amount'].values())
                amount_pad=np.asarray(fids_pad)
                amount_pad=np.expand_dims(amount_pad, axis=1)
                
                zeros = [ [0] * meta['LOS'] for _ in range(meta['Med'])]
                for i in range(0,len(amount_len)):
                    zeros[i][0:len(amount_len[i])]=amount_len[i]

                #zeros = amount_pad * np.asarray(zeros)
                meds_amount.append(zeros)
                
            if feature=='Proc':
                #print(list(feat_data.keys()))
                fids=list(map(procVocabDict.get, list(feat_data.keys())))
                fids_pad=list(np.zeros(meta['Proc']))
                fids_pad[0:len(fids)]=fids
                #fids=list(pd.Series(feat_data.keys()).map(medVocabDict))
                #print(fids)
                #meds.append(fids_pad) 
                proc_len=list(feat_data.values())
                fids_pad=np.asarray(fids_pad)
                fids_pad=np.expand_dims(fids_pad, axis=1)
                
                zeros = [ [0] * meta['LOS'] for _ in range(meta['Proc'])]
                zeros[0:len(proc_len)]=proc_len

                zeros = fids_pad * np.asarray(zeros)
                procs.append(zeros)
                
            if feature=='Cond':
                #print(list(feat_data.keys()))
                fids=list(pd.Series(feat_data['fids']).map(condVocabDict))
                fids_pad=list(np.zeros(meta['Cond']))
                fids_pad[0:len(fids)]=fids
                conds.append(fids_pad) 
                
        labels.append(hid_data['label'])
        eth.append(ethVocabDict[hid_data['ethnicity']])
        age.append(ageVocabDict[hid_data['age']])
        gender.append(genderVocabDict[hid_data['gender']])

    return [meds,meds_rate,meds_amount],procs,conds,[eth,gender,age],labels

