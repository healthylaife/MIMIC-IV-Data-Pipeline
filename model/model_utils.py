import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

# MAX_LEN=12
# MAX_COND_SEQ=56
# MAX_PROC_SEQ=40
# MAX_MED_SEQ=15#37
# MAX_LAB_SEQ=899
# MAX_BMI_SEQ=118



def create_vocab(file):
    with open ('./data/dict/'+file, 'rb') as fp:
        condVocab = pickle.load(fp)
    condVocabDict={}
    condVocabDict[0]=0
    for val in range(len(condVocab)):
        condVocabDict[condVocab[val]]= val+1    

    return condVocabDict

def gender_vocab():
    genderVocabDict={}
    genderVocabDict['<PAD>']=0
    genderVocabDict['M']=1
    genderVocabDict['F']=2

    return genderVocabDict

def create_batches(batch_size,chart_flag):
    with open ('./data/dict/'+'hadmDic', 'rb') as fp:
        hids = pickle.load(fp)
    
    batchDict={}
    with open ('./data/dict/'+'dataDic', 'rb') as fp:
        dataDic = pickle.load(fp)
    if chart_flag:
        batchChartDict={}
        with open ('./data/dict/'+'dataChartDic', 'rb') as fp:
            dataChartDic = pickle.load(fp)
    
    batch_idx=0
    ids=range(0,len(hids))
    for i in range(0,int(len(hids)/batch_size)):
        rids=random.sample(ids, batch_size)
        ids=list(set(ids)-set(rids))
        batch_hids=hids[rids]
        batchDict[batch_idx]=dict((k, dataDic[k]) for k in batch_hids)
        if chart_flag:
            batchChartDict[batch_idx]=dict((k, dataChartDic[k]) for k in batch_hids)
        batch_idx=batch_idx+1
    if chart_flag:    
        return batchDict,batchChartDict
    else:
        return batchDict


def init(diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag):
        condVocabDict={}
        procVocabDict={}
        medVocabDict={}
        outVocabDict={}
        chartVocabDict={}
        labVocabDict={}
        ethVocabDict={}
        ageVocabDict={}
        genderVocabDict={}
        insVocabDict={}
        
        ethVocabDict=create_vocab('ethVocab')
        with open('./data/dict/'+'ethVocabDict', 'wb') as fp:
            pickle.dump(ethVocabDict, fp)
            
        ageVocabDict=create_vocab('ageVocab')
        with open('./data/dict/'+'ageVocabDict', 'wb') as fp:
            pickle.dump(ageVocabDict, fp)
        
        genderVocabDict=gender_vocab()
        with open('./data/dict/'+'genderVocabDict', 'wb') as fp:
            pickle.dump(genderVocabDict, fp)
            
        insVocabDict=create_vocab('insVocab')
        with open('./data/dict/'+'insVocabDict', 'wb') as fp:
            pickle.dump(insVocabDict, fp)
        
        if diag_flag:
            file='condVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                condVocabDict = pickle.load(fp)
        if proc_flag:
            file='procVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                procVocabDict = pickle.load(fp)
        if med_flag:
            file='medVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                medVocabDict = pickle.load(fp)
        if out_flag:
            file='outVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                outVocabDict = pickle.load(fp)
        if chart_flag:
            file='chartVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                chartVocabDict = pickle.load(fp)
        if lab_flag:
            file='labsVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                labVocabDict = pickle.load(fp)
        
        return len(condVocabDict),len(procVocabDict),len(medVocabDict),len(outVocabDict),len(chartVocabDict),len(labVocabDict),ethVocabDict,genderVocabDict,ageVocabDict,insVocabDict

    
def init_read(batch_size,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag):
        condVocabDict={}
        procVocabDict={}
        medVocabDict={}
        outVocabDict={}
        chartVocabDict={}
        labVocabDict={}
        ethVocabDict={}
        ageVocabDict={}
        genderVocabDict={}
        insVocabDict={}

        with open('./data/dict/'+'ethVocabDict', 'rb') as fp:
            ethVocabDict= pickle.load(fp)

        with open('./data/dict/'+'ageVocabDict', 'rb') as fp:
            ageVocabDict= pickle.load(fp)

        with open('./data/dict/'+'genderVocabDict', 'rb') as fp:
            genderVocabDict= pickle.load(fp)
            
        with open('./data/dict/'+'insVocabDict', 'rb') as fp:
            insVocabDict= pickle.load(fp)
        
        if diag_flag:
            file='condVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                condVocabDict = pickle.load(fp)
        if proc_flag:
            file='procVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                procVocabDict = pickle.load(fp)
        if med_flag:
            file='medVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                medVocabDict = pickle.load(fp)
        if out_flag:
            file='outVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                outVocabDict = pickle.load(fp)
        if chart_flag:
            file='chartVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                chartVocabDict = pickle.load(fp)
        if lab_flag:
            file='labsVocab'
            with open ('./data/dict/'+file, 'rb') as fp:
                labVocabDict = pickle.load(fp)
        

        return len(condVocabDict),len(procVocabDict),len(medVocabDict),len(outVocabDict),len(chartVocabDict),len(labVocabDict),ethVocabDict,genderVocabDict,ageVocabDict,insVocabDict

    
    
def get_meta():
    with open ('./data/dict/'+'metaDic', 'rb') as fp:
        meta = pickle.load(fp)
    if "Lab" in meta.keys():
        return meta['Med'],meta['Cond'],meta['Proc'],0,0,meta['Lab']
    else:
        return meta['Med'],meta['Cond'],meta['Proc'],meta['Out'],meta['Chart'],0

def get_batches():
    with open ('./data/dict/'+'batchDict', 'rb') as fp:
        batchDict = pickle.load(fp)
    return batchDict


def get_batch_data(key,data,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag): 
    conds=[]  
    procs=[]
    meds=[]  
    meds_rate=[]  
    meds_amount=[] 
    outs=[]
    charts=[] 
    charts_val=[]
    labs=[]
    labs_val=[]
    labels=[]
    eth=[]
    age=[]
    gender=[]
    
    
    with open ('./data/dict/'+'metaDic', 'rb') as fp:
        meta = pickle.load(fp)
    if diag_flag:
        with open ('./data/dict/'+'condVocabDict', 'rb') as fp:
            condVocabDict = pickle.load(fp)
    if proc_flag:
        with open ('./data/dict/'+'procVocabDict', 'rb') as fp:
            procVocabDict = pickle.load(fp)
    if med_flag:
        with open ('./data/dict/'+'medVocabDict', 'rb') as fp:
            medVocabDict = pickle.load(fp)
    if out_flag:
        with open ('./data/dict/'+'outVocabDict', 'rb') as fp:
            outVocabDict = pickle.load(fp)
    if lab_flag:
        with open ('./data/dict/'+'labVocabDict', 'rb') as fp:
            labVocabDict = pickle.load(fp)
   
    with open ('./data/dict/'+'ethVocabDict', 'rb') as fp:
        ethVocabDict = pickle.load(fp)
    with open ('./data/dict/'+'genderVocabDict', 'rb') as fp:
        genderVocabDict = pickle.load(fp)    
    with open ('./data/dict/'+'ageVocabDict', 'rb') as fp:
        ageVocabDict = pickle.load(fp)
        
        
    if chart_flag:
        with open ('./data/dict/'+'chartVocabDict', 'rb') as fp:
            chartVocabDict = pickle.load(fp)
        with open ('./data/dict/'+'batchChartDict', 'rb') as fp:
            batchChartDict = pickle.load(fp)
            batchChartDict=batchChartDict[key]
            
        for hid, hid_data in batchChartDict.items():
        #print(hid)
            for feature, feat_data in hid_data.items():
                #print(feature)
                if feature=='Chart':
                    #print(list(feat_data.keys()))
                    fids=list(map(chartVocabDict.get, list(feat_data['signal'].keys())))
                    fids_pad=list(np.zeros(meta['Chart']))
                    fids_pad[0:len(fids)]=fids
                    #fids=list(pd.Series(feat_data.keys()).map(medVocabDict))
                    #print(fids)
                    #meds.append(fids_pad) 
                    chart_len=list(feat_data['signal'].values())
                    chart_pad=np.asarray(fids_pad)
                    chart_pad=np.expand_dims(chart_pad, axis=1)
                    #print(med_pad)
                    zeros = [ [0] * meta['LOS'] for _ in range(meta['Chart'])]
                    zeros[0:len(chart_len)]=chart_len
                    #print(zeros)
                    zeros = chart_pad * np.asarray(zeros)
                    charts.append(zeros)


                    val_len=list(feat_data['val'].values())
                    val_pad=np.asarray(fids_pad)
                    val_pad=np.expand_dims(val_pad, axis=1)

                    zeros = [ [0] * meta['LOS'] for _ in range(meta['Chart'])]
                    for i in range(0,len(val_len )):
                        zeros[i][0:len(val_len[i])]=val_len[i]

                    #zeros = rate_pad * np.asarray(zeros)
                    charts_val.append(zeros)
    


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
                
                if 'rate' in feat_data.keys():
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
                else:
                    rate_len=list(feat_data['val'].values())
                    rate_pad=np.asarray(fids_pad)
                    rate_pad=np.expand_dims(rate_pad, axis=1)

                    zeros = [ [0] * meta['LOS'] for _ in range(meta['Med'])]
                    for i in range(0,len(rate_len )):
                        zeros[i][0:len(rate_len[i])]=rate_len[i]

                    #zeros = rate_pad * np.asarray(zeros)
                    meds_rate.append(zeros)
                
            if feature=='Lab':
                #print(list(feat_data.keys()))
                fids=list(map(labVocabDict.get, list(feat_data['signal'].keys())))
                fids_pad=list(np.zeros(meta['Lab']))
                fids_pad[0:len(fids)]=fids
                #fids=list(pd.Series(feat_data.keys()).map(medVocabDict))
                #print(fids)
                #meds.append(fids_pad) 
                lab_len=list(feat_data['signal'].values())
                lab_pad=np.asarray(fids_pad)
                lab_pad=np.expand_dims(lab_pad, axis=1)
                #print(med_pad)
                zeros = [ [0] * meta['LOS'] for _ in range(meta['Lab'])]
                zeros[0:len(lab_len)]=lab_len
                #print(zeros)
                zeros = lab_pad * np.asarray(zeros)
                labs.append(zeros)
                
                
                val_len=list(feat_data['val'].values())
                val_pad=np.asarray(fids_pad)
                val_pad=np.expand_dims(val_pad, axis=1)
                
                zeros = [ [0] * meta['LOS'] for _ in range(meta['Lab'])]
                for i in range(0,len(val_len )):
                    zeros[i][0:len(val_len[i])]=val_len[i]

                #zeros = rate_pad * np.asarray(zeros)
                labs_val.append(zeros)
            
                
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
                
            if feature=='Out':
                #print(list(feat_data.keys()))
                fids=list(map(outVocabDict.get, list(feat_data.keys())))
                fids_pad=list(np.zeros(meta['Out']))
                fids_pad[0:len(fids)]=fids
                #fids=list(pd.Series(feat_data.keys()).map(medVocabDict))
                #print(fids)
                #meds.append(fids_pad) 
                out_len=list(feat_data.values())
                fids_pad=np.asarray(fids_pad)
                fids_pad=np.expand_dims(fids_pad, axis=1)
                
                zeros = [ [0] * meta['LOS'] for _ in range(meta['Out'])]
                zeros[0:len(out_len)]=out_len

                zeros = fids_pad * np.asarray(zeros)
                outs.append(zeros)
                
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

    return [meds,meds_rate],procs,outs,[charts,charts_val],[labs,labs_val],conds,[eth,gender,age],labels

