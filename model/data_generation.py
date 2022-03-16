import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
import datetime
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

class Generator():
    def __init__(self,cohort_output,feat_cond,feat_lab,feat_proc,feat_med,include_time=24,bucket=1,predW=0):
        self.feat_cond,self.feat_proc,self.feat_med,self.feat_lab = feat_cond,feat_proc,feat_med,feat_lab
        self.cohort_output=cohort_output
        self.data = self.generate_adm()
        self.generate_feat()
        self.input_length(include_time,predW)
        self.smooth_meds(bucket)
    
    def generate_feat(self):
        if(self.feat_cond):
            self.generate_cond()
        if(self.feat_proc):
            self.generate_proc()
        if(self.feat_med):
            self.generate_meds()
        if(self.feat_lab):
            self.generate_labs()
            
    def generate_adm(self):
        data=pd.read_csv(f"./data/cohort/{summary_output}.csv.gz", compression='gzip', header=0, index_col=None)
        data['admittime'] = pd.to_datetime(data['admittime'])
        data['dischtime'] = pd.to_datetime(data['dischtime'])
        data['los']=pd.to_timedelta(data['dischtime']-data['admittime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy','hours']] = data['los'].str.split(' ', -1, expand=True)
        data[['hours','min','sec']] = data['hours'].str.split(':', -1, expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]
        return data
    
    def generate_cond(self):
        cond=pd.read_csv("./data/long_format/diag/long_diag_icd10_roots_norm.csv.gz", compression='gzip', header=0, index_col=None)
        cond=cond[cond['hadm_id'].isin(data['hadm_id'])]
        cond_per_adm = cond.groupby('hadm_id').size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm
    
    def generate_proc(self):
        proc=pd.read_csv("./data/long_format/proc/preproc_proc_icd10.csv.gz", compression='gzip', header=0, index_col=None)
        proc=proc[proc['hadm_id'].isin(data['hadm_id'])]
        proc[['start_days', 'dummy','start_hours']] = proc['proc_time_from_admit'].str.split(' ', -1, expand=True)
        proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', -1, expand=True)
        proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
        proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        proc=proc[proc['start_time']>=0]
        
        ###Remove where event time is after discharge time
        proc=pd.merge(proc,data[['hadm_id','los']],on='hadm_id',how='left')
        proc['sanity']=proc['los']-proc['start_time']
        proc=proc[proc['sanity']>0]
        del proc['sanity']
        
        self.proc=proc
        
    def generate_meds(self):
        meds=pd.read_csv("./data/long_format/meds/preproc_med_nonproprietary.csv.gz", compression='gzip', header=0, index_col=None)
        meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        #####Sanity check
        meds['sanity']=meds['stop_time']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        #####Select hadm_id as in main file
        meds=meds[meds['hadm_id'].isin(data['hadm_id'])]
        meds=pd.merge(meds,data[['hadm_id','los']],on='hadm_id',how='left')

        #####Remove where start time is after end of visit
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']
        self.meds=meds
        
    def input_length(self,include_time,predW):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
        self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]
        self.proc=self.proc[self.proc['hadm_id'].isin(self.data['hadm_id'])]
        self.labs=self.labs[self.labs['hadm_id'].isin(self.data['hadm_id'])]
        
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=pd.merge(self.meds,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
            self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
            self.meds=self.meds[self.meds['stop_time']>=0]
            self.meds.loc[self.meds.start_time <0, 'start_time']=0
        
        ###PROCS
        if(self.feat_proc):
            self.proc=pd.merge(self.proc,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.proc['start_time']=self.proc['start_time']-self.proc['select_time']
            self.proc=self.proc[self.proc['start_time']>=0]
        
        ###LABS
        if(self.feat_lab):
            self.labs=pd.merge(self.labs,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.labs['start_time']=self.labs['start_time']-self.labs['select_time']
            self.labs=self.labs[self.labs['start_time']>=0]

        if(predW):
            self.los=include_time-predW
            if(self.feat_med):
                self.meds=self.meds[self.meds['start_time']<los]
                self.meds.loc[self.meds['stop_time'] > los,'stop_time']=los
            if(self.feat_proc):
                self.proc=self.proc[self.proc['start_time']<los]
            if(self.feat_lab):
                self.labs=self.labs[self.labs['start_time']<los]

            
    def smooth_meds(self,bucket):
        final_meds=pd.DataFrame()
        final_proc=pd.DataFrame()
        
        if(self.feat_med):
            self.meds=self.meds.sort_values(by=['start_time'])
        if(self.feat_proc):
            self.proc=self.proc.sort_values(by=['start_time'])
        
        t=0
        for i in tqdm(range(0,self.los,bucket)): 
            ###MEDS
             if(self.feat_med):
                sub_meds=self.meds[(self.meds['start_time']>=i) & (self.meds['start_time']<i+bucket)].groupby(['hadm_id','nonproprietaryname']).agg({'stop_time':'max','subject_id':'max'})
                sub_meds=sub_meds.reset_index()
                sub_meds['start_time']=t
                sub_meds['stop_time']=sub_meds['stop_time']/bucket
                if final_meds.empty:
                    final_meds=sub_meds
                else:
                    final_meds=final_meds.append(sub_meds)
            
            ###PROC
             if(self.feat_proc):
                sub_proc=self.proc[(self.proc['start_time']>=i) & (self.proc['start_time']<i+bucket)].groupby(['hadm_id','icd_code']).agg({'subject_id':'max'})
                sub_proc=sub_proc.reset_index()
                sub_proc['start_time']=t
                if final_proc.empty:
                    final_proc=sub_proc
                else:    
                    final_proc=final_proc.append(sub_proc)
            
             t=t+1
        los=int(los/bucket)
        
        ###MEDS
        if(self.feat_med):
            f2_meds=final_meds.groupby(['hadm_id','nonproprietaryname']).size()
            self.med_per_adm=f2_meds.groupby('hadm_id').sum().reset_index()[0].max()        
            self.medlength_per_adm=final_meds.groupby('hadm_id').size().max()
        
        ###PROC
        if(self.feat_proc):
            f2_proc=final_proc.groupby(['hadm_id','nonproprietaryname']).size()
            self.proc_per_adm=final_proc.groupby('hadm_id').sum().reset_index()[0].max()        
            self.proclength_per_adm=final_proc.groupby('hadm_id').size().max()

        ###CREATE DICT
        create_Dict(final_meds,final_proc,los)
        
        
    def create_Dict(self,los,meds,proc):
        for hid in tqdm(hids):
            ###MEDS
            if(self.feat_med):
                df2=meds[meds['hadm_id']==hid]
                df2=df2.pivot(index='start_time',columns='nonproprietaryname',values='stop_time')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.ffill()
                df2=df2.fillna(0)
                #print(df2.head())
                df2.iloc[:,1:]=df2.iloc[:,1:].sub(df2.index,0)
                df2[df2>0]=1
                df2[df2<0]=0
                #print(df2.head())
                dataDic[hid]['Med']=df2.iloc[:,1:].to_dict(orient="list")
            
            
            ###PROCS
            if(self.feat_proc):
                df2=proc[proc['hadm_id']==hid]
                df2=df2.pivot(index='start_time',columns='icd_code',values='start_time')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)
                df2[df2>0]=1
                #print(df2.head())
                dataDic[hid]['Proc']=df2.to_dict(orient="list")
            
            ##########COND#########
            if(self.feat_cond):
                grp=self.cond[self.cond['hadm_id']==hid]
                if(grp.shape[0]==0):
                    dataDic[hid]['Cond']={'fids':list(['<PAD>'])}
                else:
                    dataDic[hid]['Cond']={'fids':list(grp['root'])}
                
                
        ######SAVE DICTIONARIES##############
        path='C:/Users/mehak/OneDrive - University of Delaware - o365/Beheshti, Rahmat - Mehak - Brennan/model/data/'

        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(hids, fp)
        
        if(self.feat_med):
            with open("./data/dict/medVocab", 'wb') as fp:
                pickle.dump(list(meds['nonproprietaryname'].unique()), fp)
            self.med_vocab = meds['nonproprietaryname'].nunique()
        
        if(self.feat_cond):
            with open("./data/dict/condVocab", 'wb') as fp:
                pickle.dump(list(self.cond['root'].unique()), fp)
            self.cond_vocab = cond['root'].nunique()
        
        if(self.feat_proc):    
            with open("./data/dict/procVocab", 'wb') as fp:
                pickle.dump(list(self.proc['root'].unique()), fp)
            self.proc_vocab = self.proc['root'].unique()
            



