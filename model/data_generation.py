import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
path='C:/Users/mehak/OneDrive - University of Delaware - o365/Beheshti, Rahmat - Mehak - Brennan/data/'

class Generator():
    def __init__(self,include_time=24,bucket=1):
        self.data = generate_adm()
        self.cond, self.cond_per_adm = generate_cond()
        self.meds = generate_meds()
        self.los = input_length(include_time)
        self.los,self.med_per_adm,self.medlength_per_adm, self.med_vocab, self.cond_vocab = smooth_meds(self.los,bucket)

    def generate_adm():
        data=pd.read_csv(path+'day_intervals/cohort/cohort30day.csv.gz', compression='gzip', header=0, index_col=None)
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
    
    def generate_cond():
        cond=pd.read_csv(path+'long_format/diag/long_diag_icd10_roots_norm.csv.gz', compression='gzip', header=0, index_col=None)
        cond=cond[cond['hadm_id'].isin(data['hadm_id'])]
        cond_per_adm = cond.groupby('hadm_id').size().max()
        return cond, cond_per_adm
        
    def generate_meds():
        meds=pd.read_csv(path+'long_format/meds/preproc_med_nonproprietary.csv.gz', compression='gzip', header=0, index_col=None)
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
        return meds
        
    def input_length(include_time):
        los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
        self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]
        
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        self.meds=pd.merge(self.meds,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
        self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
        self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
        self.meds=self.meds[self.meds['stop_time']>=0]
        self.meds.loc[self.meds.start_time <0, 'start_time']=0

        if(predW):
            los=include_time-predW
            self.meds=self.meds[self.meds['start_time']<los]
            self.meds.loc[self.meds['stop_time'] > los,'stop_time']=los
        return los
            
    def smooth_meds(los,bucket):
        final=pd.DataFrame()
        self.meds=self.meds.sort_values(by=['start_time'])
        t=0
        for i in tqdm(range(0,los,bucket)): 
            sub=self.meds[(self.meds['start_time']>=i) & (self.meds['start_time']<i+bucket)].groupby(['hadm_id','nonproprietaryname']).agg({'stop_time':'max','subject_id':'max'})
            sub=sub.reset_index()
            sub['start_time']=t
            t=t+1
            sub['stop_time']=sub['stop_time']/bucket
            if final.empty:
                final=sub
            else:
                final=final.append(sub)
        los=int(los/bucket)
        
        f2=final.groupby(['hadm_id','nonproprietaryname']).size()
        med_per_adm=f2.groupby('hadm_id').sum().reset_index()[0].max()
        
        medlength_per_adm=final.groupby('hadm_id').size().max()

        med_vocab, cond_vocab = create_Dic(final,los)
        return los, med_per_adm, medlength_per_adm, med_vocab, cond_vocab
        
    def create_Dic(meds,los):
        for hid in tqdm(hids):
            df2=meds[meds['hadm_id']==hid]
            df2=df2.pivot(index='start_time',columns='nonproprietaryname',values='stop_time')
            #print(df2.shape)
            add_indices = pd.Index(range(los)).difference(df2.index)
            add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
            df2=pd.concat([df2, add_df])
            df2=df2.ffill()
            df2=df2.fillna(0)
            #print(df2.head())
            df2.iloc[:,1:]=df2.iloc[:,1:].sub(df2.index,0)
            df2[df2>0]=1
            df2[df2<0]=0
            #print(df2.head())
            dataDic[hid]['Med']=df2.iloc[:,1:].to_dict(orient="list")
            
            ##########COND#########
            grp=self.cond[self.cond['hadm_id']==hid]
            if(grp.shape[0]==0):
                dataDic[hid]['Cond']={'fids':list(['<PAD>'])}
            else:
                dataDic[hid]['Cond']={'fids':list(grp['root'])}
                
                
        ######SAVE DICTIONARIES##############
        path='C:/Users/mehak/OneDrive - University of Delaware - o365/Beheshti, Rahmat - Mehak - Brennan/model/data/'

        with open(path+'dataDic', 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open(path+'hadmDic', 'wb') as fp:
            pickle.dump(hids, fp)

        with open(path+'medVocab', 'wb') as fp:
            pickle.dump(list(meds['nonproprietaryname'].unique()), fp)

        with open(path+'condVocab', 'wb') as fp:
            pickle.dump(list(self.cond['root'].unique()), fp)
            
        return meds['nonproprietaryname'].nunique(),cond['root'].nunique()



