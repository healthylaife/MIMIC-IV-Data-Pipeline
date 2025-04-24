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
if not os.path.exists("./data/dict"):
    os.makedirs("./data/dict")

class Generator():
    def __init__(self,cohort_output,if_mort,if_admn,if_los,feat_cond,feat_lab,feat_proc,feat_med,impute,include_time=24,bucket=1,predW=0):
        self.impute=impute
        self.feat_cond,self.feat_proc,self.feat_med,self.feat_lab = feat_cond,feat_proc,feat_med,feat_lab
        self.cohort_output=cohort_output
        self.max_num_days = 15
        self.include_interval = 6
        SELECTED_TIMES = list(range(0, 24 * self.max_num_days + 1, self.include_interval))

        self.data = self.generate_adm()
        for time_to_generate in SELECTED_TIMES:
            print("[ READ COHORT ]")
            self.generate_feat()
            print("[ READ ALL FEATURES ]")
            if if_mort:
                self.mortality_length(include_time)
                print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
            elif if_admn:
                self.readmission_length(include_time)
                print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
            elif if_los:
                self.los_length(include_time)
                print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
            self.smooth_meds(bucket)
            print(f" ----- {time_to_generate} ------ [ SUCCESSFULLY SAVED DATA DICTIONARIES ]")


    def generate_feat(self):
        if(self.feat_cond):
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if(self.feat_proc):
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if(self.feat_med):
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()
        if(self.feat_lab):
            print("[ ======READING LABS ]")
            self.generate_labs()


    def generate_adm(self):
        data=pd.read_csv(f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        data['admittime'] = pd.to_datetime(data['admittime'])
        data['dischtime'] = pd.to_datetime(data['dischtime'])
        data['los']=pd.to_timedelta(data['dischtime']-data['admittime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy','hours']] = data['los'].str.split(' ', -1, expand=True)
        data[['hours','min','sec']] = data['hours'].str.split(':', -1, expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]
        data['Age']=data['Age'].astype(int)
        return data

    def generate_cond(self):
        cond=pd.read_csv("./data/features/preproc_diag.csv.gz", compression='gzip', header=0, index_col=None)
        cond=cond[cond['hadm_id'].isin(self.data['hadm_id'])]
        cond_per_adm = cond.groupby('hadm_id').size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm

    def generate_proc(self):
        proc=pd.read_csv("./data/features/preproc_proc.csv.gz", compression='gzip', header=0, index_col=None)
        proc=proc[proc['hadm_id'].isin(self.data['hadm_id'])]
        proc[['start_days', 'dummy','start_hours']] = proc['proc_time_from_admit'].str.split(' ', -1, expand=True)
        proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', -1, expand=True)
        proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
        proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        proc=proc[proc['start_time']>=0]

        ###Remove where event time is after discharge time
        proc=pd.merge(proc,self.data[['hadm_id','los']],on='hadm_id',how='left')
        proc['sanity']=proc['los']-proc['start_time']
        proc=proc[proc['sanity']>0]
        del proc['sanity']

        self.proc=proc

    def generate_labs(self):
        chunksize = 10000000
        final=pd.DataFrame()
        for labs in tqdm(pd.read_csv("./data/features/preproc_labs.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
            labs=labs[labs['hadm_id'].isin(self.data['hadm_id'])]
            labs[['start_days', 'dummy','start_hours']] = labs['lab_time_from_admit'].str.split(' ', -1, expand=True)
            labs[['start_hours','min','sec']] = labs['start_hours'].str.split(':', -1, expand=True)
            labs['start_time']=pd.to_numeric(labs['start_days'])*24+pd.to_numeric(labs['start_hours'])
            labs=labs.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
            labs=labs[labs['start_time']>=0]

            ###Remove where event time is after discharge time
            labs=pd.merge(labs,self.data[['hadm_id','los']],on='hadm_id',how='left')
            labs['sanity']=labs['los']-labs['start_time']
            labs=labs[labs['sanity']>0]
            del labs['sanity']

            if final.empty:
                final=labs
            else:
                final=final.append(labs, ignore_index=True)

        self.labs=final

    def generate_meds(self):
        meds=pd.read_csv("./data/features/preproc_med.csv.gz", compression='gzip', header=0, index_col=None)
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
        meds=meds[meds['hadm_id'].isin(self.data['hadm_id'])]
        meds=pd.merge(meds,self.data[['hadm_id','los']],on='hadm_id',how='left')

        #####Remove where start time is after end of visit
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']

        meds['dose_val_rx']=meds['dose_val_rx'].apply(pd.to_numeric, errors='coerce')


        self.meds=meds


    def mortality_length(self,include_time):
        self.los=include_time
        if include_time >= (self.max_num_days * 24) - 1:
            self.data=self.data[(self.data['los']>=include_time)]
        else:
            self.data=self.data[(self.data['los']>=include_time) & (self.data['los'] < include_time + self.include_interval)]
        self.hids=self.data['hadm_id'].unique()

        if(self.feat_cond):
            self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]

        self.data['los']=include_time
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time


        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['hadm_id'].isin(self.data['hadm_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]

        ###LAB
        if(self.feat_lab):
            self.labs=self.labs[self.labs['hadm_id'].isin(self.data['hadm_id'])]
            self.labs=self.labs[self.labs['start_time']<=include_time]


        self.los=include_time

    def los_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['hadm_id'].unique()

        if(self.feat_cond):
            self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]

        self.data['los']=include_time
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time


        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['hadm_id'].isin(self.data['hadm_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]

        ###LAB
        if(self.feat_lab):
            self.labs=self.labs[self.labs['hadm_id'].isin(self.data['hadm_id'])]
            self.labs=self.labs[self.labs['start_time']<=include_time]


        #self.los=include_time

    def readmission_length(self,include_time):
        self.los=include_time
        if include_time >= (self.max_num_days * 24) - 1:
            self.data=self.data[(self.data['los']>=include_time)]
        else:
            self.data=self.data[(self.data['los']>=include_time) & (self.data['los'] < include_time + self.include_interval)]
        self.hids=self.data['hadm_id'].unique()
        if(self.feat_cond):
            self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed

        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
            self.meds=pd.merge(self.meds,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
            self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
            self.meds=self.meds[self.meds['stop_time']>=0]
            self.meds.loc[self.meds.start_time <0, 'start_time']=0

        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['hadm_id'].isin(self.data['hadm_id'])]
            self.proc=pd.merge(self.proc,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.proc['start_time']=self.proc['start_time']-self.proc['select_time']
            self.proc=self.proc[self.proc['start_time']>=0]

        ###LABS
        if(self.feat_lab):
            self.labs=self.labs[self.labs['hadm_id'].isin(self.data['hadm_id'])]
            self.labs=pd.merge(self.labs,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.labs['start_time']=self.labs['start_time']-self.labs['select_time']
            self.labs=self.labs[self.labs['start_time']>=0]


    def smooth_meds(self,bucket):
        final_meds=pd.DataFrame()
        final_proc=pd.DataFrame()
        final_labs=pd.DataFrame()

        if(self.feat_med):
            self.meds=self.meds.sort_values(by=['start_time'])
        if(self.feat_proc):
            self.proc=self.proc.sort_values(by=['start_time'])

        t=0
        for i in tqdm(range(0,self.los,bucket)):
            ###MEDS
             if(self.feat_med):
                sub_meds=self.meds[(self.meds['start_time']>=i) & (self.meds['start_time']<i+bucket)].groupby(['hadm_id','drug_name']).agg({'stop_time':'max','subject_id':'max','dose_val_rx':np.nanmean})
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

            ###LABS
             if(self.feat_lab):
                sub_labs=self.labs[(self.labs['start_time']>=i) & (self.labs['start_time']<i+bucket)].groupby(['hadm_id','itemid']).agg({'subject_id':'max','valuenum':np.nanmean})
                sub_labs=sub_labs.reset_index()
                sub_labs['start_time']=t
                if final_labs.empty:
                    final_labs=sub_labs
                else:
                    final_labs=final_labs.append(sub_labs)

             t=t+1
        los=int(self.los/bucket)

        ###MEDS
        if(self.feat_med):
            f2_meds=final_meds.groupby(['hadm_id','drug_name']).size()
            self.med_per_adm=f2_meds.groupby('hadm_id').sum().reset_index()[0].max()
            self.medlength_per_adm=final_meds.groupby('hadm_id').size().max()

        ###PROC
        if(self.feat_proc):
            f2_proc=final_proc.groupby(['hadm_id','icd_code']).size()
            self.proc_per_adm=f2_proc.groupby('hadm_id').sum().reset_index()[0].max()
            self.proclength_per_adm=final_proc.groupby('hadm_id').size().max()

       ###LABS
        if(self.feat_lab):
            f2_labs=final_labs.groupby(['hadm_id','itemid']).size()
            self.labs_per_adm=f2_labs.groupby('hadm_id').sum().reset_index()[0].max()
            self.labslength_per_adm=final_labs.groupby('hadm_id').size().max()

        ###CREATE DICT
        print("[ PROCESSED TIME SERIES TO EQUAL TIME INTERVAL ]")
        self.create_Dict(final_meds,final_proc,final_labs,los)


    def create_Dict(self,meds,proc,labs,los):
        print("[ CREATING DATA DICTIONARIES ]")
        dataDic={}
        labels_csv=pd.DataFrame(columns=['hadm_id','label'])
        labels_csv['hadm_id']=pd.Series(self.hids)
        labels_csv['label']=0
        for hid in self.hids:
            grp=self.data[self.data['hadm_id']==hid]
            #print(grp.head())
            #print(grp['gender'])
            #print(int(grp['Age']))
            #print(grp['ethnicity'].iloc[0])
            dataDic[hid]={'Cond':{},'Proc':{},'Med':{},'Lab':{},'ethnicity':grp['ethnicity'].iloc[0],'age':int(grp['Age']),'gender':grp['gender'].iloc[0],'label':int(grp['label'])}
            labels_csv.loc[labels_csv['hadm_id']==hid,'label']=int(grp['label'])
        for hid in tqdm(self.hids):
            grp=self.data[self.data['hadm_id']==hid]
            demo_csv=grp[['Age','gender','ethnicity','insurance']]
            if not os.path.exists("./data/csv/"+str(hid)):
                os.makedirs("./data/csv/"+str(hid))
            demo_csv.to_csv('./data/csv/'+str(hid)+'/demo.csv',index=False)

            dyn_csv=pd.DataFrame()
            ###MEDS
            if(self.feat_med):
                feat=meds['drug_name'].unique()
                df2=meds[meds['hadm_id']==hid]
                if df2.shape[0]==0:
                    val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    val=val.fillna(0)
                    val.columns=pd.MultiIndex.from_product([["MEDS"], val.columns])
                else:
                    val=df2.pivot_table(index='start_time',columns='drug_name',values='dose_val_rx')
                    df2=df2.pivot_table(index='start_time',columns='drug_name',values='stop_time')
                    #print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.ffill()
                    df2=df2.fillna(0)

                    val=pd.concat([val, add_df])
                    val=val.sort_index()
                    val=val.ffill()
                    val=val.fillna(-1)
                    #print(df2.head())
                    df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0)
                    df2[df2>0]=1
                    df2[df2<0]=0
                    val.iloc[:,0:]=df2.iloc[:,0:]*val.iloc[:,0:]
                    #print(df2.head())
                    dataDic[hid]['Med']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Med']['val']=val.iloc[:,0:].to_dict(orient="list")


                    feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))

                    val=pd.concat([val,feat_df],axis=1)

                    val=val[feat]
                    val=val.fillna(0)

                    val.columns=pd.MultiIndex.from_product([["MEDS"], val.columns])
                if(dyn_csv.empty):
                    dyn_csv=val
                else:
                    dyn_csv=pd.concat([dyn_csv,val],axis=1)



            ###PROCS
            if(self.feat_proc):
                feat=proc['icd_code'].unique()
                df2=proc[proc['hadm_id']==hid]
                if df2.shape[0]==0:
                    df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
                else:
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='icd_code',values='val')
                    #print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    #print(df2.head())
                    dataDic[hid]['Proc']=df2.to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])

                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)

            ###LABS
            if(self.feat_lab):
                feat=labs['itemid'].unique()
                df2=labs[labs['hadm_id']==hid]
                if df2.shape[0]==0:
                    val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    val=val.fillna(0)
                    val.columns=pd.MultiIndex.from_product([["LAB"], val.columns])
                else:
                    val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    #print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)

                    val=pd.concat([val, add_df])
                    val=val.sort_index()
                    if self.impute=='Mean':
                        val=val.ffill()
                        val=val.bfill()
                        val=val.fillna(val.mean())
                    elif self.impute=='Median':
                        val=val.ffill()
                        val=val.bfill()
                        val=val.fillna(val.median())
                    val=val.fillna(0)

                    df2[df2>0]=1
                    df2[df2<0]=0

                    #print(df2.head())
                    dataDic[hid]['Lab']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Lab']['val']=val.iloc[:,0:].to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                    val=pd.concat([val,feat_df],axis=1)

                    val=val[feat]
                    val=val.fillna(0)
                    val.columns=pd.MultiIndex.from_product([["LAB"], val.columns])

                if(dyn_csv.empty):
                    dyn_csv=val
                else:
                    dyn_csv=pd.concat([dyn_csv,val],axis=1)

            #Save temporal data to csv
            dyn_csv.to_csv('./data/csv/'+str(hid)+'/dynamic.csv',index=False)

            ##########COND#########
            if(self.feat_cond):
                feat=self.cond['new_icd_code'].unique()
                grp=self.cond[self.cond['hadm_id']==hid]
                if(grp.shape[0]==0):
                    dataDic[hid]['Cond']={'fids':list(['<PAD>'])}
                    feat_df=pd.DataFrame(np.zeros([1,len(feat)]),columns=feat)
                    grp=feat_df.fillna(0)
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
                else:
                    dataDic[hid]['Cond']={'fids':list(grp['new_icd_code'])}
                    grp['val']=1
                    grp=grp.drop_duplicates()
                    grp=grp.pivot(index='hadm_id',columns='new_icd_code',values='val').reset_index(drop=True)
                    feat_df=pd.DataFrame(columns=list(set(feat)-set(grp.columns)))
                    grp=pd.concat([grp,feat_df],axis=1)
                    grp=grp.fillna(0)
                    grp=grp[feat]
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
            grp.to_csv('./data/csv/'+str(hid)+'/static.csv',index=False)
            labels_csv.to_csv('./data/csv/labels.csv',index=False)


        ######SAVE DICTIONARIES##############
        metaDic={'Cond':{},'Proc':{},'Med':{},'Lab':{},'LOS':{}}
        metaDic['LOS']=los
        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(self.hids, fp)

        with open("./data/dict/ethVocab", 'wb') as fp:
            pickle.dump(list(self.data['ethnicity'].unique()), fp)
            self.eth_vocab = self.data['ethnicity'].nunique()

        with open("./data/dict/ageVocab", 'wb') as fp:
            pickle.dump(list(self.data['Age'].unique()), fp)
            self.age_vocab = self.data['Age'].nunique()

        with open("./data/dict/insVocab", 'wb') as fp:
            pickle.dump(list(self.data['insurance'].unique()), fp)
            self.ins_vocab = self.data['insurance'].nunique()

        if(self.feat_med):
            with open("./data/dict/medVocab", 'wb') as fp:
                pickle.dump(list(meds['drug_name'].unique()), fp)
            self.med_vocab = meds['drug_name'].nunique()
            metaDic['Med']=self.med_per_adm

        if(self.feat_cond):
            with open("./data/dict/condVocab", 'wb') as fp:
                pickle.dump(list(self.cond['new_icd_code'].unique()), fp)
            self.cond_vocab = self.cond['new_icd_code'].nunique()
            metaDic['Cond']=self.cond_per_adm

        if(self.feat_proc):
            with open("./data/dict/procVocab", 'wb') as fp:
                pickle.dump(list(proc['icd_code'].unique()), fp)
            self.proc_vocab = proc['icd_code'].unique()
            metaDic['Proc']=self.proc_per_adm

        if(self.feat_lab):
            with open("./data/dict/labsVocab", 'wb') as fp:
                pickle.dump(list(labs['itemid'].unique()), fp)
            self.lab_vocab = labs['itemid'].unique()
            metaDic['Lab']=self.labs_per_adm

        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)




