import os
import pickle
import glob
import importlib
#print(os.getcwd())
#os.chdir('../../')
#print(os.getcwd())
import utils.icu_preprocess_util
from utils.icu_preprocess_util import * 
importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *  
importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *


if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/features/chartevents"):
    os.makedirs("./data/features/chartevents")

def feature_icu(cohort_output, version_path, diag_flag=True,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True):
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module("./"+version_path+"/hosp/diagnoses_icd.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
        diag[['subject_id', 'hadm_id', 'stay_id', 'icd_code','root_icd10_convert','root']].to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if out_flag:  
        print("[EXTRACTING OUPTPUT EVENTS DATA]")
        out = preproc_out("./"+version_path+"/icu/outputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=None)
        out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    
    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart=preproc_chart("./"+version_path+"/icu/chartevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=['stay_id','charttime','itemid','valuenum','valueuom'])
        chart = drop_wrong_uom(chart, 0.95)
        chart[['stay_id', 'itemid','event_time_from_admit','valuenum']].to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc("./"+version_path+"/icu/procedureevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'starttime', dtypes=None, usecols=['stay_id','starttime','itemid'])
        proc[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds("./"+version_path+"/icu/inputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz')
        med[['subject_id', 'hadm_id', 'stay_id', 'itemid' ,'starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit','rate','amount','orderid']].to_csv('./data/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

def preprocess_features_icu(cohort_output, diag_flag, group_diag,chart_flag,clean_chart,impute_outlier_chart,thresh,left_thresh):
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        if(group_diag=='Keep both ICD-9 and ICD-10 codes'):
            diag['new_icd_code']=diag['icd_code']
        if(group_diag=='Convert ICD-9 to ICD-10 codes'):
            diag['new_icd_code']=diag['root_icd10_convert']
        if(group_diag=='Convert ICD-9 to ICD-10 and group ICD-10 codes'):
            diag['new_icd_code']=diag['root']

        diag=diag[['subject_id', 'hadm_id', 'stay_id', 'new_icd_code']].dropna()
        print("Total number of rows",diag.shape[0])
        diag.to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        
    if chart_flag:
        if clean_chart:   
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
            chart = outlier_imputation(chart, 'itemid', 'valuenum', thresh,left_thresh,impute_outlier_chart)
            
#             for i in [227441, 229357, 229358, 229360]:
#                 try:
#                     maj = chart.loc[chart.itemid == i].valueuom.value_counts().index[0]
#                     chart = chart.loc[~((chart.itemid == i) & (chart.valueuom == maj))]
#                 except IndexError:
#                     print(f"{idx} not found")
            print("Total number of rows",chart.shape[0])
            chart.to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
        
        
def generate_summary_icu(diag_flag,proc_flag,med_flag,out_flag,chart_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        freq=diag.groupby(['stay_id','new_icd_code']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['new_icd_code'])['mean_frequency'].mean().reset_index()
        total=diag.groupby('new_icd_code').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='new_icd_code',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/diag_summary.csv',index=False)
        summary['new_icd_code'].to_csv('./data/summary/diag_features.csv',index=False)


    if med_flag:
        med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
        freq=med.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=med[med['amount']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=med.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/med_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/med_features.csv',index=False)

    
    
    if proc_flag:
        proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
        freq=proc.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=proc.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/proc_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/proc_features.csv',index=False)

        
    if out_flag:
        out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
        freq=out.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=out.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/out_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/out_features.csv',index=False)
        
    if chart_flag:
        chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
        freq=chart.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

        missing=chart[chart['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=chart.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing_perc']=100*(summary['missing_count']/summary['total_count'])
        #summary=summary.fillna(0)

#         final.groupby('itemid')['missing_count'].sum().reset_index()
#         final.groupby('itemid')['total_count'].sum().reset_index()
#         final.groupby('itemid')['missing%'].mean().reset_index()
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/chart_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/chart_features.csv',index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")
    
def features_selection_icu(cohort_output, diag_flag,proc_flag,med_flag,out_flag,chart_flag,group_diag,group_med,group_proc,group_out,group_chart):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/diag_features.csv",header=0)
            diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
        
            print("Total number of rows",diag.shape[0])
            diag.to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/med_features.csv",header=0)
            med=med[med['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            med.to_csv('./data/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    
    
    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/proc_features.csv",header=0)
            proc=proc[proc['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",proc.shape[0])
            proc.to_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if out_flag:
        if group_out:            
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/out_features.csv",header=0)
            out=out[out['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",out.shape[0])
            out.to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
            
    if chart_flag:
        if group_chart:            
            print("[FEATURE SELECTION CHART EVENTS DATA]")
            
            chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0, index_col=None)
            
            features=pd.read_csv("./data/summary/chart_features.csv",header=0)
            chart=chart[chart['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",chart.shape[0])
            chart.to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")