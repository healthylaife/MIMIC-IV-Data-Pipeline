import os
import pickle
#os.chdir('../../')
from utils.hosp_preprocess_util import *    # module of preprocessing functions


def feature_nonicu(cohort_output, diag_flag=True,lab_flag=True,proc_flag=True,med_flag=True):
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module("./mimic-iv-1.0/hosp/diagnoses_icd.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
        diag[['subject_id', 'hadm_id', 'icd_code','root_icd10_convert','root']].to_csv("./data/features/preproc_diag.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

#     if lab_flag:    
#         out = preproc_out("./mimic-iv-1.0/icu/outputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=None)
#         out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc("./mimic-iv-1.0/hosp/procedures_icd.csv.gz", './data/cohort/cohort_non-icu_30_day_readmission.csv.gz', 'chartdate', 'base_anchor_year', dtypes=None, usecols=None)
        proc[['subject_id', 'hadm_id', 'icd_code', 'chartdate', 'admittime', 'proc_time_from_admit']].to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds("./mimic-iv-1.0/hosp/prescriptions.csv.gz", './data/cohort/cohort_non-icu_30_day_readmission.csv.gz','./utils/mappings/ndc_product.txt')
        med[['subject_id', 'hadm_id', 'starttime','stoptime','drug','nonproprietaryname', 'start_hours_from_admit', 'stop_hours_from_admit','dose_val_rx ']].to_csv('./data/features/preproc_med.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
        
        
        
def preprocess_features_hosp(cohort_output, diag_flag,proc_flag,med_flag,group_diag,group_med,group_proc):
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv("./data/features/preproc_diag.csv.gz", compression='gzip',header=0)
        if(group_diag=='Keep both ICD-9 and ICD-10 codes'):
            diag['new_icd_code']=diag['icd_code']
        if(group_diag=='Convert ICD-9 to ICD-10 codes'):
            diag['new_icd_code']=diag['root_icd10_convert']
        if(group_diag=='Convert ICD-9 to ICD-10 and group ICD-10 codes'):
            diag['new_icd_code']=diag['root']

        diag=diag[['subject_id', 'hadm_id', 'new_icd_code']].dropna()
        print("Total number of rows",diag.shape[0])
        diag.to_csv("./data/features/preproc_diag.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:
        print("[PROCESSING MEDICATIONS DATA]")
        if group_med:           
            med = pd.read_csv("./data/features/preproc_med.csv.gz", compression='gzip',header=0)
            if med_group:
                med['drug_name']=med['nonproprietaryname']
            else:
                med['drug_name']=med['drug']
            med=med[['subject_id', 'hadm_id', 'starttime','stoptime','drug_name', 'start_hours_from_admit', 'stop_hours_from_admit']].dropna()
            print("Total number of rows",med.shape[0])
            med.to_csv('./data/features/preproc_med.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    
    
    if proc_flag:
        print("[PROCESSING PROCEDURES DATA]")
        proc = pd.read_csv("./data/features/preproc_proc.csv.gz", compression='gzip',header=0)
        if(group_proc=='ICD-9 and ICD-10'):
            proc=proc[['subject_id', 'hadm_id', 'icd_code', 'chartdate', 'admittime', 'proc_time_from_admit']]
            print("Total number of rows",proc.shape[0])
            proc.dropna().to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
        elif(group_proc=='ICD-10'):
            proc=proc.loc[proc.icd_version == 10][['subject_id', 'hadm_id', 'icd_code', 'chartdate', 'admittime', 'proc_time_from_admit']].dropna()
            print("Total number of rows",proc.shape[0])
            proc.to_csv("./data/features/preproc_proc.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
