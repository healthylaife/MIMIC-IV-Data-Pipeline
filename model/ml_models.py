import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import importlib
import sys
import numpy as np
import evaluation
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

importlib.reload(evaluation)
import evaluation
# MAX_LEN=12
# MAX_COND_SEQ=56
# MAX_PROC_SEQ=40
# MAX_MED_SEQ=15#37
# MAX_LAB_SEQ=899
# MAX_BMI_SEQ=118


class ML_models():
    def __init__(self,data_icu,k_fold,model_type,concat,oversampling):
        self.data_icu=data_icu
        self.k_fold=k_fold
        self.model_type=model_type
        self.concat=concat
        self.oversampling=oversampling
        self.loss=evaluation.Loss('cpu',True,True,True,True,True,True,True,True,True,True,True)
        self.ml_train()
    def create_kfolds(self):
        labels=pd.read_csv('./data/csv/labels.csv', header=0)
        
        if (self.k_fold==0):
            k_fold=5
            self.k_fold=1
        else:
            k_fold=self.k_fold
        hids=labels.iloc[:,0]
        y=labels.iloc[:,1]
        print("Total Samples",len(hids))
        print("Positive Samples",y.sum())
        #print(len(hids))
        if self.oversampling:
            print("=============OVERSAMPLING===============")
            oversample = RandomOverSampler(sampling_strategy='minority')
            hids=np.asarray(hids).reshape(-1,1)
            hids, y = oversample.fit_resample(hids, y)
            #print(hids.shape)
            hids=hids[:,0]
            print("Total Samples",len(hids))
            print("Positive Samples",y.sum())
        
        ids=range(0,len(hids))
        batch_size=int(len(ids)/k_fold)
        k_hids=[]
        for i in range(0,k_fold):
            rids = random.sample(ids, batch_size)
            ids = list(set(ids)-set(rids))
            if i==0:
                k_hids.append(hids[rids])             
            else:
                k_hids.append(hids[rids])
        return k_hids


    def ml_train(self):
        k_hids=self.create_kfolds()
        
        labels=pd.read_csv('./data/csv/labels.csv', header=0)
        for i in range(self.k_fold):
            print("==================={0:2d} FOLD=====================".format(i))
            test_hids=k_hids[i]
            train_ids=list(set([0,1,2,3,4])-set([i]))
            train_hids=[]
            for j in train_ids:
                train_hids.extend(k_hids[j])                    
            
            concat_cols=[]
            if(self.concat):
                dyn=pd.read_csv('./data/csv/'+str(train_hids[0])+'/dynamic.csv',header=[0,1])
                dyn.columns=dyn.columns.droplevel(0)
                cols=dyn.columns
                time=dyn.shape[0]

                for t in range(time):
                    cols_t = [x + "_"+str(t) for x in cols]

                    concat_cols.extend(cols_t)
            print('train_hids',len(train_hids))
            X_train,Y_train=self.getXY(train_hids,labels,concat_cols)
            #encoding categorical
            gen_encoder = LabelEncoder()
            eth_encoder = LabelEncoder()
            ins_encoder = LabelEncoder()
            age_encoder = LabelEncoder()
            gen_encoder.fit(X_train['gender'])
            eth_encoder.fit(X_train['ethnicity'])
            ins_encoder.fit(X_train['insurance'])
            #age_encoder.fit(X_train['Age'])
            X_train['gender']=gen_encoder.transform(X_train['gender'])
            X_train['ethnicity']=eth_encoder.transform(X_train['ethnicity'])
            X_train['insurance']=ins_encoder.transform(X_train['insurance'])
            #X_train['Age']=age_encoder.transform(X_train['Age'])

            print(X_train.shape)
            print(Y_train.shape)
            print('test_hids',len(test_hids))
            X_test,Y_test=self.getXY(test_hids,labels,concat_cols)
            self.test_data=X_test.copy(deep=True)
            X_test['gender']=gen_encoder.transform(X_test['gender'])
            X_test['ethnicity']=eth_encoder.transform(X_test['ethnicity'])
            X_test['insurance']=ins_encoder.transform(X_test['insurance'])
            #X_test['Age']=age_encoder.transform(X_test['Age'])
            
            
            print(X_test.shape)
            print(Y_test.shape)
            #print("just before training")
            #print(X_test.head())
            self.train_model(X_train,Y_train,X_test,Y_test)
    
    def train_model(self,X_train,Y_train,X_test,Y_test):
        #logits=[]
        print("===============MODEL TRAINING===============")
        if self.model_type=='Gradient Bossting':
            model = HistGradientBoostingClassifier(categorical_features=[X_train.shape[1]-3,X_train.shape[1]-2,X_train.shape[1]-1]).fit(X_train, Y_train)
            
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True)
            self.save_output(Y_test,prob[:,1],logits)
        
        elif self.model_type=='Logistic Regression':
            X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            
            model = LogisticRegression().fit(X_train, Y_train) 
            logits=model.predict_log_proba(X_test)
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.coef_[0],X_train.columns)
        
        elif self.model_type=='Random Forest':
            X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            model = RandomForestClassifier().fit(X_train, Y_train)
            logits=model.predict_log_proba(X_test)
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.feature_importances_,X_train.columns)
        
        elif self.model_type=='Xgboost':
            X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, Y_train)
            #logits=model.predict_log_proba(X_test)
            #print(self.test_data['ethnicity'])
            #print(self.test_data.shape)
            #print(self.test_data.head())
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True)
            self.save_outputImp(Y_test,prob[:,1],logits,model.feature_importances_,X_train.columns)


    
    def getXY(self,ids,labels,concat_cols):
        X_df=pd.DataFrame()   
        y_df=pd.DataFrame()   
        features=[]
        #print(ids)
        for sample in ids:
            if self.data_icu:
                y=labels[labels['stay_id']==sample]['label']
            else:
                y=labels[labels['hadm_id']==sample]['label']
            
            #print(sample)
            dyn=pd.read_csv('./data/csv/'+str(sample)+'/dynamic.csv',header=[0,1])
            
            if self.concat:
                dyn.columns=dyn.columns.droplevel(0)
                dyn=dyn.to_numpy()
                dyn=dyn.reshape(1,-1)
                #print(dyn.shape)
                #print(len(concat_cols))
                dyn_df=pd.DataFrame(data=dyn,columns=concat_cols)
                features=concat_cols
            else:
                dyn_df=pd.DataFrame()
                #print(dyn)
                for key in dyn.columns.levels[0]:
                    #print(sample)                    
                    dyn_temp=dyn[key]
                    if self.data_icu:
                        if ((key=="CHART") or (key=="MEDS")):
                            agg=dyn_temp.aggregate("mean")
                            agg=agg.reset_index()
                        else:
                            agg=dyn_temp.aggregate("max")
                            agg=agg.reset_index()
                    else:
                        if ((key=="LAB") or (key=="MEDS")):
                            agg=dyn_temp.aggregate("mean")
                            agg=agg.reset_index()
                        else:
                            agg=dyn_temp.aggregate("max")
                            agg=agg.reset_index()
                    if dyn_df.empty:
                        dyn_df=agg
                    else:
                        dyn_df=pd.concat([dyn_df,agg],axis=0)
                #dyn_df=dyn_df.drop(index=(0))
#                 print(dyn_df.shape)
#                 print(dyn_df.head())
                dyn_df=dyn_df.T
                dyn_df.columns = dyn_df.iloc[0]
                dyn_df=dyn_df.iloc[1:,:]
                        
#             print(dyn.shape)
#             print(dyn_df.shape)
#             print(dyn_df.head())
            stat=pd.read_csv('./data/csv/'+str(sample)+'/static.csv',header=[0,1])
            stat=stat['COND']
#             print(stat.shape)
#             print(stat.head())
            demo=pd.read_csv('./data/csv/'+str(sample)+'/demo.csv',header=0)
#             print(demo.shape)
#             print(demo.head())
            if X_df.empty:
                X_df=pd.concat([dyn_df,stat],axis=1)
                X_df=pd.concat([X_df,demo],axis=1)
            else:
                X_df=pd.concat([X_df,pd.concat([pd.concat([dyn_df,stat],axis=1),demo],axis=1)],axis=0)
            if y_df.empty:
                y_df=y
            else:
                y_df=pd.concat([y_df,y],axis=0)
#             print("X_df",X_df.shape)
#             print("y_df",y_df.shape)
        print("X_df",X_df.shape)
        print("y_df",y_df.shape)
        return X_df ,y_df
    
    def save_output(self,labels,prob,logits):
        
        output_df=pd.DataFrame()
        output_df['Labels']=labels.values
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
    
    def save_outputImp(self,labels,prob,logits,importance,features):
        
        output_df=pd.DataFrame()
        output_df['Labels']=labels.values
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
        imp_df=pd.DataFrame()
        imp_df['imp']=importance
        imp_df['feature']=features
        imp_df.to_csv('./data/output/'+'feature_importance.csv', index=False)
                
                

