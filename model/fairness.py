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
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/output"):
    os.makedirs("./data/output")

def fairness_evaluation(inputFile, outputFile):
    if os.path.isfile('./data/output/'+inputFile):
        output_dict = pickle.load(open('./data/output/'+inputFile,"rb"))
    else:
        print("fairnessDict file not found.")


    output_dict["Predicted"] = output_dict.apply(lambda row:1 if row.Prob>=0.5 else 0, axis=1)

    output_dict["age_binned"] = output_dict.age.apply(lambda x:"{}-{}".format((x//10)*10,(x//10 + 1)*10))
    sensitive_columns = ["ethnicity", "gender", "age_binned"]


    def get_cm_parameters(gt, pred):
        zipped_gt_pred = list(zip(gt,pred))
        tp = len([pair for pair in zipped_gt_pred if pair == (1,1)])
        tn = len([pair for pair in zipped_gt_pred if pair == (0,0)])
        fp = len([pair for pair in zipped_gt_pred if pair == (0,1)])
        fn = len([pair for pair in zipped_gt_pred if pair == (1,0)])

        try:
            tpr = tp/(tp + fn)
        except ZeroDivisionError:
            tpr = None
        try:
            tnr = tn/(tn + fp)
        except ZeroDivisionError:
            tnr = None
        try:
            fpr = fp/(fp + tn)
        except ZeroDivisionError:
            fpr = None
        try:
            fnr = fn/(fn + tp)
        except ZeroDivisionError:
            fnr = None
        try:
            pr = (tp + fp)/(len(zipped_gt_pred))
        except:
            pr = None
        try:
            nr = (tn + fn)/(len(zipped_gt_pred))
        except:
            nr = None
        try:
            acc = (tp+tn)/(len(zipped_gt_pred))
        except ZeroDivisionError:
            acc = None

        return tp, tn, fp, fn, tpr, tnr, fpr, fnr, pr, nr, acc

    report_list = []
    for sens_col in sensitive_columns:
        for group, aggregate in output_dict.groupby(sens_col):
            tmp_dct = {"sensitive_attribute": sens_col}
            tp, tn, fp, fn, tpr, tnr, fpr, fnr, pr, nr, acc = get_cm_parameters(list(aggregate.Labels), list(aggregate.Predicted))
            tmp_dct.update(dict(
                group=group,tp=tp, tn=tn, fp=fp, fn=fn, tpr=tpr, tnr=tnr, fpr= fpr, fnr=fnr, pr=pr, nr=nr, accuracy=acc    
                )
            )
            report_list.append(tmp_dct)

    report = pd.DataFrame(report_list)
    report_groups = {c:i for i,c in enumerate(report.sensitive_attribute.unique())}


    def highlight(s):
        colors = [['background-color: yellow'], ['background-color: green'], ['background-color: red']]
        return colors[report_groups[s.sensitive_attribute]%len(colors)] * len(s)


    try:
        import jinja2
        display(report.style.apply(highlight, axis=1))
    except ImportError:
        display(report)

    report.to_csv('./data/output/'+outputFile+'.csv',index=False)

