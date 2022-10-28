#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


def compute_outlier_imputation(arr, cut_off,left_thresh,impute):
    perc_up = np.percentile(arr, left_thresh)
    perc_down = np.percentile(arr, cut_off)
    #print(perc_up,perc_down)
    if impute:
        arr[arr < perc_up] = perc_up
        arr[arr > perc_down] = perc_down
    else:
        #print(arr[arr < perc_up].shape,arr[arr > perc_down].shape)
        arr[arr < perc_up] = np.nan
        arr[arr > perc_down] = np.nan
    return arr


def outlier_imputation(data, id_attribute, value_attribute, cut_off,left_thresh,impute):
    grouped = data.groupby([id_attribute])[value_attribute]
    #print(cut_off)
    for id_number, values in grouped:
        #print("=========")
        #print(id_number)
        #print(values.max(),values.min(),values.mean())
        index = values.index
        values = compute_outlier_imputation(values, cut_off,left_thresh,impute)
        data[value_attribute].iloc[index] = values
    data=data.dropna(subset=[value_attribute])
    #print(data.shape)
    return data

