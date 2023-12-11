from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from pipeline.feature.lab_events import Lab
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.file_info.common import PREPROC_PATH
from pipeline.file_info.preproc.cohort import COHORT_PATH
from pipeline.file_info.preproc.feature import (
    EXTRACT_CHART_ICU_PATH,
    EXTRACT_DIAG_ICU_PATH,
    EXTRACT_DIAG_PATH,
    EXTRACT_LABS_PATH,
    EXTRACT_MED_ICU_PATH,
    EXTRACT_MED_PATH,
    EXTRACT_OUT_ICU_PATH,
    EXTRACT_PROC_ICU_PATH,
    EXTRACT_PROC_PATH,
)
from pipeline.prediction_task import PredictionTask, TargetType
import logging

from pipeline.features_extractor import FeatureExtractor
from pipeline.feature.chart_events import Chart, ChartEvents
from pipeline.feature.diagnoses import Diagnoses
from pipeline.preprocessing.cohort import read_cohort
from pipeline.feature.feature_abc import Feature

logger = logging.getLogger()

CSVPATH = PREPROC_PATH / "csv"


class DictMaker:
    def __init__(self, feature_extractor: FeatureExtractor, hids):
        self.feature_extractor = feature_extractor
        self.hids = hids

    def create_dict(self, meds, proc, out, labs, chart, cohort: pd.DataFrame, los):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"
        self.dataDic = {}
        self.labels_csv = pd.DataFrame(columns=[group_col, "label"])
        self.labels_csv[group_col] = pd.Series(self.hids)
        self.labels_csv["label"] = 0

        for hid in self.hids:
            grp = cohort[cohort[group_col] == hid]
            if len(grp) == 0:
                continue
            self.dataDic[hid] = {
                "Cond": {},
                "Proc": {},
                "Med": {},
                "Out": {},
                "Chart": {},
                "Lab": {},
                "ethnicity": grp["ethnicity"].iloc[0],
                "age": int(grp["age"].iloc[0]),
                "gender": grp["gender"].iloc[0],
                "label": int(grp["label"].iloc[0]),
            }
            self.labels_csv.loc[self.labels_csv[group_col] == hid, "label"] = int(
                grp["label"].iloc[0]
            )
        for hid in tqdm(self.hids):
            grp = cohort[cohort[group_col] == hid]
            self.demo_csv = grp[["age", "gender", "ethnicity", "insurance"]]
            if not os.path.exists(CSVPATH / str(hid)):
                os.makedirs(CSVPATH / str(hid))
            self.demo_csv.to_csv(CSVPATH / str(hid) / "demo.csv", index=False)

            dyn_csv = pd.DataFrame()
            if self.feature_extractor.for_medications:
                self.process_med_by_hid(meds, los, hid, dyn_csv)
                self.process_proc_by_hid(proc, los, hid, dyn_csv)
            # self.process_feature_data(proc, "Proc", los)
            # self.process_feature_data(out, "Out", los)
            # self.process_feature_data(labs, "Lab", los)
            # self.process_feature_data(chart, "Chart", los)
            # self.save_csv_files()
            dyn_csv.to_csv(CSVPATH / str(hid) / "dynamic.csv", index=False)
        return

    def process_med_by_hid(self, feature, los, hid, dyn_csv):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"
        code_col = "itemid" if self.feature_extractor.use_icu else "drug_name"
        feat = feature[code_col].unique()
        df2 = feature[feature[group_col] == hid]
        if df2.shape[0] == 0:
            if self.feature_extractor.use_icu:
                amount = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                amount = amount.fillna(0)
                amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])
            else:
                val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                val = val.fillna(0)
                val.columns = pd.MultiIndex.from_product([["MEDS"], val.columns])
        else:
            if self.feature_extractor.use_icu:
                rate = df2.pivot_table(
                    index="start_time", columns="itemid", values="rate"
                )
                amount = df2.pivot_table(
                    index="start_time", columns="itemid", values="amount"
                )
                df2 = df2.pivot_table(
                    index="start_time", columns="itemid", values="stop_time"
                )
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.ffill()
                df2 = df2.fillna(0)
                rate = pd.concat([rate, add_df])
                rate = rate.sort_index()
                rate = rate.ffill()
                rate = rate.fillna(-1)
                amount = pd.concat([amount, add_df])
                amount = amount.sort_index()
                amount = amount.ffill()
                amount = amount.fillna(-1)
                df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
                df2[df2 > 0] = 1
                df2[df2 < 0] = 0
                rate.iloc[:, 0:] = df2.iloc[:, 0:] * rate.iloc[:, 0:]
                amount.iloc[:, 0:] = df2.iloc[:, 0:] * amount.iloc[:, 0:]
                self.dataDic[hid]["Med"]["signal"] = df2.iloc[:, 0:].to_dict(
                    orient="list"
                )
                self.dataDic[hid]["Med"]["rate"] = rate.iloc[:, 0:].to_dict(
                    orient="list"
                )
                self.dataDic[hid]["Med"]["amount"] = amount.iloc[:, 0:].to_dict(
                    orient="list"
                )
                feat_df = pd.DataFrame(columns=list(set(feat) - set(amount.columns)))
                amount = pd.concat([amount, feat_df], axis=1)
                amount = amount[feat]
                amount = amount.fillna(0)
                amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])
            else:
                val = df2.pivot_table(
                    index="start_time", columns="drug_name", values="dose_val_rx"
                )
                df2 = df2.pivot_table(
                    index="start_time", columns="drug_name", values="stop_time"
                )
                # print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.ffill()
                df2 = df2.fillna(0)
                val = pd.concat([val, add_df])
                val = val.sort_index()
                val = val.ffill()
                val = val.fillna(-1)
                df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
                df2[df2 > 0] = 1
                df2[df2 < 0] = 0
                val.iloc[:, 0:] = df2.iloc[:, 0:] * val.iloc[:, 0:]
                self.dataDic[hid]["Med"]["signal"] = df2.iloc[:, 0:].to_dict(
                    orient="list"
                )
                self.dataDic[hid]["Med"]["val"] = val.iloc[:, 0:].to_dict(orient="list")
                feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))
                val = pd.concat([val, feat_df], axis=1)
                val = val[feat]
                val = val.fillna(0)
                val.columns = pd.MultiIndex.from_product([["MEDS"], val.columns])

        if self.feature_extractor.use_icu:
            if dyn_csv.empty:
                dyn_csv = amount
            else:
                dyn_csv = pd.concat([dyn_csv, amount], axis=1)
        else:
            if dyn_csv.empty:
                dyn_csv = val
            else:
                dyn_csv = pd.concat([dyn_csv, val], axis=1)

    def process_proc_by_hid(self, feature, los, hid, dyn_csv):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"
        code_col = "itemid" if self.feature_extractor.use_icu else "icd_code"
        df2 = feature[feature[group_col] == hid]
        feat = feature[code_col].unique()
        if self.feature_extractor.use_icu:
            if df2.shape[0] == 0:
                df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
            else:
                df2["val"] = 1
                df2 = df2.pivot_table(
                    index="start_time", columns="itemid", values="val"
                )
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)
                df2[df2 > 0] = 1
                self.dataDic[hid]["Proc"] = df2.to_dict(orient="list")
                feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)
                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
        else:
            if df2.shape[0] == 0:
                df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
            else:
                df2["val"] = 1
                df2 = df2.pivot_table(
                    index="start_time", columns="icd_code", values="val"
                )
                # print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)
                df2[df2 > 0] = 1
                # print(df2.head())
                self.dataDic[hid]["Proc"] = df2.to_dict(orient="list")

                feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)

                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])

        if dyn_csv.empty:
            dyn_csv = df2
        else:
            dyn_csv = pd.concat([dyn_csv, df2], axis=1)

    def process_out_by_hid(self, feature, name, los, hid, dyn_csv, group_col, code_col):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"

    def process_chart_by_hid(
        self, feature, name, los, hid, dyn_csv, group_col, code_col
    ):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"

        # if dyn_csv.empty:
        #     dyn_csv = amount_or_val
        # else:
        #     dyn_csv = pd.concat([dyn_csv, amount_or_val], axis=1)

    def process_lab_by_hid(self, feature, name, los, hid, dyn_csv, group_col, code_col):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"

        # if dyn_csv.empty:
        #     dyn_csv = amount_or_val
        # else:
        #     dyn_csv = pd.concat([dyn_csv, amount_or_val], axis=1)

    def process_dia_by_hid(self, feature, name, los, hid, dyn_csv, group_col, code_col):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"

        # amount_or_val = amount if self.feature_extractor.use_icu else val
        # if dyn_csv.empty:
        #     dyn_csv = amount_or_val
        # else:
        #     dyn_csv = pd.concat([dyn_csv, amount_or_val], axis=1)

    # def save_csv_files(self):
    #     for hid in self.hids:
    #         self.save_individual_csv(hid)
    #     self.labels_csv.to_csv(CSVPATH / "labels.csv", index=False)

    # def save_individual_csv(self, hid):
    #     Save demographic and dynamic data to CSV files
    #     static_csv_path = os.path.join(PREPROC_PATH / f"/csv/{str(hid)}static.csv")
    #     return
    #     Save corresponding DataFrames to these paths
