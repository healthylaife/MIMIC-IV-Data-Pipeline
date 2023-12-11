from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from pipeline.file_info.common import PREPROC_PATH

import logging

from pipeline.features_extractor import FeatureExtractor

logger = logging.getLogger()

CSVPATH = PREPROC_PATH / "csv"
DICT_PATH = PREPROC_PATH / "dict"


class DictMaker:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        hids,
        med_per_adm,
        out_per_adm,
        chart_per_adm,
        dia_per_adm,
        proc_per_adm,
        labs_per_adm,
    ):
        self.feature_extractor = feature_extractor
        self.hids = hids
        self.med_per_adm = med_per_adm
        self.out_per_adm = out_per_adm
        self.chart_per_adm = chart_per_adm
        self.dia_per_adm = dia_per_adm
        self.proc_per_adm = proc_per_adm
        self.labs_per_adm = labs_per_adm

    def create_dict(
        self, diag, meds, proc, out, labs, chart, cohort: pd.DataFrame, los
    ):
        group_col = "stay_id" if self.feature_extractor.use_icu else "hadm_id"
        self.dataDic = {}
        self.labels_csv = pd.DataFrame(columns=[group_col, "label"])
        self.labels_csv[group_col] = pd.Series(self.hids)
        self.labels_csv["label"] = 0

        for hid in self.hids:
            grp = cohort[cohort[group_col] == hid]
            if len(grp) == 0:
                self.dataDic[hid] = {
                    "Cond": {},
                    "Proc": {},
                    "Med": {},
                    "Out": {},
                    "Chart": {},
                    "Lab": {},
                    "ethnicity": {},
                    "age": {},
                    "gender": {},
                    "label": {},
                }

            else:
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
            if self.feature_extractor.for_procedures:
                self.process_proc_by_hid(proc, los, hid, dyn_csv)
            if self.feature_extractor.for_labs and not self.feature_extractor.use_icu:
                self.process_lab_by_hid(labs, los, hid, dyn_csv)
            if self.feature_extractor.for_labs and self.feature_extractor.use_icu:
                self.process_out_by_hid(out, los, hid, dyn_csv)
            if (
                self.feature_extractor.for_chart_events
                and self.feature_extractor.use_icu
            ):
                self.process_chart_by_hid(chart, los, hid, dyn_csv)
            # self.save_csv_files()
            dyn_csv.to_csv(CSVPATH / str(hid) / "dynamic.csv", index=False)
            if self.feature_extractor.for_diagnoses:
                self.process_dia_by_hid(diag, los, hid, dyn_csv)

            grp.to_csv(CSVPATH / str(hid) / "static.csv", index=False)
            self.labels_csv.to_csv(CSVPATH / str(hid) / "labels.csv", index=False)

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

        if dyn_csv.empty:
            dyn_csv = df2
        else:
            dyn_csv = pd.concat([dyn_csv, df2], axis=1)

    def process_out_by_hid(self, feature, los, hid, dyn_csv):
        feat = feature["itemid"].unique()
        df2 = feature[feature["stay_id"] == hid]
        if df2.shape[0] == 0:
            df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
            df2 = df2.fillna(0)
            df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])
        else:
            df2["val"] = 1
            df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
            add_indices = pd.Index(range(los)).difference(df2.index)
            add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
            df2 = pd.concat([df2, add_df])
            df2 = df2.sort_index()
            df2 = df2.fillna(0)
            df2[df2 > 0] = 1
            self.dataDic[hid]["Out"] = df2.to_dict(orient="list")
            feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
            df2 = pd.concat([df2, feat_df], axis=1)
            df2 = df2[feat]
            df2 = df2.fillna(0)
            df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])
        if dyn_csv.empty:
            dyn_csv = df2
        else:
            dyn_csv = pd.concat([dyn_csv, df2], axis=1)

    def process_chart_by_hid(self, feature, los, hid, dyn_csv):
        feat = feature["itemid"].unique()
        df2 = feature[feature["stay_id"] == hid]
        if df2.shape[0] == 0:
            val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
            val = val.fillna(0)
            val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])
        else:
            val = df2.pivot_table(
                index="start_time", columns="itemid", values="valuenum"
            )
            df2["val"] = 1
            df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
            add_indices = pd.Index(range(los)).difference(df2.index)
            add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
            df2 = pd.concat([df2, add_df])
            df2 = df2.sort_index()
            df2 = df2.fillna(0)

            val = pd.concat([val, add_df])
            val = val.sort_index()
            if self.impute == "Mean":
                val = val.ffill()
                val = val.bfill()
                val = val.fillna(val.mean())
            elif self.impute == "Median":
                val = val.ffill()
                val = val.bfill()
                val = val.fillna(val.median())
            val = val.fillna(0)

            df2[df2 > 0] = 1
            df2[df2 < 0] = 0
            self.dataDic[hid]["Chart"]["signal"] = df2.iloc[:, 0:].to_dict(
                orient="list"
            )
            self.dataDic[hid]["Chart"]["val"] = val.iloc[:, 0:].to_dict(orient="list")

            feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))
            val = pd.concat([val, feat_df], axis=1)

            val = val[feat]
            val = val.fillna(0)
            val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])

        if dyn_csv.empty:
            dyn_csv = val
        else:
            dyn_csv = pd.concat([dyn_csv, val], axis=1)

    def process_lab_by_hid(self, feature, los, hid, dyn_csv):
        feat = feature["itemid"].unique()
        df2 = feature[feature["hadm_id"] == hid]
        if df2.shape[0] == 0:
            val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
            val = val.fillna(0)
            val.columns = pd.MultiIndex.from_product([["LAB"], val.columns])
        else:
            val = df2.pivot_table(
                index="start_time", columns="itemid", values="valuenum"
            )
            df2["val"] = 1
            df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
            add_indices = pd.Index(range(los)).difference(df2.index)
            add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
            df2 = pd.concat([df2, add_df])
            df2 = df2.sort_index()
            df2 = df2.fillna(0)

            val = pd.concat([val, add_df])
            val = val.sort_index()
            if self.impute == "Mean":
                val = val.ffill()
                val = val.bfill()
                val = val.fillna(val.mean())
            elif self.impute == "Median":
                val = val.ffill()
                val = val.bfill()
                val = val.fillna(val.median())
            val = val.fillna(0)

            df2[df2 > 0] = 1
            df2[df2 < 0] = 0

            # print(df2.head())
            self.dataDic[hid]["Lab"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
            self.dataDic[hid]["Lab"]["val"] = val.iloc[:, 0:].to_dict(orient="list")

            feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))
            val = pd.concat([val, feat_df], axis=1)

            val = val[feat]
            val = val.fillna(0)
            val.columns = pd.MultiIndex.from_product([["LAB"], val.columns])

            if dyn_csv.empty:
                dyn_csv = val
            else:
                dyn_csv = pd.concat([dyn_csv, val], axis=1)

        # if dyn_csv.empty:
        #     dyn_csv = amount_or_val
        # else:
        #     dyn_csv = pd.concat([dyn_csv, amount_or_val], axis=1)

    def process_dia_by_hid(self, feature, los, hid, dyn_csv):
        if self.feature_extractor.use_icu:
            feat = feature["new_icd_code"].unique()
            grp = feature[feature["stay_id"] == hid]
            if grp.shape[0] == 0:
                self.dataDic[hid]["Cond"] = {"fids": list(["<PAD>"])}
                feat_df = pd.DataFrame(np.zeros([1, len(feat)]), columns=feat)
                grp = feat_df.fillna(0)
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
            else:
                self.dataDic[hid]["Cond"] = {"fids": list(grp["new_icd_code"])}
                grp["val"] = 1
                grp = grp.drop_duplicates()
                grp = grp.pivot(
                    index="stay_id", columns="new_icd_code", values="val"
                ).reset_index(drop=True)
                feat_df = pd.DataFrame(columns=list(set(feat) - set(grp.columns)))
                grp = pd.concat([grp, feat_df], axis=1)
                grp = grp.fillna(0)
                grp = grp[feat]
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
        else:
            feat = feature["new_icd_code"].unique()
            grp = feature[feature["hadm_id"] == hid].copy()
            if grp.shape[0] == 0:
                self.dataDic[hid]["Cond"] = {"fids": list(["<PAD>"])}
                feat_df = pd.DataFrame(np.zeros([1, len(feat)]), columns=feat)
                grp = feat_df.fillna(0)
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
            else:
                self.dataDic[hid]["Cond"] = {"fids": list(grp["new_icd_code"])}
                grp["val"] = 1
                grp = grp.drop_duplicates()
                grp = grp.pivot(
                    index="hadm_id", columns="new_icd_code", values="val"
                ).reset_index(drop=True)
                feat_df = pd.DataFrame(columns=list(set(feat) - set(grp.columns)))
                grp = pd.concat([grp, feat_df], axis=1)
                grp = grp.fillna(0)
                grp = grp[feat]
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])

    def save_dictionaries(
        self, diag, meds, proc, out, labs, chart, cohort: pd.DataFrame, los
    ):
        if self.feature_extractor.use_icu:
            metaDic = {
                "Cond": {},
                "Proc": {},
                "Med": {},
                "Out": {},
                "Chart": {},
                "LOS": {},
            }
            metaDic["LOS"] = los
            with open(DICT_PATH / "dataDic", "wb") as fp:
                pickle.dump(self.dataDic, fp)

            with open(DICT_PATH / "hadmDic", "wb") as fp:
                pickle.dump(self.hids, fp)

            with open(DICT_PATH / "ethVocab", "wb") as fp:
                pickle.dump(list(cohort["ethnicity"].unique()), fp)
                self.eth_vocab = cohort["ethnicity"].nunique()

            with open(DICT_PATH / "ageVocab", "wb") as fp:
                pickle.dump(list(cohort["age"].unique()), fp)
                self.age_vocab = cohort["age"].nunique()

            with open(DICT_PATH / "insVocab", "wb") as fp:
                pickle.dump(list(cohort["insurance"].unique()), fp)
                self.ins_vocab = cohort["insurance"].nunique()

            if self.feature_extractor.for_medications:
                with open(DICT_PATH / "medVocab", "wb") as fp:
                    pickle.dump(list(meds["itemid"].unique()), fp)
                self.med_vocab = meds["itemid"].nunique()
                metaDic["Med"] = self.med_per_adm

            if self.feature_extractor.for_output_events:
                with open(DICT_PATH / "outVocab", "wb") as fp:
                    pickle.dump(list(out["itemid"].unique()), fp)
                self.out_vocab = out["itemid"].nunique()
                metaDic["Out"] = self.out_per_adm

            if self.feature_extractor.for_chart_events:
                with open(DICT_PATH / "chartVocab", "wb") as fp:
                    pickle.dump(list(chart["itemid"].unique()), fp)
                self.chart_vocab = chart["itemid"].nunique()
                metaDic["Chart"] = self.chart_per_adm

            if self.feature_extractor.for_diagnoses:
                with open(DICT_PATH / "condVocab", "wb") as fp:
                    pickle.dump(list(diag["new_icd_code"].unique()), fp)
                self.cond_vocab = diag["new_icd_code"].nunique()
                metaDic["Cond"] = self.dia_per_adm

            if self.feature_extractor.for_procedures:
                with open(DICT_PATH / "procVocab", "wb") as fp:
                    pickle.dump(list(proc["itemid"].unique()), fp)
                self.proc_vocab = proc["itemid"].nunique()
                metaDic["Proc"] = self.proc_per_adm

            with open(DICT_PATH / "metaDic", "wb") as fp:
                pickle.dump(metaDic, fp)
        else:
            metaDic = {"Cond": {}, "Proc": {}, "Med": {}, "Lab": {}, "LOS": {}}
            metaDic["LOS"] = los
            with open(DICT_PATH / "dataDic", "wb") as fp:
                pickle.dump(self.dataDic, fp)

            with open(DICT_PATH / "hadmDic", "wb") as fp:
                pickle.dump(self.hids, fp)

            with open(DICT_PATH / "ethVocab", "wb") as fp:
                pickle.dump(list(cohort["ethnicity"].unique()), fp)
                self.eth_vocab = cohort["ethnicity"].nunique()

            with open(DICT_PATH / "ageVocab", "wb") as fp:
                pickle.dump(list(cohort["age"].unique()), fp)
                self.age_vocab = cohort["age"].nunique()

            with open(DICT_PATH / "insVocab", "wb") as fp:
                pickle.dump(list(cohort["insurance"].unique()), fp)
                self.ins_vocab = cohort["insurance"].nunique()

            if self.feature_extractor.for_medications:
                with open(DICT_PATH / "medVocab", "wb") as fp:
                    pickle.dump(list(meds["drug_name"].unique()), fp)
                self.med_vocab = meds["drug_name"].nunique()
                metaDic["Med"] = self.med_per_adm

            if self.feature_extractor.for_diagnoses:
                with open(DICT_PATH / "condVocab", "wb") as fp:
                    pickle.dump(list(diag["new_icd_code"].unique()), fp)
                self.cond_vocab = diag["new_icd_code"].nunique()
                metaDic["Cond"] = self.dia_per_adm

            if self.feature_extractor.for_procedures:
                with open(DICT_PATH / "procVocab", "wb") as fp:
                    pickle.dump(list(proc["icd_code"].unique()), fp)
                self.proc_vocab = proc["icd_code"].unique()
                metaDic["Proc"] = self.proc_per_adm

            if self.feature_extractor.for_labs:
                with open(DICT_PATH / "labsVocab", "wb") as fp:
                    pickle.dump(list(labs["itemid"].unique()), fp)
                self.lab_vocab = labs["itemid"].unique()
                metaDic["Lab"] = self.labs_per_adm

            with open(DICT_PATH / "metaDic", "wb") as fp:
                pickle.dump(metaDic, fp)
