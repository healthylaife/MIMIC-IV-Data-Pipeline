from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
import numpy as np
from pipeline.file_info.preproc.feature import (
    ProceduresHeader,
    IcuProceduresHeader,
    NonIcuProceduresHeader,
    PREPROC_PROC_ICU_PATH,
    PREPROC_PROC_PATH,
)
from pipeline.file_info.preproc.cohort import (
    CohortHeader,
    IcuCohortHeader,
    NonIcuCohortHeader,
)
from pipeline.file_info.preproc.summary import PROC_FEATURES_PATH, PROC_SUMMARY_PATH
from pipeline.file_info.raw.hosp import HospProceduresIcd, load_hosp_procedures_icd
from pipeline.file_info.raw.icu import load_icu_procedure_events
from pipeline.file_info.common import save_data
from pathlib import Path

logger = logging.getLogger()


class Procedures(Feature):
    def __init__(self, cohort: pd.DataFrame, use_icu: bool, keep_icd9: bool = True):
        self.cohort = cohort
        self.use_icu = use_icu
        self.keep_icd9 = keep_icd9
        self.df = pd.DataFrame()
        self.final_df = pd.DataFrame()
        self.feature_path = PREPROC_PROC_ICU_PATH if self.use_icu else PREPROC_PROC_PATH

    def make(self) -> pd.DataFrame:
        logger.info("[EXTRACTING PROCEDURES DATA]")
        raw_procedures = (
            load_icu_procedure_events() if self.use_icu else load_hosp_procedures_icd()
        )
        procedures = raw_procedures.merge(
            self.cohort[
                [
                    CohortHeader.PATIENT_ID,
                    CohortHeader.HOSPITAL_ADMISSION_ID,
                    IcuCohortHeader.STAY_ID,
                    IcuCohortHeader.IN_TIME,
                    IcuCohortHeader.OUT_TIME,
                ]
                if self.use_icu
                else [
                    CohortHeader.HOSPITAL_ADMISSION_ID,
                    NonIcuCohortHeader.ADMIT_TIME,
                    NonIcuCohortHeader.DISCH_TIME,
                ]
            ],
            on=IcuCohortHeader.STAY_ID
            if self.use_icu
            else HospProceduresIcd.HOSPITAL_ADMISSION_ID,
        )
        procedures[
            IcuProceduresHeader.EVENT_TIME_FROM_ADMIT
            if self.use_icu
            else NonIcuProceduresHeader.PROC_TIME_FROM_ADMIT
        ] = (
            procedures[
                IcuProceduresHeader.START_TIME
                if self.use_icu
                else NonIcuProceduresHeader.CHART_DATE
            ]
            - procedures[
                IcuProceduresHeader.IN_TIME
                if self.use_icu
                else NonIcuProceduresHeader.ADMIT_TIME
            ]
        )
        procedures = procedures.dropna()
        self.log_icu(procedures) if self.use_icu else self.log_non_icu(procedures)
        return procedures

    def log_icu(self, procedures: pd.DataFrame) -> None:
        logger.info(
            f"# Unique Events: {procedures[IcuProceduresHeader.ITEM_ID].dropna().nunique()}"
        )
        logger.info(
            f"# Admissions:   {procedures[IcuProceduresHeader.STAY_ID].nunique()}"
        )
        logger.info(f"Total rows: {procedures.shape[0]}")

    def log_non_icu(self, procedures: pd.DataFrame) -> None:
        for v in [9, 10]:
            unique_procedures_count = (
                procedures.loc[procedures[NonIcuProceduresHeader.ICD_VERSION] == v][
                    NonIcuProceduresHeader.ICD_CODE
                ]
                .dropna()
                .nunique()
            )
            logger.info(f" # Unique ICD{v} Procedures:{ unique_procedures_count}")

        logger.info(
            f"\nValue counts of each ICD version:\n {procedures[NonIcuProceduresHeader.ICD_VERSION].value_counts()}"
        )
        logger.info(
            f"# Admissions:{procedures[CohortHeader.HOSPITAL_ADMISSION_ID].nunique()}"
        )
        logger.info(f"Total number of rows: {procedures.shape[0]}")

    def save(self) -> pd.DataFrame:
        proc = self.make()
        proc = proc[
            [h.value for h in ProceduresHeader]
            + [
                h.value
                for h in (
                    IcuProceduresHeader if self.use_icu else NonIcuProceduresHeader
                )
            ]
        ]

        return save_data(proc, self.feature_path, "PROCEDURES")

    def preproc(self):
        logger.info("[PROCESSING PROCEDURES DATA]")
        proc = pd.read_csv(
            PREPROC_PROC_PATH,
            compression="gzip",
        )
        if not self.keep_icd9:
            proc = proc.loc[proc[NonIcuProceduresHeader.ICD_VERSION] == 10]
        proc = proc[
            [
                ProceduresHeader.PATIENT_ID,
                ProceduresHeader.HOSPITAL_ADMISSION_ID,
                NonIcuProceduresHeader.ICD_CODE,
                NonIcuProceduresHeader.CHART_DATE,
                NonIcuProceduresHeader.ADMIT_TIME,
                NonIcuProceduresHeader.PROC_TIME_FROM_ADMIT,
            ]
        ]
        if not self.keep_icd9:
            proc = proc.dropna()
        logger.info(f"Total number of rows: {proc.shape[0]}")
        return save_data(proc, self.feature_path, "PROCEDURES")

    def summary(self):
        proc = pd.read_csv(
            self.feature_path,
            compression="gzip",
        )
        feature_name = (
            IcuProceduresHeader.ITEM_ID
            if self.use_icu
            else NonIcuProceduresHeader.ICD_CODE
        )
        freq = (
            proc.groupby(
                [
                    "stay_id" if self.use_icu else "hadm_id",
                    feature_name,
                ]
            )
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(feature_name)["mean_frequency"].mean().reset_index()
        total = proc.groupby(feature_name).size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on=feature_name, how="right")
        summary = summary.fillna(0)
        summary.to_csv(PROC_SUMMARY_PATH, index=False)
        summary[feature_name].to_csv(PROC_FEATURES_PATH, index=False)
        return summary[feature_name]

    def generate_fun(self):
        proc = pd.read_csv(self.feature_path(), compression="gzip")
        proc = proc[
            proc[ProceduresHeader.HOSPITAL_ADMISSION_ID].isin(self.cohort["hadm_id"])
        ]
        proc[["start_days", "dummy", "start_hours"]] = proc[
            "proc_time_from_admit"
        ].str.split(" ", expand=True)
        proc[["start_hours", "min", "sec"]] = proc["start_hours"].str.split(
            ":", expand=True
        )
        proc["start_time"] = pd.to_numeric(proc["start_days"]) * 24 + pd.to_numeric(
            proc["start_hours"]
        )
        proc = proc.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
        proc = proc[proc["start_time"] >= 0]

        ###Remove where event time is after discharge time
        proc = pd.merge(proc, self.cohort[["hadm_id", "los"]], on="hadm_id", how="left")
        proc["sanity"] = proc["los"] - proc["start_time"]
        proc = proc[proc["sanity"] > 0]
        del proc["sanity"]
        self.df = proc
        return proc

    def mortality_length(self, include_time):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(self.df[col])]
        self.df = self.df[self.df[col] <= include_time]

    def los_length(self, include_time):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(self.cohort[col])]
        self.df = self.df[self.df[col] <= include_time]

    def read_length(self):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(self.cohort[col])]
        self.df = pd.merge(
            self.df, self.cohort[[col, "select_time"]], on=col, how="left"
        )
        self.df["start_time"] = self.proc["start_time"] - self.proc["select_time"]
        self.df = self.df[self.df["start_time"] >= 0]

    def smooth_meds_step(self, bucket, i, t):
        sub_proc = (
            self.proc[
                (self.proc["start_time"] >= i) & (self.proc["start_time"] < i + bucket)
            ]
            .groupby(["stay_id", "itemid"] if self.use_icu else ["hadm_id", "icd_code"])
            .agg({"subject_id": "max"})
        )
        sub_proc = sub_proc.reset_index()
        sub_proc["start_time"] = t
        if self.final_df.empty:
            self.final_df = sub_proc
        else:
            self.final_df = self.final_df.append(sub_proc)

    def smooth_meds(self):
        f2_df = self.final_df.groupby(
            ["stay_id", "itemid", "orderid"]
            if self.use_icu
            else ["hadm_id", "icd_code"]
        ).size()
        df_per_adm = (
            f2_df.groupby("stay_id" if self.use_icd else "hadm_id")
            .sum()
            .reset_index()[0]
            .max()
        )
        dflength_per_adm = (
            self.final_df.groupby("stay_id" if self.use_icd else "hadm_id").size().max()
        )
        return f2_df, df_per_adm, dflength_per_adm

    def dict_step(self, hid, los, dataDic):
        if self.use_icu:
            feat = self.final_df["itemid"].unique()
            df2 = self.final_df[self.final_df["stay_id"] == hid]
            if df2.shape[0] == 0:
                df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
            else:
                df2["val"] = 1
                # print(df2)
                df2 = df2.pivot_table(
                    index="start_time", columns="itemid", values="val"
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
                dataDic[hid]["Proc"] = df2.to_dict(orient="list")

                feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)

                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
        else:
            feat = self.final_df["icd_code"].unique()
            df2 = self.final_df[self.final_df["hadm_id"] == hid]
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
                dataDic[hid]["Proc"] = df2.to_dict(orient="list")

                feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)

                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])

        return df2
