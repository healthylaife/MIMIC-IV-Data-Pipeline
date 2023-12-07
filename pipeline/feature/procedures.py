from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
from pipeline.file_info.preproc.feature import (
    ProceduresHeader,
    IcuProceduresHeader,
    NonIcuProceduresHeader,
    PREPROC_PROC_ICU_PATH,
    PREPROC_PROC_PATH,
)
from pipeline.file_info.preproc.cohort import CohortHeader
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

    def summary_path(self) -> Path:
        pass

    def feature_path(self) -> Path:
        return PREPROC_PROC_ICU_PATH if self.use_icu else PREPROC_PROC_PATH

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
                    CohortHeader.STAY_ID,
                    CohortHeader.IN_TIME,
                    CohortHeader.OUT_TIME,
                ]
                if self.use_icu
                else [
                    CohortHeader.HOSPITAL_ADMISSION_ID,
                    CohortHeader.ADMIT_TIME,
                    CohortHeader.DISCH_TIME,
                ]
            ],
            on=CohortHeader.STAY_ID
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

        # TODO: CHECK SUMMARY? as for diag?
        return save_data(proc, self.feature_path(), "PROCEDURES")

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
        return save_data(proc, self.feature_path(), "PROCEDURES")

    def summary(self):
        proc = pd.read_csv(
            self.feature_path(),
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
