from enum import StrEnum
from typing import Dict
from pipeline.conversion.icd import IcdConverter
from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
from pipeline.file_info.preproc.feature import (
    DiagnosesHeader,
    DiagnosesIcuHeader,
    EXTRACT_DIAG_ICU_PATH,
    EXTRACT_DIAG_PATH,
    PREPROC_DIAG_ICU_PATH,
    PREPROC_DIAG_PATH,
)
from pipeline.file_info.preproc.cohort import CohortHeader, IcuCohortHeader
from pipeline.file_info.preproc.feature import PreprocDiagnosesHeader
from pipeline.file_info.preproc.summary import DIAG_FEATURES_PATH, DIAG_SUMMARY_PATH
from pipeline.file_info.raw.hosp import load_hosp_diagnosis_icd
from pipeline.file_info.common import save_data
from pathlib import Path

logger = logging.getLogger()


class IcdGroupOption(StrEnum):
    KEEP = "Keep both ICD-9 and ICD-10 codes"
    CONVERT = "Convert ICD-9 to ICD-10 codes"
    GROUP = "Convert ICD-9 to ICD-10 and group ICD-10 codes"


MEAN_FREQUENCY_HEADER = "mean_frequency"


class Diagnoses(Feature):
    def __init__(
        self,
        cohort: pd.DataFrame,
        use_icu: bool,
        icd_group_option: IcdGroupOption | None = None,
        df: pd.DataFrame = pd.DataFrame,
    ):
        self.cohort = cohort
        self.use_icu = use_icu
        self.icd_group_option = icd_group_option
        self.feature_path = EXTRACT_DIAG_ICU_PATH if self.use_icu else EXTRACT_DIAG_PATH
        self.preproc_feature_path = (
            PREPROC_DIAG_ICU_PATH if self.use_icu else PREPROC_DIAG_PATH
        )
        self.df = df

    def df(self):
        return self.df

    def extract_from(self, cohort: pd.DataFrame) -> pd.DataFrame:
        logger.info("[EXTRACTING DIAGNOSIS DATA]")
        hosp_diagnose = load_hosp_diagnosis_icd()
        admissions_cohort_cols = (
            [
                CohortHeader.HOSPITAL_ADMISSION_ID,
                IcuCohortHeader.STAY_ID,
                CohortHeader.LABEL,
            ]
            if self.use_icu
            else [CohortHeader.HOSPITAL_ADMISSION_ID, CohortHeader.LABEL]
        )
        diag = hosp_diagnose.merge(
            cohort[admissions_cohort_cols],
            on=DiagnosesHeader.HOSPITAL_ADMISSION_ID,
        )
        icd_converter = IcdConverter()
        diag = icd_converter.standardize_icd(diag)
        diag = diag[
            [h.value for h in DiagnosesHeader]
            + ([DiagnosesIcuHeader.STAY_ID] if self.use_icu else [])
        ]
        self.df = diag
        return diag

    def preproc(self, group_diag_icd: IcdGroupOption) -> pd.DataFrame:
        logger.info(f"[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(self.feature_path, compression="gzip")
        preproc_code = {
            IcdGroupOption.KEEP: DiagnosesHeader.ICD_CODE,
            IcdGroupOption.CONVERT: DiagnosesHeader.ROOT_ICD10,
            IcdGroupOption.GROUP: DiagnosesHeader.ROOT,
        }.get(group_diag_icd)
        diag[PreprocDiagnosesHeader.NEW_ICD_CODE] = diag[preproc_code]
        diag = diag[
            [c for c in PreprocDiagnosesHeader]
            + ([DiagnosesIcuHeader.STAY_ID] if self.use_icu else [])
        ]
        self.icd_group_option = group_diag_icd
        logger.info(f"Total number of rows: {diag.shape[0]}")
        return save_data(diag, self.preproc_feature_path, "DIAGNOSES")

    def summary(self):
        diag = pd.read_csv(
            self.preproc_feature_path,
            compression="gzip",
        )
        freq = (
            diag.groupby(
                [
                    DiagnosesIcuHeader.STAY_ID
                    if self.use_icu
                    else DiagnosesHeader.HOSPITAL_ADMISSION_ID,
                    PreprocDiagnosesHeader.NEW_ICD_CODE,
                ]
            )
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = (
            freq.groupby(PreprocDiagnosesHeader.NEW_ICD_CODE)[MEAN_FREQUENCY_HEADER]
            .mean()
            .reset_index()
        )
        total = (
            diag.groupby(PreprocDiagnosesHeader.NEW_ICD_CODE)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(
            freq, total, on=PreprocDiagnosesHeader.NEW_ICD_CODE, how="right"
        )
        summary = summary.fillna(0)

        summary.to_csv(DIAG_SUMMARY_PATH, index=False)
        summary[PreprocDiagnosesHeader.NEW_ICD_CODE].to_csv(
            DIAG_FEATURES_PATH, index=False
        )
        return summary

    def generate_fun(self):
        diag = pd.read_csv(self.summary_path(), compression="gzip")
        diag = diag[
            diag[DiagnosesHeader.HOSPITAL_ADMISSION_ID].isin(
                self.cohort[CohortHeader.HOSPITAL_ADMISSION_ID]
            )
        ]
        diag_per_adm = diag.groupby(DiagnosesHeader.HOSPITAL_ADMISSION_ID).size().max()
        self.df = diag
        self.df_per_adm = diag_per_adm
        return diag, diag_per_adm

    def mortality_length(self):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(self.cohort[col])]

    def los_length(self):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(self.cohort[col])]

    def read_length(self):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(self.cohort[col])]
