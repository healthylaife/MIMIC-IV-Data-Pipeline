from enum import StrEnum
from pipeline.conversion.icd import IcdConverter
from pipeline.feature.feature_abc import Feature, Name
import logging
import pandas as pd
from pipeline.file_info.preproc.feature import (
    DiagnosesHeader,
    DiagnosesIcuHeader,
)
from pipeline.file_info.preproc.cohort import CohortHeader, IcuCohortHeader
from pipeline.file_info.preproc.feature import PreprocDiagnosesHeader
from pipeline.file_info.raw.hosp import load_hosp_diagnosis_icd

logger = logging.getLogger()


class IcdGroupOption(StrEnum):
    KEEP = "Keep both ICD-9 and ICD-10 codes"
    CONVERT = "Convert ICD-9 to ICD-10 codes"
    GROUP = "Convert ICD-9 to ICD-10 and group ICD-10 codes"


MEAN_FREQUENCY_HEADER = "mean_frequency"


class Diagnoses(Feature):
    def __init__(self, use_icu: bool, df: pd.DataFrame = pd.DataFrame()):
        self.use_icu = use_icu
        self.df = df

    def name() -> str:
        return Name.DIAGNOSES

    def df(self) -> pd.DataFrame:
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
        preproc_code = {
            IcdGroupOption.KEEP: DiagnosesHeader.ICD_CODE,
            IcdGroupOption.CONVERT: DiagnosesHeader.ROOT_ICD10,
            IcdGroupOption.GROUP: DiagnosesHeader.ROOT,
        }.get(group_diag_icd)
        self.df[PreprocDiagnosesHeader.NEW_ICD_CODE] = self.df[preproc_code]
        self.df = self.df[
            [c for c in PreprocDiagnosesHeader]
            + ([DiagnosesIcuHeader.STAY_ID] if self.use_icu else [])
        ]
        self.icd_group_option = group_diag_icd
        logger.info(f"Total number of rows: {self.df.shape[0]}")
        return self.df

    def summary(self):
        diag: pd.DataFrame = self.df
        group_column = (
            DiagnosesIcuHeader.STAY_ID
            if self.use_icu
            else DiagnosesHeader.HOSPITAL_ADMISSION_ID
        )
        freq = diag.groupby([group_column, PreprocDiagnosesHeader.NEW_ICD_CODE]).size()
        freq = freq.reset_index(name="mean_frequency")
        mean_freq = freq.groupby(PreprocDiagnosesHeader.NEW_ICD_CODE)[
            "mean_frequency"
        ].mean()
        total = (
            diag.groupby(PreprocDiagnosesHeader.NEW_ICD_CODE)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(
            mean_freq, total, on=PreprocDiagnosesHeader.NEW_ICD_CODE, how="right"
        )
        summary = summary.fillna(0)
        return summary

    def generate_fun(self, cohort: pd.DataFrame):
        diag: pd.DataFrame = self.df
        diag = diag[
            diag[DiagnosesHeader.HOSPITAL_ADMISSION_ID].isin(
                cohort[CohortHeader.HOSPITAL_ADMISSION_ID]
            )
        ]
        diag_per_adm = diag.groupby(DiagnosesHeader.HOSPITAL_ADMISSION_ID).size().max()
        self.df = diag
        return diag, diag_per_adm

    def mortality_length(self, cohort):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(cohort[col])]

    def los_length(self, cohort):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(cohort[col])]

    def read_length(self, cohort):
        col = "stay_id" if self.use_icu else "hadm_id"
        self.df = self.df[self.df[col].isin(cohort[col])]
