from enum import StrEnum
from my_preprocessing.feature.feature import Feature
import logging
import pandas as pd
from my_preprocessing.preproc.feature import (
    DiagnosesHeader,
    DiagnosesIcuHeader,
    PREPROC_DIAG_ICU_PATH,
    PREPROC_DIAG_PATH,
)
from my_preprocessing.preproc.cohort import CohortHeader
from my_preprocessing.preproc.feature import PreprocDiagnosesHeader
from my_preprocessing.preproc.summary import DIAG_FEATURES_PATH, DIAG_SUMMARY_PATH
from my_preprocessing.raw.hosp import load_hosp_diagnosis_icd
from my_preprocessing.icd_conversion import standardize_icd
from my_preprocessing.file_info import save_data
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
    ):
        self.cohort = cohort
        self.use_icu = use_icu
        self.icd_group_option = icd_group_option

    def summary_path(self) -> Path:
        pass

    def feature_path(self):
        return PREPROC_DIAG_ICU_PATH if self.use_icu else PREPROC_DIAG_PATH

    def make(self) -> pd.DataFrame:
        hosp_diagnose = load_hosp_diagnosis_icd()
        admissions_cohort_cols = (
            [
                CohortHeader.HOSPITAL_ADMISSION_ID,
                CohortHeader.STAY_ID,
                CohortHeader.LABEL,
            ]
            if self.use_icu
            else [CohortHeader.HOSPITAL_ADMISSION_ID, CohortHeader.LABEL]
        )
        diag = hosp_diagnose.merge(
            self.cohort[admissions_cohort_cols],
            on=DiagnosesHeader.HOSPITAL_ADMISSION_ID,
        )
        diag = standardize_icd(diag)
        return diag

    def save(self) -> pd.DataFrame:
        logger.info("[EXTRACTING DIAGNOSIS DATA]")
        diag = self.make()
        cols = [h.value for h in DiagnosesHeader]
        if self.use_icu:
            cols = cols + [h.value for h in DiagnosesIcuHeader]

        diag = diag[cols]
        return save_data(diag, self.feature_path(), "DIAGNOSES")

    def preproc(self) -> pd.DataFrame:
        logger.info(f"[PROCESSING DIAGNOSIS DATA]")
        path = self.feature_path()
        diag = pd.read_csv(path, compression="gzip")
        if self.icd_group_option == IcdGroupOption.KEEP:
            diag[PreprocDiagnosesHeader.NEW_ICD_CODE] = diag[DiagnosesHeader.ICD_CODE]
        if self.icd_group_option == IcdGroupOption.CONVERT:
            diag[PreprocDiagnosesHeader.NEW_ICD_CODE] = diag[DiagnosesHeader.ROOT_ICD10]
        if self.icd_group_option == IcdGroupOption.GROUP:
            diag[PreprocDiagnosesHeader.NEW_ICD_CODE] = diag[DiagnosesHeader.ROOT]
        cols_to_keep = [c for c in PreprocDiagnosesHeader]
        if self.use_icu:
            cols_to_keep = cols_to_keep + [h.value for h in DiagnosesIcuHeader]
        diag = diag[cols_to_keep]
        logger.info(f"Total number of rows: {diag.shape[0]}")
        return save_data(diag, self.feature_path(), "DIAGNOSES")

    def summary(self):
        diag = pd.read_csv(
            self.feature_path(),
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
