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
from my_preprocessing.raw.hosp import load_hosp_diagnosis_icd
from my_preprocessing.icd_conversion import standardize_icd
from my_preprocessing.file_info import save_data
from pathlib import Path

logger = logging.getLogger()


class Diagnoses(Feature):
    def __init__(self, use_icu: bool, cohort: pd.DataFrame):
        self.use_icu = use_icu
        self.cohort = cohort

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
        return save_data(diag, self.summary_path(), "DIAGNOSES")

    def preproc(self):
        pass
