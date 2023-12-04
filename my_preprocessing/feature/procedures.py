from my_preprocessing.feature.feature import Feature
import logging
import pandas as pd
from my_preprocessing.preproc.feature import (
    ProceduresHeader,
    IcuProceduresHeader,
    NonIcuProceduresHeader,
    PREPROC_PROC_ICU_PATH,
    PREPROC_PROC_PATH,
)
from my_preprocessing.preproc.cohort import CohortHeader
from my_preprocessing.raw.hosp import HospProceduresIcd, load_hosp_procedures_icd
from my_preprocessing.raw.icu import load_icu_procedure_events
from my_preprocessing.file_info import save_data
from pathlib import Path

logger = logging.getLogger()


class Procedures(Feature):
    def __init__(self, use_icu: bool, cohort: pd.DataFrame):
        self.use_icu = use_icu
        self.cohort = cohort

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
                if self.us_icu
                else NonIcuProceduresHeader.ADMIT_TIME
            ]
        )
        procedures = procedures.dropna()
        self.log_icu() if self.use_icu else self.log_non_icu()
        return procedures

    def log_icu(self, procedures: pd.DataFrame) -> None:
        logger.info(
            "# Unique Events:  ",
            procedures[IcuProceduresHeader.ITEM_ID].dropna().nunique(),
        )
        logger.info(
            "# Admissions:  ", procedures[IcuProceduresHeader.STAY_ID].nunique()
        )
        logger.info("Total rows", procedures.shape[0])

    def log_non_icu(self, procedures: pd.DataFrame) -> None:
        for v in [9, 10]:
            print(
                f"# Unique ICD{v} Procedures:  ",
                procedures.loc[procedures[NonIcuProceduresHeader.ICD_VERSION] == v][
                    NonIcuProceduresHeader.ICD_CODE
                ]
                .dropna()
                .nunique(),
            )

        print(
            "\nValue counts of each ICD version:\n",
            procedures[NonIcuProceduresHeader.ICD_VERSION].value_counts(),
        )
        print(
            "# Admissions:  ",
            procedures[CohortHeader.HOSPITAL_ADMISSION_ID].nunique(),
        )
        print("Total number of rows: ", procedures.shape[0])

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
