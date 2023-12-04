import tqdm
from my_preprocessing.admission_imputer import INPUTED_HOSPITAL_ADMISSION_ID_HEADER
from my_preprocessing.feature.feature import Feature
import logging
import pandas as pd
from my_preprocessing.preproc.feature import (
    PREPROC_LABS_PATH,
    LabEventsHeader,
)
from my_preprocessing.preproc.cohort import CohortHeader
from my_preprocessing.raw.hosp import (
    HospAdmissions,
    HospLabEvents,
    load_hosp_admissions,
    load_hosp_lab_events,
)

from my_preprocessing.file_info import save_data
from pathlib import Path

from my_preprocessing.uom_conversion import drop_wrong_uom

logger = logging.getLogger()


class Lab(Feature):
    def __init__(self, chunksize: int, cohort: pd.DataFrame):
        self.chunksize = chunksize
        self.cohort = cohort

    def summary_path(self):
        pass

    def feature_path(self) -> Path:
        return PREPROC_LABS_PATH

    def save(self) -> pd.DataFrame:
        logger.info("[EXTRACTING LABS DATA]")
        labevents = self.make()
        labevents = labevents[[h.value for h in LabEventsHeader]]
        return save_data(labevents, self.feature_path(), "LABS")

    def make(self) -> pd.DataFrame:
        """Process and transform lab events data."""
        admissions = load_hosp_admissions()[
            [
                HospAdmissions.PATIENT_ID,
                HospAdmissions.ID,
                HospAdmissions.ADMITTIME,
                HospAdmissions.DISCHTIME,
            ]
        ]
        usecols = [
            HospLabEvents.ITEM_ID,
            HospLabEvents.PATIENT_ID,
            HospLabEvents.HOSPITAL_ADMISSION_ID,
            HospLabEvents.CHART_TIME,
            HospLabEvents.VALUE_NUM,
            HospLabEvents.VALUE_UOM,
        ]

        processed_chunks = [
            self.process_lab_chunk(chunk, admissions)
            for chunk in tqdm(
                load_hosp_lab_events(chunksize=self.chunksize, use_cols=usecols)
            )
        ]

        return pd.concat(processed_chunks, ignore_index=True)

    def process_lab_chunk(
        self, chunk: pd.DataFrame, admissions: pd.DataFrame
    ) -> pd.DataFrame:
        """Process a single chunk of lab events."""
        chunk = chunk.dropna(subset=[HospLabEvents.VALUE_NUM]).fillna(
            {HospLabEvents.VALUE_UOM: 0}
        )
        chunk = chunk[
            chunk[LabEventsHeader.PATIENT_ID].isin(self.cohort[CohortHeader.PATIENT_ID])
        ]
        chunk_with_hadm, chunk_no_hadm = (
            chunk[chunk[HospLabEvents.HOSPITAL_ADMISSION_ID].notna()],
            chunk[chunk[HospLabEvents.HOSPITAL_ADMISSION_ID].isna()],
        )
        chunk_imputed = self.impute_hadm_ids(chunk_no_hadm.copy(), admissions)
        chunk_imputed[HospLabEvents.HOSPITAL_ADMISSION_ID] = chunk_imputed[
            INPUTED_HOSPITAL_ADMISSION_ID_HEADER
        ]
        chunk_imputed = chunk_imputed[
            [
                HospLabEvents.PATIENT_ID,
                HospLabEvents.HOSPITAL_ADMISSION_ID,
                HospLabEvents.ITEM_ID,
                HospLabEvents.CHART_TIME,
                HospLabEvents.VALUE_NUM,
                HospLabEvents.VALUE_UOM,
            ]
        ]
        merged_chunk = pd.concat([chunk_with_hadm, chunk_imputed], ignore_index=True)
        return self.merge_with_cohort_and_calculate_lab_time(merged_chunk)

    def merge_with_cohort_and_calculate_lab_time(
        self, chunk: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge chunk with cohort data and calculate the lab time from admit time."""
        chunk = chunk.merge(
            self.cohort[
                [
                    CohortHeader.HOSPITAL_ADMISSION_ID,
                    CohortHeader.ADMIT_TIME,
                    CohortHeader.DISCH_TIME,
                ]
            ],
            on=LabEventsHeader.HOSPITAL_ADMISSION_ID,
        )
        chunk[LabEventsHeader.CHART_TIME] = pd.to_datetime(
            chunk[LabEventsHeader.CHART_TIME]
        )
        chunk[LabEventsHeader.LAB_TIME_FROM_ADMIT] = (
            chunk[LabEventsHeader.CHART_TIME] - chunk[LabEventsHeader.ADMIT_TIME]
        )
        return chunk.dropna()

    def preproc(self):
        pass
