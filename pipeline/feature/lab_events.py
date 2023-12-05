from tqdm import tqdm
from pipeline.preprocessing.admission_imputer import (
    INPUTED_HOSPITAL_ADMISSION_ID_HEADER,
    impute_hadm_ids,
)
from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
from pipeline.preprocessing.outlier_removal import outlier_imputation
from pipeline.file_info.preproc.feature import (
    PREPROC_LABS_PATH,
    LabEventsHeader,
)
from pipeline.file_info.preproc.cohort import CohortHeader
from pipeline.file_info.preproc.summary import LABS_FEATURES_PATH, LABS_SUMMARY_PATH
from pipeline.file_info.raw.hosp import (
    HospAdmissions,
    HospLabEvents,
    load_hosp_admissions,
    load_hosp_lab_events,
)

from pipeline.file_info.common import save_data
from pathlib import Path

from pipeline.conversion.uom import drop_wrong_uom

logger = logging.getLogger()


class Lab(Feature):
    def __init__(
        self,
        cohort: pd.DataFrame,
        chunksize: int = 10000000,
        thresh=1,
        left_thresh=0,
        impute_outlier=False,
    ):
        self.cohort = cohort
        self.chunksize = chunksize
        self.thresh = thresh
        self.left_thresh = left_thresh
        self.impute = impute_outlier

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
        chunk_imputed = impute_hadm_ids(chunk_no_hadm.copy(), admissions)
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
        print("[PROCESSING LABS DATA]")
        labs = pd.read_csv(PREPROC_LABS_PATH, compression="gzip")
        labs = outlier_imputation(
            labs,
            "itemid",
            "valuenum",
            self.thresh,
            self.left_thresh,
            self.impute,
        )

        print("Total number of rows", labs.shape[0])
        labs.to_csv(PREPROC_LABS_PATH, compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")

        return labs

    def summary(self):
        labs = pd.DataFrame()
        for chunk in tqdm(
            pd.read_csv(
                PREPROC_LABS_PATH,
                compression="gzip",
                chunksize=self.chunksize,
            )
        ):
            if labs.empty:
                labs = chunk
            else:
                labs = labs.append(chunk, ignore_index=True)
        freq = (
            labs.groupby(
                [LabEventsHeader.HOSPITAL_ADMISSION_ID, LabEventsHeader.ITEM_ID]
            )
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = (
            freq.groupby([LabEventsHeader.ITEM_ID])["mean_frequency"]
            .mean()
            .reset_index()
        )

        missing = (
            labs[labs[LabEventsHeader.VALUE_NUM] == 0]
            .groupby(LabEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="missing_count")
        )
        total = (
            labs.groupby(LabEventsHeader.ITEM_ID).size().reset_index(name="total_count")
        )
        summary = pd.merge(missing, total, on=LabEventsHeader.ITEM_ID, how="right")
        summary = pd.merge(freq, summary, on=LabEventsHeader.ITEM_ID, how="right")
        summary["missing%"] = 100 * (summary["missing_count"] / summary["total_count"])
        summary = summary.fillna(0)
        summary.to_csv(LABS_SUMMARY_PATH, index=False)
        summary[LabEventsHeader.ITEM_ID].to_csv(LABS_FEATURES_PATH, index=False)
        return summary[LabEventsHeader.ITEM_ID]
