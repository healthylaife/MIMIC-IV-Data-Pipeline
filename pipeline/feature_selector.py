import pandas as pd
import logging
from pipeline.file_info.preproc.feature import (
    PREPROC_DIAG_ICU_PATH,
    PREPROC_DIAG_PATH,
    PreprocDiagnosesHeader,
    PREPROC_MED_ICU_PATH,
    PREPROC_MED_PATH,
    IcuMedicationHeader,
    PreprocMedicationHeader,
    PREPROC_OUT_ICU_PATH,
    PREPROC_LABS_PATH,
    PREPROC_PROC_ICU_PATH,
    PREPROC_PROC_PATH,
    PREPROC_CHART_ICU_PATH,
    IcuProceduresHeader,
    NonIcuProceduresHeader,
)
from pipeline.features_extractor import FeatureExtractor
from typing import List
from pathlib import Path

from pipeline.file_info.preproc.summary import (
    CHART_FEATURES_PATH,
    DIAG_FEATURES_PATH,
    LABS_FEATURES_PATH,
    MED_FEATURES_PATH,
    OUT_FEATURES_PATH,
    PROC_FEATURES_PATH,
)
from pipeline.feature.chart_events import Chart

logger = logging.getLogger()


class FeatureSelector:
    def __init__(
        self,
        use_icu: bool,
        select_dia: bool,
        select_med: bool,
        select_proc: bool,
        select_labs: bool,
        select_chart: bool,
        select_out: bool,
    ):
        self.use_icu = use_icu

        self.select_dia = select_dia
        self.select_med = select_med
        self.select_proc = select_proc
        self.select_dia = select_dia
        self.select_labs = select_labs
        self.select_chart = select_chart
        self.select_out = select_out

    def feature_selection(self) -> List[pd.DataFrame]:
        features: List[pd.DataFrame] = []
        if self.select_dia:
            features.append(
                self.process_feature_selection(
                    PREPROC_DIAG_ICU_PATH if self.use_icu else PREPROC_DIAG_PATH,
                    DIAG_FEATURES_PATH,
                    PreprocDiagnosesHeader.NEW_ICD_CODE.value,
                    "Diagnosis",
                )
            )

        if self.select_med:
            path = PREPROC_MED_ICU_PATH if self.use_icu else PREPROC_MED_PATH
            feature_name = (
                IcuMedicationHeader.ITEM_ID
                if self.use_icu
                else PreprocMedicationHeader.DRUG_NAME
            )
            features.append(
                self.process_feature_selection(
                    path, MED_FEATURES_PATH, feature_name, "Medications"
                )
            )

        if self.select_proc:
            path = PREPROC_PROC_ICU_PATH if self.use_icu else PREPROC_PROC_PATH
            features.append(
                self.process_feature_selection(
                    path,
                    PROC_FEATURES_PATH,
                    IcuProceduresHeader.ITEM_ID
                    if self.use_icu
                    else NonIcuProceduresHeader.ICD_CODE.value,
                    "Procedures",
                )
            )

        if self.select_labs:
            labs = self.concat_csv_chunks(PREPROC_LABS_PATH, 10000000)
            feature_df = pd.read_csv(LABS_FEATURES_PATH)
            labs = labs[labs["itemid"].isin(feature_df["itemid"].unique())]
            self.log_and_save(labs, PREPROC_LABS_PATH, "Labs")
            features.append(labs)

        if self.select_chart:
            features.append(
                self.process_feature_selection(
                    PREPROC_CHART_ICU_PATH,
                    CHART_FEATURES_PATH,
                    "itemid",
                    "Output Events",
                )
            )

        if self.select_out:
            features.append(
                self.process_feature_selection(
                    PREPROC_OUT_ICU_PATH, OUT_FEATURES_PATH, "itemid", "Output Events"
                )
            )

        return features

    def process_feature_selection(
        self, data_path: Path, feature_path: Path, feature_col: str, data_type: str
    ):
        """Generalized method for processing feature selection."""
        data_df = pd.read_csv(data_path, compression="gzip")
        feature_df = pd.read_csv(feature_path)
        data_df = data_df[data_df[feature_col].isin(feature_df[feature_col].unique())]
        self.log_and_save(data_df, data_path, data_type)
        return data_df

    def concat_csv_chunks(self, file_path: Path, chunksize: int):
        """Concatenate chunks from a CSV file."""
        chunks = pd.read_csv(file_path, compression="gzip", chunksize=chunksize)
        return pd.concat(chunks, ignore_index=True)

    def log_and_save(self, df: pd.DataFrame, path: Path, data_type: str):
        """Log information and save DataFrame to a CSV file."""
        logger.info(f"Total number of rows in {data_type}: {df.shape[0]}")
        df.to_csv(path, compression="gzip", index=False)
        logger.info(f"[SUCCESSFULLY SAVED {data_type} DATA]")
