import pandas as pd
import logging
from tqdm import tqdm
from my_preprocessing.feature.diagnoses import Diagnoses, IcdGroupOption
from my_preprocessing.feature.lab_events import Lab
from my_preprocessing.feature.medications import Medications
from my_preprocessing.feature.output_events import OutputEvents
from my_preprocessing.feature.procedures import Procedures
from my_preprocessing.preproc.feature import *
from my_preprocessing.features_extractor import FeatureExtractor
from typing import List
from pathlib import Path

from my_preprocessing.preproc.summary import (
    CHART_FEATURES_PATH,
    DIAG_FEATURES_PATH,
    LABS_FEATURES_PATH,
    MED_FEATURES_PATH,
    OUT_FEATURES_PATH,
    PROC_FEATURES_PATH,
)
from my_preprocessing.feature.chart_events import Chart

logger = logging.getLogger()


class FeaturePreprocessor:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        group_diag_icd: IcdGroupOption,
        group_med_code: bool,
        keep_proc_icd9: bool,
        clean_chart,
        impute_outlier_chart,
        clean_labs,
        impute_labs,
        thresh,
        left_thresh,
    ):
        self.feature_extractor = feature_extractor
        self.group_diag_icd = group_diag_icd
        self.group_med_code = group_med_code
        self.keep_proc_icd9 = keep_proc_icd9
        self.clean_chart = clean_chart
        self.impute_outlier_chart = impute_outlier_chart
        self.clean_labs = clean_labs
        self.impute_labs = impute_labs
        self.thresh = thresh
        self.left_thresh = left_thresh

    def feature_selection(self) -> List[pd.DataFrame]:
        features: List[pd.DataFrame] = []
        if self.group_diag_icd:
            features.append(
                self.process_feature_selection(
                    PREPROC_DIAG_ICU_PATH
                    if self.feature_extractor.use_icu
                    else PREPROC_DIAG_PATH,
                    DIAG_FEATURES_PATH,
                    PreprocDiagnosesHeader.NEW_ICD_CODE.value,
                    "Diagnosis",
                )
            )

        if self.group_med_code:
            path = (
                PREPROC_MED_ICU_PATH
                if self.feature_extractor.use_icu
                else PREPROC_MED_PATH
            )
            feature_name = (
                IcuMedicationHeader.ITEM_ID
                if self.feature_extractor.use_icu
                else PreprocMedicationHeader.DRUG_NAME
            )
            features.append(
                self.process_feature_selection(
                    path, MED_FEATURES_PATH, feature_name, "Medications"
                )
            )

        if self.feature_extractor.for_procedures:
            path = (
                PREPROC_PROC_ICU_PATH
                if self.feature_extractor.use_icu
                else PREPROC_PROC_PATH
            )
            features.append(
                self.process_feature_selection(
                    path,
                    PROC_FEATURES_PATH,
                    IcuProceduresHeader.ITEM_ID
                    if self.feature_extractor.use_icu
                    else NonIcuProceduresHeader.ICD_CODE.value,
                    "Procedures",
                )
            )

        if self.feature_extractor.for_labs and self.clean_labs:
            labs = self.concat_csv_chunks(PREPROC_LABS_PATH, 10000000)
            feature_df = pd.read_csv(LABS_FEATURES_PATH)
            labs = labs[labs["itemid"].isin(feature_df["itemid"].unique())]
            self.log_and_save(labs, PREPROC_LABS_PATH, "Labs")
            features.append(labs)

        if self.feature_extractor.for_chart_events:
            features.append(
                self.process_feature_selection(
                    PREPROC_CHART_ICU_PATH,
                    CHART_FEATURES_PATH,
                    "itemid",
                    "Output Events",
                )
            )

        if self.feature_extractor.for_output_events:
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

    def preproc_events_features(self):
        event_preproc_features: List[pd.DataFrame] = []
        if self.clean_chart and self.feature_extractor.use_icu:
            chart = Chart(
                cohort=pd.DataFrame(),
                thresh=self.thresh,
                left_thresh=self.left_thresh,
                impute_outlier=self.impute_outlier_chart,
            )
            event_preproc_features.append(chart.preproc())
        if self.clean_labs and not self.feature_extractor.use_icu:
            lab = Lab(
                cohort=pd.DataFrame(),
                thresh=self.thresh,
                left_thresh=self.left_thresh,
                impute_outlier=self.impute_labs,
            )
            event_preproc_features.append(lab.preproc())
        return event_preproc_features

    def preprocess(self):
        self.preprocess_no_event_features()
        self.save_summaries()
        self.feature_selection()
        self.preproc_events_features()

    def preprocess_no_event_features(self):
        no_event_preproc_features = []
        empty_cohort = pd.DataFrame()
        if self.feature_extractor.for_diagnoses:
            dia = Diagnoses(
                cohort=empty_cohort,
                use_icu=self.feature_extractor.use_icu,
                icd_group_option=self.group_diag_icd,
            )
            no_event_preproc_features.append(dia.preproc())
        if not self.feature_extractor.use_icu:
            if self.feature_extractor.for_medications:
                med = Medications(
                    cohort=empty_cohort,
                    use_icu=self.feature_extractor.use_icu,
                    group_code=self.group_med_code,
                )
                no_event_preproc_features.append(med.preproc())
            if self.feature_extractor.for_procedures:
                proc = Procedures(
                    cohort=empty_cohort,
                    use_icu=self.feature_extractor.use_icu,
                    keep_icd9=self.keep_proc_icd9,
                )
                no_event_preproc_features.append(proc.preproc())
        return no_event_preproc_features

    def save_summaries(self):
        summaries = []
        if self.feature_extractor.for_diagnoses:
            dia = Diagnoses(
                cohort=pd.DataFrame(),
                use_icu=self.feature_extractor.use_icu,
                icd_group_option=self.group_diag_icd,
            )
            summaries.append(dia.summary())
        if self.feature_extractor.for_medications:
            med = Medications(
                cohort=pd.DataFrame(),
                use_icu=self.feature_extractor.use_icu,
                group_code=self.group_med_code,
            )
            summaries.append(med.summary())
        if self.feature_extractor.for_procedures:
            proc = Procedures(
                cohort=pd.DataFrame(),
                use_icu=self.feature_extractor.use_icu,
                keep_icd9=self.keep_proc_icd9,
            )
            summaries.append(proc.summary())
        if self.feature_extractor.for_output_events:
            out = OutputEvents(cohort=pd.DataFrame())
            summaries.append(out.summary())
        if self.feature_extractor.for_chart_events:
            chart = Chart(cohort=pd.DataFrame())
            summaries.append(chart.summary())
        if self.feature_extractor.for_labs:
            lab = Lab(cohort=pd.DataFrame())
            summaries.append(lab.summary)
        return summaries
