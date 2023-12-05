import pandas as pd
import logging
from tqdm import tqdm
from my_preprocessing.features_summary_generator import FeatureSummaryGenerator
from my_preprocessing.outlier_removal import outlier_imputation
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

logger = logging.getLogger()


class IcdGroupOption(StrEnum):
    KEEP = "Keep both ICD-9 and ICD-10 codes"
    CONVERT = "Convert ICD-9 to ICD-10 codes"
    GROUP = "Convert ICD-9 to ICD-10 and group ICD-10 codes"


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

    def preprocess_no_event_features(self):
        features = []
        if self.feature_extractor.for_diagnoses:
            features.append(self.preprocess_diag_features())
        if not self.feature_extractor.use_icu:
            if self.feature_extractor.for_medications:
                features.append(self.preprocess_med_features())
            if self.feature_extractor.for_procedures:
                features.append(self.preprocess_proc_features())

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

    def clean_events_features(self):
        features: List[pd.DataFrame] = []
        if self.clean_chart and self.feature_extractor.use_icu:
            features.append(self.clean_chart_features())
        if self.clean_labs and not self.feature_extractor.use_icu:
            features.append(self.clean_lab_features())
        return features

    def preprocess_diag_features(self) -> pd.DataFrame:
        logger.info("[PROCESSING DIAGNOSIS DATA]")
        path = (
            PREPROC_DIAG_ICU_PATH
            if self.feature_extractor.use_icu
            else PREPROC_DIAG_PATH
        )
        diag = pd.read_csv(path, compression="gzip")
        if self.group_diag_icd == IcdGroupOption.KEEP:
            diag[PreprocDiagnosesHeader.NEW_ICD_CODE] = diag[DiagnosesHeader.ICD_CODE]
        if self.group_diag_icd == IcdGroupOption.CONVERT:
            diag[PreprocDiagnosesHeader.NEW_ICD_CODE] = diag[DiagnosesHeader.ROOT_ICD10]
        if self.group_diag_icd == IcdGroupOption.GROUP:
            diag[PreprocDiagnosesHeader.NEW_ICD_CODE] = diag[DiagnosesHeader.ROOT]
        cols_to_keep = [c for c in PreprocDiagnosesHeader]
        if self.feature_extractor.use_icu:
            cols_to_keep = cols_to_keep + [h.value for h in DiagnosesIcuHeader]
        diag = diag[cols_to_keep]
        logger.info("Total number of rows", diag.shape[0])
        diag.to_csv(path, compression="gzip", index=False)
        logger.info("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        return diag

    def preprocess_med_features(self) -> pd.DataFrame:
        logger.info("[PROCESSING MEDICATIONS DATA]")
        path = (
            PREPROC_MED_ICU_PATH if self.feature_extractor.use_icu else PREPROC_MED_PATH
        )
        med = pd.read_csv(path, compression="gzip")
        med[PreprocMedicationHeader.DRUG_NAME] = (
            med[NonIcuMedicationHeader.NON_PROPRIEATARY_NAME]
            if self.group_med_code
            else med[NonIcuMedicationHeader.DRUG]
        )
        med = med.drop(
            columns=[
                NonIcuMedicationHeader.NON_PROPRIEATARY_NAME,
                NonIcuMedicationHeader.DRUG,
            ]
        )
        med.dropna()
        print("Total number of rows", med.shape[0])
        med.to_csv(path, compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
        return med

    def preprocess_proc_features(self) -> pd.DataFrame:
        logger.info("[PROCESSING PROCEDURES DATA]")
        proc = pd.read_csv(
            PREPROC_PROC_PATH,
            compression="gzip",
        )
        if not self.keep_proc_icd9:
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
        if not self.keep_proc_icd9:
            proc = proc.dropna()
        logger.info("Total number of rows", proc.shape[0])
        proc.to_csv(PREPROC_PROC_PATH, compression="gzip", index=False)
        logger.info("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        return proc

    def clean_chart_features(self) -> pd.DataFrame:
        logger.info("[PROCESSING CHART EVENTS DATA]")
        chart = pd.read_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
        chart = outlier_imputation(
            chart,
            "itemid",
            "valuenum",
            self.thresh,
            self.left_thresh,
            self.impute_outlier_chart,
        )

        logger.info("Total number of rows", chart.shape[0])
        chart.to_csv(
            PREPROC_CHART_ICU_PATH,
            compression="gzip",
            index=False,
        )
        logger.info("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

        return chart

    def clean_lab_features(self) -> pd.DataFrame:
        print("[PROCESSING LABS DATA]")
        labs = pd.read_csv(PREPROC_LABS_PATH, compression="gzip")
        labs = outlier_imputation(
            labs,
            "itemid",
            "valuenum",
            self.thresh,
            self.left_thresh,
            self.impute_labs,
        )

        print("Total number of rows", labs.shape[0])
        labs.to_csv(PREPROC_LABS_PATH, compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")

        return labs

    def preprocess(self):
        self.preprocess_no_event_features()
        summary_generator = FeatureSummaryGenerator(self.feature_extractor)
        summary_generator.save_summaries()
        self.feature_selection()
        self.clean_events_features()
