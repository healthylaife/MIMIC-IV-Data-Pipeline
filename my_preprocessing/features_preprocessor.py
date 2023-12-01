import pandas as pd
import logging
from tqdm import tqdm
from my_preprocessing.outlier_removal import outlier_imputation
from my_preprocessing.preproc_file_info import *
from my_preprocessing.features_extractor import FeatureExtractor

logger = logging.getLogger()


class FeaturePreprocessor:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        group_diag,
        group_med,
        group_proc,
        group_out,
        group_chart,
        clean_chart,
        impute_outlier_chart,
        clean_labs,
        impute_labs,
        thresh,
        left_thresh,
    ):
        self.feature_extractor = feature_extractor
        self.group_diag = group_diag
        self.group_med = group_med
        self.group_proc = group_proc
        self.group_out = group_out
        self.group_chart = group_chart
        self.clean_chart = clean_chart
        self.impute_outlier_chart = impute_outlier_chart
        self.clean_labs = (clean_labs,)
        self.impute_labs = impute_labs
        self.thresh = thresh
        self.left_thresh = left_thresh

    def features_selection_icu(self):
        if self.feature_extractor.for_diagnoses:
            if self.group_diag:
                logger.info("[FEATURE SELECTION DIAGNOSIS DATA]")
                diag = pd.read_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
                features = pd.read_csv(DIAG_FEATURES_PATH)
                diag = diag[
                    diag["new_icd_code"].isin(features["new_icd_code"].unique())
                ]

                logger.info("Total number of rows", diag.shape[0])
                diag.to_csv(
                    PREPROC_DIAG_ICU_PATH,
                    compression="gzip",
                    index=False,
                )
                logger.info("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        if self.feature_extractor.for_medications:
            if self.group_med:
                logger.info("[FEATURE SELECTION MEDICATIONS DATA]")
                med = pd.read_csv(PREPROC_MED_ICU_PATH, compression="gzip")
                features = pd.read_csv(MED_FEATURES_PATH)
                med = med[med["itemid"].isin(features["itemid"].unique())]
                logger.info("Total number of rows", med.shape[0])
                med.to_csv(
                    PREPROC_MED_ICU_PATH,
                    compression="gzip",
                    index=False,
                )
                logger.info("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

        if self.feature_extractor.for_procedures:
            if self.group_proc:
                logger.info("[FEATURE SELECTION PROCEDURES DATA]")
                proc = pd.read_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
                features = pd.read_csv(PROC_FEATURES_PATH)
                proc = proc[proc["itemid"].isin(features["itemid"].unique())]
                logger.info("Total number of rows", proc.shape[0])
                proc.to_csv(
                    PREPROC_PROC_ICU_PATH,
                    compression="gzip",
                    index=False,
                )
                logger.info("[SUCCESSFULLY SAVED PROCEDURES DATA]")

        if self.feature_extractor.for_output_events:
            if self.group_out:
                logger.info("[FEATURE SELECTION OUTPUT EVENTS DATA]")
                out = pd.read_csv(PREPROC_OUT_ICU_PATH, compression="gzip")
                features = pd.read_csv(OUT_FEATURES_PATH)
                out = out[out["itemid"].isin(features["itemid"].unique())]
                logger.info("Total number of rows", out.shape[0])
                out.to_csv(
                    PREPROC_OUT_ICU_PATH,
                    compression="gzip",
                    index=False,
                )
                logger.info("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")

        if self.feature_extractor.for_chart_events:
            if self.group_chart:
                logger.info("[FEATURE SELECTION CHART EVENTS DATA]")

                chart = pd.read_csv(
                    PREPROC_CHART_ICU_PATH,
                    compression="gzip",
                    index_col=None,
                )

                features = pd.read_csv(CHART_FEATURES_PATH, header=0)
                chart = chart[chart["itemid"].isin(features["itemid"].unique())]
                logger.info("Total number of rows", chart.shape[0])
                chart.to_csv(
                    PREPROC_CHART_ICU_PATH,
                    compression="gzip",
                    index=False,
                )
                logger.info("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    def features_selection_hosp(self):
        if self.feature_extractor.for_diagnoses:
            if self.group_diag:
                print("[FEATURE SELECTION DIAGNOSIS DATA]")
                diag = pd.read_csv(PREPROC_DIAG_PATH, compression="gzip")
                features = pd.read_csv(DIAG_FEATURES_PATH)
                diag = diag[
                    diag["new_icd_code"].isin(features["new_icd_code"].unique())
                ]

                print("Total number of rows", diag.shape[0])
                diag.to_csv(PREPROC_DIAG_PATH, compression="gzip", index=False)
                print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

        if self.feature_extractor.for_medications:
            if self.group_med:
                print("[FEATURE SELECTION MEDICATIONS DATA]")
                med = pd.read_csv(PREPROC_MED_PATH, compression="gzip")
                features = pd.read_csv(MED_FEATURES_PATH)
                med = med[med["drug_name"].isin(features["drug_name"].unique())]
                print("Total number of rows", med.shape[0])
                med.to_csv(PREPROC_MED_PATH, compression="gzip", index=False)
                print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

        if self.feature_extractor.for_procedures:
            if self.group_proc:
                print("[FEATURE SELECTION PROCEDURES DATA]")
                proc = pd.read_csv(PREPROC_PROC_PATH, compression="gzip")
                features = pd.read_csv(PROC_FEATURES_PATH)
                proc = proc[proc["icd_code"].isin(features["icd_code"].unique())]
                print("Total number of rows", proc.shape[0])
                proc.to_csv(PREPROC_PROC_PATH, compression="gzip", index=False)
                print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

        if self.feature_extractor.for_labs:
            if self.clean_labs:
                print("[FEATURE SELECTION LABS DATA]")
                chunksize = 10000000
                labs = pd.DataFrame()
                for chunk in tqdm(
                    pd.read_csv(
                        PREPROC_LABS_PATH,
                        compression="gzip",
                        index_col=None,
                        chunksize=chunksize,
                    )
                ):
                    if labs.empty:
                        labs = chunk
                    else:
                        labs = labs.append(chunk, ignore_index=True)
                features = pd.read_csv(LABS_FEATURES_PATH)
                labs = labs[labs["itemid"].isin(features["itemid"].unique())]
                print("Total number of rows", labs.shape[0])
                labs.to_csv(PREPROC_LABS_PATH, compression="gzip", index=False)
                print("[SUCCESSFULLY SAVED LABS DATA]")

    def preprocess_features_icu(self):
        if self.feature_extractor.for_diagnoses:
            print("[PROCESSING DIAGNOSIS DATA]")
            diag = pd.read_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
            if self.group_diag == "Keep both ICD-9 and ICD-10 codes":
                diag["new_icd_code"] = diag["icd_code"]
            if self.group_diag == "Convert ICD-9 to ICD-10 codes":
                diag["new_icd_code"] = diag["root_icd10_convert"]
            if self.group_diag == "Convert ICD-9 to ICD-10 and group ICD-10 codes":
                diag["new_icd_code"] = diag["root"]

            diag = diag[["subject_id", "hadm_id", "stay_id", "new_icd_code"]].dropna()
            print("Total number of rows", diag.shape[0])
            diag.to_csv(PREPROC_DIAG_ICU_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

        if self.feature_extractor.for_chart_events:
            if self.clean_chart:
                print("[PROCESSING CHART EVENTS DATA]")
                chart = pd.read_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
                chart = outlier_imputation(
                    chart,
                    "itemid",
                    "valuenum",
                    self.thresh,
                    self.left_thresh,
                    self.impute_outlier_chart,
                )

                print("Total number of rows", chart.shape[0])
                chart.to_csv(
                    PREPROC_CHART_ICU_PATH,
                    compression="gzip",
                    index=False,
                )
                print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    def preprocess_features_hosp(self):
        if self.feature_extractor.for_diagnoses:
            print("[PROCESSING DIAGNOSIS DATA]")
            diag = pd.read_csv(PREPROC_DIAG_PATH, compression="gzip")
            if self.group_diag == "Keep both ICD-9 and ICD-10 codes":
                diag["new_icd_code"] = diag["icd_code"]
            if self.group_diag == "Convert ICD-9 to ICD-10 codes":
                diag["new_icd_code"] = diag["root_icd10_convert"]
            if self.group_diag == "Convert ICD-9 to ICD-10 and group ICD-10 codes":
                diag["new_icd_code"] = diag["root"]

            diag = diag[["subject_id", "hadm_id", "new_icd_code"]].dropna()
            print("Total number of rows", diag.shape[0])
            diag.to_csv(PREPROC_DIAG_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

        if self.feature_extractor.for_medications:
            print("[PROCESSING MEDICATIONS DATA]")
            if self.group_med:
                med = pd.read_csv(PREPROC_MED_PATH, compression="gzip")
                if self.group_med:
                    med["drug_name"] = med["nonproprietaryname"]
                else:
                    med["drug_name"] = med["drug"]
                med = med[
                    [
                        "subject_id",
                        "hadm_id",
                        "starttime",
                        "stoptime",
                        "drug_name",
                        "start_hours_from_admit",
                        "stop_hours_from_admit",
                        "dose_val_rx",
                    ]
                ].dropna()
                print("Total number of rows", med.shape[0])
                med.to_csv(PREPROC_MED_PATH, compression="gzip", index=False)
                print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

        if self.feature_extractor.for_procedures:
            print("[PROCESSING PROCEDURES DATA]")
            proc = pd.read_csv(
                PREPROC_PROC_PATH,
                compression="gzip",
            )
            if self.group_proc == "ICD-9 and ICD-10":
                proc = proc[
                    [
                        "subject_id",
                        "hadm_id",
                        "icd_code",
                        "chartdate",
                        "admittime",
                        "proc_time_from_admit",
                    ]
                ]
                print("Total number of rows", proc.shape[0])
                proc.dropna().to_csv(PREPROC_PROC_PATH, compression="gzip", index=False)
            elif self.group_proc == "ICD-10":
                proc = proc.loc[proc.icd_version == 10][
                    [
                        "subject_id",
                        "hadm_id",
                        "icd_code",
                        "chartdate",
                        "admittime",
                        "proc_time_from_admit",
                    ]
                ].dropna()
                print("Total number of rows", proc.shape[0])
                proc.to_csv(PREPROC_PROC_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

        if self.feature_extractor.for_labs:
            if self.clean_labs:
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
