import pandas as pd
import logging
from tqdm import tqdm
from my_preprocessing.features_extractor import FeatureExtractor
from my_preprocessing.preproc_file_info import *

logger = logging.getLogger()


class FeatureSummaryGenerator:
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def generate_summary_icu(self):
        logger.info("[GENERATING FEATURE SUMMARY]")
        if self.feature_extractor.for_diagnoses:
            diag = pd.read_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
            freq = (
                diag.groupby(["stay_id", "new_icd_code"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["new_icd_code"])["mean_frequency"].mean().reset_index()
            total = diag.groupby("new_icd_code").size().reset_index(name="total_count")
            summary = pd.merge(freq, total, on="new_icd_code", how="right")
            summary = summary.fillna(0)
            summary.to_csv(DIAG_SUMMARY_PATH, index=False)
            summary["new_icd_code"].to_csv(DIAG_FEATURES_PATH, index=False)

        if self.feature_extractor.for_medications:
            med = pd.read_csv(PREPROC_MED_ICU_PATH, compression="gzip")
            freq = (
                med.groupby(["stay_id", "itemid"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

            missing = (
                med[med["amount"] == 0]
                .groupby("itemid")
                .size()
                .reset_index(name="missing_count")
            )
            total = med.groupby("itemid").size().reset_index(name="total_count")
            summary = pd.merge(missing, total, on="itemid", how="right")
            summary = pd.merge(freq, summary, on="itemid", how="right")
            summary = summary.fillna(0)
            summary.to_csv(MED_SUMMARY_PATH, index=False)
            summary["itemid"].to_csv(MED_FEATURES_PATH, index=False)

        if self.feature_extractor.for_procedures:
            proc = pd.read_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
            freq = (
                proc.groupby(["stay_id", "itemid"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
            total = proc.groupby("itemid").size().reset_index(name="total_count")
            summary = pd.merge(freq, total, on="itemid", how="right")
            summary = summary.fillna(0)
            summary.to_csv(PROC_SUMMARY_PATH, index=False)
            summary["itemid"].to_csv(PROC_FEATURES_PATH, index=False)

        if self.feature_extractor.for_output_events:
            out = pd.read_csv(PREPROC_OUT_ICU_PATH, compression="gzip")
            freq = (
                out.groupby(["stay_id", "itemid"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
            total = out.groupby("itemid").size().reset_index(name="total_count")
            summary = pd.merge(freq, total, on="itemid", how="right")
            summary = summary.fillna(0)
            summary.to_csv(OUT_SUMMARY_PATH, index=False)
            summary["itemid"].to_csv(OUT_FEATURES_PATH, index=False)

        if self.feature_extractor.for_chart_events:
            chart = pd.read_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
            freq = (
                chart.groupby(["stay_id", "itemid"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

            missing = (
                chart[chart["valuenum"] == 0]
                .groupby("itemid")
                .size()
                .reset_index(name="missing_count")
            )
            total = chart.groupby("itemid").size().reset_index(name="total_count")
            summary = pd.merge(missing, total, on="itemid", how="right")
            summary = pd.merge(freq, summary, on="itemid", how="right")
            summary = summary.fillna(0)
            summary.to_csv(CHART_SUMMARY_PATH, index=False)
            summary["itemid"].to_csv(CHART_FEATURES_PATH, index=False)

        print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")

    def generate_summary_hosp(self):
        print("[GENERATING FEATURE SUMMARY]")
        if self.feature_extractor.for_diagnoses:
            diag = pd.read_csv(PREPROC_DIAG_PATH, compression="gzip", header=0)
            freq = (
                diag.groupby(["hadm_id", "new_icd_code"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["new_icd_code"])["mean_frequency"].mean().reset_index()
            total = diag.groupby("new_icd_code").size().reset_index(name="total_count")
            summary = pd.merge(freq, total, on="new_icd_code", how="right")
            summary = summary.fillna(0)
            summary.to_csv(DIAG_SUMMARY_PATH, index=False)
            summary["new_icd_code"].to_csv(DIAG_FEATURES_PATH, index=False)

        if self.feature_extractor.for_medications:
            med = pd.read_csv(PREPROC_MED_PATH, compression="gzip")
            freq = (
                med.groupby(["hadm_id", "drug_name"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["drug_name"])["mean_frequency"].mean().reset_index()

            missing = (
                med[med["dose_val_rx"] == 0]
                .groupby("drug_name")
                .size()
                .reset_index(name="missing_count")
            )
            total = med.groupby("drug_name").size().reset_index(name="total_count")
            summary = pd.merge(missing, total, on="drug_name", how="right")
            summary = pd.merge(freq, summary, on="drug_name", how="right")
            summary["missing%"] = 100 * (
                summary["missing_count"] / summary["total_count"]
            )
            summary = summary.fillna(0)
            summary.to_csv(MED_SUMMARY_PATH, index=False)
            summary["drug_name"].to_csv(MED_FEATURES_PATH, index=False)

        if self.feature_extractor.for_procedures:
            proc = pd.read_csv(PREPROC_PROC_PATH, compression="gzip")
            freq = (
                proc.groupby(["hadm_id", "icd_code"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["icd_code"])["mean_frequency"].mean().reset_index()
            total = proc.groupby("icd_code").size().reset_index(name="total_count")
            summary = pd.merge(freq, total, on="icd_code", how="right")
            summary = summary.fillna(0)
            summary.to_csv(PROC_SUMMARY_PATH, index=False)
            summary["icd_code"].to_csv(PROC_FEATURES_PATH, index=False)

        if self.feature_extractor.for_labs:
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
            freq = (
                labs.groupby(["hadm_id", "itemid"])
                .size()
                .reset_index(name="mean_frequency")
            )
            freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

            missing = (
                labs[labs["valuenum"] == 0]
                .groupby("itemid")
                .size()
                .reset_index(name="missing_count")
            )
            total = labs.groupby("itemid").size().reset_index(name="total_count")
            summary = pd.merge(missing, total, on="itemid", how="right")
            summary = pd.merge(freq, summary, on="itemid", how="right")
            summary["missing%"] = 100 * (
                summary["missing_count"] / summary["total_count"]
            )
            summary = summary.fillna(0)
            summary.to_csv(LABS_SUMMARY_PATH, index=False)
            summary["itemid"].to_csv(LABS_FEATURES_PATH, index=False)

        logger.info("[SUCCESSFULLY SAVED FEATURE SUMMARY]")
