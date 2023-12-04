import pandas as pd
import logging
from tqdm import tqdm
from my_preprocessing.features_extractor import FeatureExtractor
from my_preprocessing.preproc.feature import *
from my_preprocessing.preproc.summary import *

logger = logging.getLogger()

MEAN_FREQUENCY_HEADER = "mean_frequency"


class FeatureSummaryGenerator:
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def save_summaries(self):
        summaries = []
        if self.feature_extractor.for_diagnoses:
            summaries.append(self.diag_summary())
        if self.feature_extractor.for_medications:
            summaries.append(self.med_summary())
        if self.feature_extractor.for_procedures:
            summaries.append(self.proc_summary())
        if self.feature_extractor.for_output_events:
            summaries.append(self.out_summary())
        if self.feature_extractor.for_chart_events:
            summaries.append(self.chart_summary())
        if self.feature_extractor.for_labs:
            summaries.append(self.lab_summary())
        return summaries

    def diag_summary(self) -> pd.DataFrame:
        diag = pd.read_csv(
            PREPROC_DIAG_ICU_PATH
            if self.feature_extractor.use_icu
            else PREPROC_DIAG_PATH,
            compression="gzip",
        )
        freq = (
            diag.groupby(
                [
                    DiagnosesIcuHeader.STAY_ID
                    if self.feature_extractor.use_icu
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

    def med_summary(self) -> pd.DataFrame:
        path = (
            PREPROC_MED_ICU_PATH if self.feature_extractor.use_icu else PREPROC_MED_PATH
        )
        med = pd.read_csv(path, compression="gzip")
        feature_name = (
            IcuMedicationHeader.ITEM_ID.value
            if self.feature_extractor.use_icu
            else PreprocMedicationHeader.DRUG_NAME.value
        )
        freq = (
            med.groupby(
                [IcuMedicationHeader.STAY_ID, IcuMedicationHeader.ITEM_ID]
                if self.feature_extractor.use_icu
                else [
                    MedicationsHeader.HOSPITAL_ADMISSION_ID,
                    PreprocMedicationHeader.DRUG_NAME,
                ]
            )
            .size()
            .reset_index(name="mean_frequency")
        )

        missing = (
            med[
                med[
                    IcuMedicationHeader.AMOUNT
                    if self.feature_extractor.use_icu
                    else NonIcuMedicationHeader.DOSE_VAL_RX
                ]
                == 0
            ]
            .groupby(feature_name)
            .size()
            .reset_index(name="missing_count")
        )
        total = med.groupby(feature_name).size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on=feature_name, how="right")
        summary = pd.merge(freq, summary, on=feature_name, how="right")
        summary["missing%"] = 100 * (summary["missing_count"] / summary["total_count"])
        summary = summary.fillna(0)
        summary.to_csv(MED_SUMMARY_PATH, index=False)
        summary[feature_name].to_csv(MED_FEATURES_PATH, index=False)
        return summary

    def proc_summary(self) -> pd.DataFrame:
        proc = pd.read_csv(
            PREPROC_PROC_ICU_PATH
            if self.feature_extractor.use_icu
            else PREPROC_PROC_PATH,
            compression="gzip",
        )
        feature_name = (
            IcuProceduresHeader.ITEM_ID
            if self.feature_extractor.use_icu
            else NonIcuProceduresHeader.ICD_CODE
        )
        freq = (
            proc.groupby(
                [
                    "stay_id" if self.feature_extractor.use_icu else "hadm_id",
                    feature_name,
                ]
            )
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(feature_name)["mean_frequency"].mean().reset_index()
        total = proc.groupby(feature_name).size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on=feature_name, how="right")
        summary = summary.fillna(0)
        summary.to_csv(PROC_SUMMARY_PATH, index=False)
        summary[feature_name].to_csv(PROC_FEATURES_PATH, index=False)
        return summary[feature_name]

    def out_summary(self) -> pd.DataFrame:
        out = pd.read_csv(PREPROC_OUT_ICU_PATH, compression="gzip")
        freq = (
            out.groupby([OutputEventsHeader.STAY_ID, OutputEventsHeader.ITEM_ID])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = (
            out.groupby(OutputEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(freq, total, on=OutputEventsHeader.ITEM_ID, how="right")
        summary = summary.fillna(0)
        summary.to_csv(OUT_SUMMARY_PATH, index=False)
        summary[OutputEventsHeader.ITEM_ID].to_csv(OUT_FEATURES_PATH, index=False)
        return summary[OutputEventsHeader.ITEM_ID]

    def chart_summary(self) -> pd.DataFrame:
        chart = pd.read_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
        freq = (
            chart.groupby([ChartEventsHeader.STAY_ID, ChartEventsHeader.ITEM_ID])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = (
            freq.groupby([ChartEventsHeader.ITEM_ID])["mean_frequency"]
            .mean()
            .reset_index()
        )

        missing = (
            chart[chart[ChartEventsHeader.VALUE_NUM] == 0]
            .groupby(ChartEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="missing_count")
        )
        total = (
            chart.groupby(ChartEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(missing, total, on=ChartEventsHeader.ITEM_ID, how="right")
        summary = pd.merge(freq, summary, on=ChartEventsHeader.ITEM_ID, how="right")
        summary = summary.fillna(0)
        summary.to_csv(CHART_SUMMARY_PATH, index=False)
        summary[ChartEventsHeader.ITEM_ID].to_csv(CHART_FEATURES_PATH, index=False)

    def lab_summary(self) -> pd.DataFrame:
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
