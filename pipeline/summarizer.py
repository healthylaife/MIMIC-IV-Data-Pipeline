import pandas as pd
import logging

from tqdm import tqdm
from pipeline.feature.diagnoses import Diagnoses
from pipeline.feature.lab_events import Lab
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.features_extractor import FeatureExtractor
from typing import List

from pipeline.feature.chart_events import Chart
from pipeline.file_info.common import save_data
from pipeline.file_info.preproc.feature import (
    EXTRACT_CHART_ICU_PATH,
    EXTRACT_LABS_PATH,
    EXTRACT_MED_ICU_PATH,
    EXTRACT_MED_PATH,
    EXTRACT_OUT_ICU_PATH,
    EXTRACT_PROC_ICU_PATH,
    EXTRACT_PROC_PATH,
    PREPROC_DIAG_ICU_PATH,
    PREPROC_DIAG_PATH,
    ChartEventsHeader,
    IcuMedicationHeader,
    IcuProceduresHeader,
    LabEventsHeader,
    NonIcuProceduresHeader,
    OutputEventsHeader,
    PreprocDiagnosesHeader,
    PreprocMedicationHeader,
)
from pipeline.file_info.preproc.summary import (
    CHART_FEATURES_PATH,
    CHART_SUMMARY_PATH,
    DIAG_FEATURES_PATH,
    DIAG_SUMMARY_PATH,
    LABS_FEATURES_PATH,
    LABS_SUMMARY_PATH,
    MED_FEATURES_PATH,
    MED_SUMMARY_PATH,
    OUT_FEATURES_PATH,
    OUT_SUMMARY_PATH,
    PROC_FEATURES_PATH,
    PROC_SUMMARY_PATH,
)
from pipeline.no_event_feature_preprocessor import NoEventFeaturePreprocessor

logger = logging.getLogger()


class Summarizer:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
    ):
        self.feature_extractor = feature_extractor

    def save_summaries(self) -> List[pd.DataFrame]:
        summaries = []
        if self.feature_extractor.for_diagnoses:
            preproc_diag = pd.read_csv(
                PREPROC_DIAG_ICU_PATH
                if self.feature_extractor.use_icu
                else PREPROC_DIAG_PATH,
                compression="gzip",
            )
            dia = Diagnoses(
                use_icu=self.feature_extractor.use_icu,
                df=preproc_diag,
            )
            summary = dia.summary()
            save_data(summary, DIAG_SUMMARY_PATH, "DIAGNOSES SUMARY")
            summary[PreprocDiagnosesHeader.NEW_ICD_CODE].to_csv(
                DIAG_FEATURES_PATH, index=False
            )
            summaries.append(summary)
        if self.feature_extractor.for_medications:
            extract_med = pd.read_csv(
                EXTRACT_MED_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_MED_PATH,
                compression="gzip",
            )
            med = Medications(
                use_icu=self.feature_extractor.use_icu,
                df=extract_med,
            )
            summary = med.summary()
            save_data(summary, MED_SUMMARY_PATH, "MEDICATIONS SUMARY")
            feature_name = (
                IcuMedicationHeader.ITEM_ID
                if self.feature_extractor.use_icu
                else PreprocMedicationHeader.DRUG_NAME
            )
            summary[feature_name].to_csv(MED_FEATURES_PATH, index=False)
            summaries.append(summary)
        if self.feature_extractor.for_procedures:
            extract_proc = pd.read_csv(
                EXTRACT_PROC_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_PROC_PATH,
                compression="gzip",
            )
            proc = Procedures(
                use_icu=self.feature_extractor.use_icu,
                df=extract_proc,
            )
            summary = proc.summary()

            save_data(summary, PROC_SUMMARY_PATH, "PROCEDURES SUMARY")
            feature_name = (
                IcuProceduresHeader.ITEM_ID
                if self.feature_extractor.use_icu
                else NonIcuProceduresHeader.ICD_CODE
            )
            summary[feature_name].to_csv(PROC_FEATURES_PATH, index=False)
            summaries.append(summary)
        if self.feature_extractor.for_output_events:
            extract_out = pd.read_csv(EXTRACT_OUT_ICU_PATH, compression="gzip")
            out = OutputEvents(df=extract_out)
            summary = out.summary()
            save_data(summary, OUT_SUMMARY_PATH, "OUTPUT EVENTS SUMARY")
            summary[OutputEventsHeader.ITEM_ID].to_csv(OUT_FEATURES_PATH, index=False)
            summaries.append(summary)
        if self.feature_extractor.for_chart_events:
            extract_chart = pd.read_csv(EXTRACT_CHART_ICU_PATH, compression="gzip")
            chart = Chart(df=extract_chart)
            summary = chart.summary()
            save_data(summary, CHART_SUMMARY_PATH, "CHART EVENTS SUMARY")
            summary[ChartEventsHeader.ITEM_ID].to_csv(CHART_FEATURES_PATH, index=False)
            summaries.append(summary)
        if self.feature_extractor.for_labs:
            labs = pd.DataFrame()
            for chunk in tqdm(
                pd.read_csv(
                    EXTRACT_LABS_PATH,
                    compression="gzip",
                    chunksize=10000000,
                )
            ):
                if labs.empty:
                    labs = chunk
                else:
                    labs = labs.append(chunk, ignore_index=True)
            lab = Lab(df=labs)
            summary = lab.summary()
            summary.to_csv(LABS_SUMMARY_PATH, index=False)
            summary[LabEventsHeader.ITEM_ID].to_csv(LABS_FEATURES_PATH, index=False)
            summaries.append(lab.summary)
        return summaries
