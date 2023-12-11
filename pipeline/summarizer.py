import pandas as pd
import logging

from tqdm import tqdm
from pipeline.feature.feature_abc import Feature
from pipeline.feature.diagnoses import Diagnoses
from pipeline.feature.lab_events import Lab
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.features_extractor import FeatureExtractor
from typing import List, Type

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
    EXTRACT_DIAG_PATH,
    EXTRACT_DIAG_ICU_PATH,
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
from pathlib import Path

logger = logging.getLogger()


class Summarizer:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
    ):
        self.feature_extractor = feature_extractor

    def process_feature(
        self,
        feature_class: Type[Feature],
        path: Path,
        summary_path: Path,
        feature_name: str,
        features_path: str,
        use_icu: bool = True,
    ) -> pd.DataFrame:
        """
        Process a feature, save its summary, and export relevant data to a CSV file.
        """
        feature = (
            feature_class(
                use_icu=self.feature_extractor.use_icu,
                df=pd.read_csv(path, compression="gzip"),
            )
            if use_icu
            else feature_class(df=pd.read_csv(path, compression="gzip"))
        )
        summary = feature.summary()
        save_data(summary, summary_path, f"{feature_class.__name__.upper()} SUMMARY")
        summary[feature_name].to_csv(features_path, index=False)
        return summary

    def save_summaries(self) -> List[pd.DataFrame]:
        summaries = []
        if self.feature_extractor.for_diagnoses:
            summary = self.process_feature(
                Diagnoses,
                EXTRACT_DIAG_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_DIAG_PATH,
                DIAG_SUMMARY_PATH,
                PreprocDiagnosesHeader.NEW_ICD_CODE,
                DIAG_FEATURES_PATH,
            )
            summaries.append(summary)
        if self.feature_extractor.for_medications:
            summary = self.process_feature(
                Medications,
                EXTRACT_MED_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_MED_PATH,
                MED_SUMMARY_PATH,
                IcuMedicationHeader.ITEM_ID
                if self.feature_extractor.use_icu
                else PreprocMedicationHeader.DRUG_NAME,
                MED_FEATURES_PATH,
            )
            summaries.append(summary)

        if self.feature_extractor.for_procedures:
            summary = self.process_feature(
                Procedures,
                EXTRACT_PROC_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_PROC_PATH,
                PROC_SUMMARY_PATH,
                IcuProceduresHeader.ITEM_ID
                if self.feature_extractor.use_icu
                else NonIcuProceduresHeader.ICD_CODE,
                PROC_FEATURES_PATH,
            )
            summaries.append(summary)

        if self.feature_extractor.for_output_events:
            summary = self.process_feature(
                OutputEvents,
                EXTRACT_OUT_ICU_PATH,
                OUT_SUMMARY_PATH,
                OutputEventsHeader.ITEM_ID,
                OUT_FEATURES_PATH,
                use_icu=False,
            )
            summaries.append(summary)

        if self.feature_extractor.for_chart_events:
            summary = self.process_feature(
                Chart,
                EXTRACT_CHART_ICU_PATH,
                CHART_SUMMARY_PATH,
                ChartEventsHeader.ITEM_ID,
                CHART_FEATURES_PATH,
                use_icu=False,
            )
            summaries.append(summary)

        if self.feature_extractor.for_labs:
            # Special handling for labs by chunk
            labs = pd.concat(
                tqdm(
                    pd.read_csv(
                        EXTRACT_LABS_PATH, compression="gzip", chunksize=10000000
                    )
                ),
                ignore_index=True,
            )
            lab = Lab(df=labs)
            summary = lab.summary()
            save_data(summary, LABS_SUMMARY_PATH, "LABS SUMMARY")
            summary[LabEventsHeader.ITEM_ID].to_csv(LABS_FEATURES_PATH, index=False)
            summaries.append(summary)
        return summaries
