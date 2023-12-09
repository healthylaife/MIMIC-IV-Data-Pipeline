import pandas as pd
import logging
from pipeline.feature.diagnoses import Diagnoses, IcdGroupOption
from pipeline.feature.lab_events import Lab
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.feature_selector import FeatureSelector
from pipeline.features_extractor import FeatureExtractor
from typing import List
from pathlib import Path

from pipeline.feature.chart_events import Chart
from pipeline.file_info.common import save_data
from pipeline.file_info.preproc.feature import (
    EXTRACT_DIAG_ICU_PATH,
    EXTRACT_DIAG_PATH,
    PREPROC_DIAG_ICU_PATH,
    PREPROC_DIAG_PATH,
)

logger = logging.getLogger()


class NoEventFeaturePreprocessor:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        group_diag_icd: IcdGroupOption,
        group_med_code: bool,
        keep_proc_icd9: bool,
    ):
        self.feature_extractor = feature_extractor
        self.group_diag_icd = group_diag_icd
        self.group_med_code = group_med_code
        self.keep_proc_icd9 = keep_proc_icd9

    def preprocess(self):
        no_event_preproc_features = []
        empty_cohort = pd.DataFrame()
        if self.feature_extractor.for_diagnoses:
            extract_diag_df = pd.read_csv(
                EXTRACT_DIAG_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_DIAG_PATH,
                compression="gzip",
            )
            dia = Diagnoses(
                cohort=empty_cohort,
                use_icu=self.feature_extractor.use_icu,
                df=extract_diag_df,
            )
            dia.preproc(self.group_diag_icd)
            save_data(
                dia.df,
                PREPROC_DIAG_ICU_PATH
                if self.feature_extractor.use_icu
                else PREPROC_DIAG_PATH,
                "DIAGNOSES",
            )
            no_event_preproc_features.append(dia.preproc(self.group_diag_icd))
        if not self.feature_extractor.use_icu:
            if self.feature_extractor.for_medications:
                med = Medications(
                    cohort=empty_cohort,
                    use_icu=self.feature_extractor.use_icu,
                )
                no_event_preproc_features.append(med.preproc(self.group_med_code))
            if self.feature_extractor.for_procedures:
                proc = Procedures(
                    cohort=empty_cohort,
                    use_icu=self.feature_extractor.use_icu,
                )
                no_event_preproc_features.append(proc.preproc(self.keep_proc_icd9))
        return no_event_preproc_features
