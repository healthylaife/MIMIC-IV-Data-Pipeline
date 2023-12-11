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

from pipeline.file_info.common import save_data
from pipeline.file_info.preproc.feature import (
    EXTRACT_DIAG_ICU_PATH,
    EXTRACT_DIAG_PATH,
    EXTRACT_MED_PATH,
    EXTRACT_PROC_PATH,
    EXTRACT_DIAG_ICU_PATH,
    EXTRACT_DIAG_PATH,
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

    def preprocess(self) -> List[pd.DataFrame]:
        no_event_preproc_features = []
        if self.feature_extractor.for_diagnoses:
            extract_dia = pd.read_csv(
                EXTRACT_DIAG_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_DIAG_PATH,
                compression="gzip",
            )
            dia = Diagnoses(
                use_icu=self.feature_extractor.use_icu,
                df=extract_dia,
            )
            preproc_dia = dia.preproc(self.group_diag_icd)
            save_data(
                preproc_dia,
                EXTRACT_DIAG_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_DIAG_PATH,
                "DIAGNOSES",
            )
            no_event_preproc_features.append(preproc_dia)
        if not self.feature_extractor.use_icu:
            if self.feature_extractor.for_medications:
                extract_med = pd.read_csv(EXTRACT_MED_PATH, compression="gzip")
                med = Medications(use_icu=False, df=extract_med)
                preproc_med = med.preproc(self.group_med_code)
                save_data(preproc_med, EXTRACT_MED_PATH, "MEDICATIONS")

                no_event_preproc_features.append(preproc_med)
            if self.feature_extractor.for_procedures:
                extract_proc = pd.read_csv(EXTRACT_PROC_PATH, compression="gzip")
                proc = Procedures(use_icu=False, df=extract_proc)
                preproc_proc = proc.preproc(self.keep_proc_icd9)
                save_data(preproc_proc, EXTRACT_PROC_PATH, "PROCEDURES")
                no_event_preproc_features.append(preproc_proc)
        return no_event_preproc_features
