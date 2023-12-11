from pathlib import Path
import pandas as pd
import logging
from pipeline.feature.feature_abc import Feature
from pipeline.feature.chart_events import Chart
from pipeline.feature.diagnoses import Diagnoses
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.file_info.common import save_data
from pipeline.file_info.preproc.cohort import load_cohort
from pipeline.feature.lab_events import Lab
from typing import List, Tuple

from pipeline.file_info.preproc.feature import (
    EXTRACT_CHART_ICU_PATH,
    EXTRACT_DIAG_ICU_PATH,
    EXTRACT_DIAG_PATH,
    EXTRACT_LABS_PATH,
    EXTRACT_MED_ICU_PATH,
    EXTRACT_MED_PATH,
    EXTRACT_OUT_ICU_PATH,
    EXTRACT_PROC_ICU_PATH,
    EXTRACT_PROC_PATH,
)
from pipeline.feature.feature_abc import Name

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class FeatureExtractor:
    """
    Extracts various features from a cohort based on specified conditions.

    Attributes:
        cohort_output (str): Output path or identifier for the cohort.
        use_icu (bool): Flag to indicate whether ICU data should be used.
        for_diagnoses (bool): Flag to extract diagnosis features.
        for_output_events (bool): Flag to extract output event features.
        for_chart_events (bool): Flag to extract chart event features.
        for_procedures (bool): Flag to extract procedure features.
        for_medications (bool): Flag to extract medication features.
        for_labs (bool): Flag to extract lab event features.
    """

    def __init__(
        self,
        cohort_output: str,
        use_icu: bool,
        for_diagnoses: bool,
        for_output_events: bool,
        for_chart_events: bool,
        for_procedures: bool,
        for_medications: bool,
        for_labs: bool,
    ):
        self.cohort_output = cohort_output
        self.use_icu = use_icu
        self.for_diagnoses = for_diagnoses
        self.for_output_events = for_output_events
        self.for_chart_events = for_chart_events
        self.for_procedures = for_procedures
        self.for_medications = for_medications
        self.for_labs = for_labs

    def save_features(self) -> List[pd.DataFrame]:
        """
        Loads the cohort and extracts features based on the specified conditions.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each containing a type of extracted feature.
        """
        cohort = load_cohort(self.use_icu, self.cohort_output)
        feature_conditions: List[Tuple[bool, Feature, Path]] = [
            (
                self.for_diagnoses,
                Diagnoses(use_icu=self.use_icu),
                EXTRACT_DIAG_ICU_PATH if self.use_icu else EXTRACT_DIAG_PATH,
            ),
            (
                self.for_procedures,
                Procedures(use_icu=self.use_icu),
                EXTRACT_PROC_ICU_PATH if self.use_icu else EXTRACT_PROC_PATH,
            ),
            (
                self.for_medications,
                Medications(use_icu=self.use_icu),
                EXTRACT_MED_ICU_PATH if self.use_icu else EXTRACT_MED_PATH,
            ),
            (
                self.for_output_events and self.use_icu,
                OutputEvents(),
                EXTRACT_OUT_ICU_PATH,
            ),
            (
                self.for_chart_events and self.use_icu,
                Chart(),
                EXTRACT_CHART_ICU_PATH,
            ),
            (
                self.for_labs and not self.use_icu,
                Lab(),
                EXTRACT_LABS_PATH,
            ),
        ]
        features = {}
        for condition, feature, path in feature_conditions:
            if condition:
                extract_feature = feature.extract_from(cohort)
                feature_name = feature.__class__.name()
                features[feature_name] = extract_feature
                save_data(extract_feature, path, feature_name)

        return features
