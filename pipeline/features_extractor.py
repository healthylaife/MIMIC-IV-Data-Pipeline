import pandas as pd
import logging
from pipeline.feature.feature_abc import Feature
from pipeline.feature.chart_events import Chart
from pipeline.feature.diagnoses import Diagnoses
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.file_info.preproc.cohort import load_cohort

from typing import List, Tuple

from pipeline.feature.lab_events import Lab

logger = logging.getLogger()


class FeatureExtractor:
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
        cohort = load_cohort(self.use_icu, self.cohort_output)
        feature_conditions: List[Tuple[bool, Feature]] = [
            (self.for_diagnoses, Diagnoses(cohort, self.use_icu)),
            (
                self.for_procedures,
                Procedures(cohort, self.use_icu),
            ),
            (
                self.for_medications,
                Medications(cohort, self.use_icu),
            ),
            (
                self.for_output_events and self.use_icu,
                OutputEvents(cohort),
            ),
            (
                self.for_chart_events and self.use_icu,
                Chart(cohort),
            ),
            (
                self.for_labs and not self.use_icu,
                Lab(cohort),
            ),
        ]
        features = []
        for condition, feature in feature_conditions:
            if condition:
                features.append(feature.extract_from(cohort))
                feature.save()

        return features
