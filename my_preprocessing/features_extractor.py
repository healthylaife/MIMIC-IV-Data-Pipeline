import pandas as pd
import logging
from my_preprocessing.preproc.cohort import COHORT_PATH
from my_preprocessing.features_extraction import (
    save_chart_events_features,
    save_diag_features,
    save_lab_events_features,
    save_output_features,
    save_procedures_features,
    save_medications_features,
)
from typing import List

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

    def load_cohort(self) -> pd.DataFrame:
        """Load cohort data from a CSV file."""
        cohort_path = COHORT_PATH / f"{self.cohort_output}.csv.gz"
        try:
            return pd.read_csv(
                cohort_path,
                compression="gzip",
                parse_dates=["intime" if self.use_icu else "admittime"],
            )
        except FileNotFoundError:
            logger.error(f"Cohort file not found at {cohort_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading cohort file: {e}")
            raise

    def save_features(self) -> List[pd.DataFrame]:
        cohort = self.load_cohort()
        feature_conditions = [
            (self.for_diagnoses, lambda: save_diag_features(cohort, self.use_icu)),
            (
                self.for_procedures,
                lambda: save_procedures_features(cohort, self.use_icu),
            ),
            (
                self.for_medications,
                lambda: save_medications_features(cohort, self.use_icu),
            ),
            (
                self.for_output_events and self.use_icu,
                lambda: save_output_features(cohort),
            ),
            (
                self.for_chart_events and self.use_icu,
                lambda: save_chart_events_features(cohort),
            ),
            (
                self.for_labs and not self.use_icu,
                lambda: save_lab_events_features(cohort),
            ),
        ]
        features = [
            feature_func()
            for condition, feature_func in feature_conditions
            if condition
        ]
        return features
