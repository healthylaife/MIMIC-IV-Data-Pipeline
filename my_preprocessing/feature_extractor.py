import pandas as pd
import logging
from my_preprocessing.preproc_file_info import COHORT_PATH
from my_preprocessing.feature_extractor_utils import (
    save_chart_events_features,
    save_hosp_prescriptions_features,
    save_diag_features,
    save_hosp_procedures_icd_features,
    save_icu_input_events_features,
    save_icu_procedures_features,
    save_lab_events_features,
    save_output_features,
)


DIAGNOSIS_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "icd_code",
    "root_icd10_convert",
    "root",
]

DIAGNOSIS_NON_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "root_icd10_convert",
    "root",
]

OUTPUT_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "charttime",
    "intime",
    "event_time_from_admit",
]

PROCEDURES_ICD_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "starttime",
    "intime",
    "event_time_from_admit",
]

PROCEDURES_ICD_NON_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "icd_version",
    "chartdate",
    "admittime",
    "proc_time_from_admit",
]

LAB_EVENTS_HEADERS = [
    "subject_id",
    "hadm_id",
    "charttime",
    "itemid",
    "admittime",
    "lab_time_from_admit",
    "valuenum",
]

PRESCRIPTIONS_HEADERS = [
    "subject_id",
    "hadm_id",
    "starttime",
    "stoptime",
    "drug",
    "nonproprietaryname",
    "start_hours_from_admit",
    "stop_hours_from_admit",
    "dose_val_rx",
]

INPUT_EVENTS_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "starttime",
    "endtime",
    "start_hours_from_admit",
    "stop_hours_from_admit",
    "rate",
    "amount",
    "orderid",
]

CHART_EVENT_HEADERS = ["stay_id", "itemid", "event_time_from_admit", "valuenum"]


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

    def save_icu_features(self):
        diag, out, chart, proc, med = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        if not self.use_icu:
            return diag, out, chart, proc, med
        cohort_path = COHORT_PATH / (self.cohort_output + ".csv.gz")
        cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["intime"])
        if self.for_diagnoses:
            diag = save_diag_features(cohort, use_icu=True)
        if self.for_output_events:
            out = save_output_features(cohort)
        if self.for_chart_events:
            chart = save_chart_events_features(cohort)
        if self.for_procedures:
            proc = save_icu_procedures_features(cohort)
        if self.for_medications:
            med = save_icu_input_events_features(cohort)
        return diag, out, chart, proc, med

    def save_non_icu_features(self):
        diag, lab, proc, med = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        if self.use_icu:
            return diag, lab, proc, med
        cohort_path = COHORT_PATH / (self.cohort_output + ".csv.gz")
        cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["admittime"])
        if self.for_diagnoses:
            diag = save_diag_features(cohort, use_icu=False)
        if self.for_labs:
            lab = save_lab_events_features(cohort)
        if self.for_procedures:
            proc = save_hosp_procedures_icd_features(cohort)
        if self.for_medications:
            med = save_hosp_prescriptions_features(cohort)
        return diag, lab, proc, med
