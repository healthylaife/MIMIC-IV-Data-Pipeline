from enum import StrEnum
from pathlib import Path
import pandas as pd

PREPROC_PATH = Path("preproc_data")

COHORT_PATH = PREPROC_PATH / "cohort"
SUMMARY_PATH = PREPROC_PATH / "summary"

FEATURE_PATH = PREPROC_PATH / "features"

PREPROC_DIAG_PATH = FEATURE_PATH / "preproc_diag.csv.gz"
PREPROC_PROC_PATH = FEATURE_PATH / "preproc_proc.csv.gz"
PREPROC_PROC_ICU_PATH = FEATURE_PATH / "preproc_proc_icu.csv.gz"
PREPROC_MED_ICU_PATH = FEATURE_PATH / "preproc_med_icu.csv.gz"
PREPROC_LABS_PATH = FEATURE_PATH / "preproc_labs.csv.gz"
PREPROC_OUT_ICU_PATH = FEATURE_PATH / "preproc_out_icu.csv.gz"
PREPROC_DIAG_ICU_PATH = FEATURE_PATH / "preproc_diag_icu.csv.gz"
PREPROC_CHART_ICU_PATH = FEATURE_PATH / "preproc_chart_icu.csv.gz"
PREPROC_MED_PATH = FEATURE_PATH / "preproc_med.csv.gz"

DIAG_FEATURES_PATH = SUMMARY_PATH / "diag_features.csv"
DIAG_SUMMARY_PATH = SUMMARY_PATH / "diag_summary.csv"
MED_FEATURES_PATH = SUMMARY_PATH / "med_features.csv"
MED_SUMMARY_PATH = SUMMARY_PATH / "med_summary.csv"
OUT_FEATURES_PATH = SUMMARY_PATH / "out_features.csv"
OUT_SUMMARY_PATH = SUMMARY_PATH / "out_summary.csv"
PROC_FEATURES_PATH = SUMMARY_PATH / "proc_features.csv"
PROC_SUMMARY_PATH = SUMMARY_PATH / "proc_summary.csv"

LABS_FEATURES_PATH = SUMMARY_PATH / "labs_features.csv"
LABS_SUMMARY_PATH = SUMMARY_PATH / "labs_summary.csv"


CHART_FEATURES_PATH = SUMMARY_PATH / "chart_features.csv"
CHART_SUMMARY_PATH = SUMMARY_PATH / "chart_summary.csv"


class OUT_ICU_HEADER(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    CHART_TIME = "charttime"
    IN_TIME = "intime"
    EVENT_TIME_FROM_ADMIT = "event_time_from_admit"


class CohortHeader(StrEnum):
    PATIENT_ID = "subject_id"
    LABEL = "label"
    AGE = "age"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    INSURANCE = "insurance"
    ETHICITY = "ethnicity"
