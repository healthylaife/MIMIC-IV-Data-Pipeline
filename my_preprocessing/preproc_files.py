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


class OUT_ICU_HEADER(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    CHART_TIME = "charttime"
    IN_TIME = "intime"
    EVENT_TIME_FROM_ADMIT = "event_time_from_admit"
