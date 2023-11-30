import pandas as pd
import logging
from tqdm import tqdm
from my_preprocessing.icd_conversion import standardize_icd
from my_preprocessing.uom_conversion import drop_wrong_uom
from my_preprocessing.outlier_removal import outlier_imputation
from my_preprocessing.raw_file_info import (
    load_hosp_diagnosis_icd,
)
from my_preprocessing.preproc_file_info import (
    PREPROC_PROC_ICU_PATH,
    PREPROC_MED_ICU_PATH,
    PREPROC_LABS_PATH,
    PREPROC_OUT_ICU_PATH,
    PREPROC_DIAG_ICU_PATH,
    PREPROC_CHART_ICU_PATH,
    PREPROC_MED_PATH,
)
from my_preprocessing.icu_features import (
    make_output_events,
    make_chart_events,
    make_icu_procedure_events,
    make_icu_input_events,
)
from my_preprocessing.hosp_features import (
    make_labs_events_features,
    make_hosp_prescriptions,
    make_hosp_procedures_icd,
)

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


def save_diag_features(cohort: pd.DataFrame, use_icu: bool) -> pd.DataFrame:
    print("[EXTRACTING DIAGNOSIS DATA]")
    hosp_diagnose = load_hosp_diagnosis_icd()
    admission_cohort = cohort
    admissions_cohort_cols = (
        ["hadm_id", "stay_id", "label"] if use_icu else ["hadm_id", "label"]
    )
    diag = hosp_diagnose.merge(
        admission_cohort[admissions_cohort_cols],
        how="inner",
        left_on="hadm_id",
        right_on="hadm_id",
    )
    cols = DIAGNOSIS_ICU_HEADERS if use_icu else DIAGNOSIS_NON_ICU_HEADERS
    diag = standardize_icd(diag)[cols]
    diag.to_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    return diag


def save_output_features(cohort: pd.DataFrame) -> pd.DataFrame:
    print("[EXTRACTING OUPTPUT EVENTS DATA]")
    out = make_output_events(cohort)
    out = out[OUTPUT_ICU_HEADERS]
    out.to_csv(PREPROC_OUT_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    return out


def save_chart_events_features(cohort: pd.DataFrame) -> pd.DataFrame:
    print("[EXTRACTING CHART EVENTS DATA]")
    chart = make_chart_events(cohort)
    chart = drop_wrong_uom(chart, 0.95)
    chart = chart[CHART_EVENT_HEADERS]
    chart.to_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
    return chart


def save_icu_procedures_features(cohort: pd.DataFrame) -> pd.DataFrame:
    print("[EXTRACTING PROCEDURES DATA]")
    proc = make_icu_procedure_events(cohort)
    proc = proc[PROCEDURES_ICD_ICU_HEADERS]
    proc.to_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    return proc


def save_hosp_procedures_icd_features(cohort: pd.DataFrame) -> pd.DataFrame:
    print("[EXTRACTING PROCEDURES DATA]")
    proc = make_hosp_procedures_icd(cohort)
    proc = proc[PROCEDURES_ICD_NON_ICU_HEADERS]
    proc.to_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    return proc


def save_icu_input_events_features(cohort: pd.DataFrame) -> pd.DataFrame:
    print("[EXTRACTING MEDICATIONS DATA]")
    med = make_icu_input_events(cohort)
    med = med[INPUT_EVENTS_HEADERS]
    med.to_csv(PREPROC_MED_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    return med


def save_lab_events_features(cohort: pd.DataFrame) -> pd.DataFrame:
    print("[EXTRACTING LABS DATA]")
    labevents = make_labs_events_features(cohort)
    labevents = drop_wrong_uom(labevents, 0.95)
    labevents = labevents[LAB_EVENTS_HEADERS]
    labevents.to_csv(PREPROC_LABS_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED LABS DATA]")
    return labevents


def save_hosp_prescriptions_features(cohort: pd.DataFrame) -> pd.DataFrame:
    print("[EXTRACTING MEDICATIONS DATA]")
    prescriptions = make_hosp_prescriptions(cohort)
    prescriptions = prescriptions[PRESCRIPTIONS_HEADERS]
    prescriptions.to_csv(PREPROC_MED_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    return prescriptions
