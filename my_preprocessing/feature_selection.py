from my_preprocessing.icd_conversion import standardize_icd
from my_preprocessing.uom_conversion import drop_wrong_uom
import pandas as pd
from my_preprocessing.raw_files import (
    load_hosp_diagnosis_icd,
    FEATURE_PATH,
    COHORT_PATH,
)
from my_preprocessing.preprocessing import (
    preproc_output_events,
    preproc_chartevents,
    preproc_icu_procedure_events,
    preprocess_icu_input_events,
)
from my_preprocessing.hosp_features import (
    preproc_labs_events_features,
    preprocess_hosp_prescriptions,
    preproc_hosp_procedures_icd,
)


DIAGNOSIS_ICU_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "icd_code",
    "root_icd10_convert",
    "root",
]

DIAGNOSIS_NON_ICU_COLUMNS = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "root_icd10_convert",
    "root",
]

OUTPUT_ICU_COLUNMS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "charttime",
    "intime",
    "event_time_from_admit",
]

PROCEDURES_ICD_ICU_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "starttime",
    "intime",
    "event_time_from_admit",
]

PROCEDURES_ICD_NON_ICU_COLUMNS = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "icd_version",
    "chartdate",
    "admittime",
    "proc_time_from_admit",
]

LAB_EVENTS_COLUNMS = [
    "subject_id",
    "hadm_id",
    "charttime",
    "itemid",
    "admittime",
    "lab_time_from_admit",
    "valuenum",
]

PRESCRIPTIONS_COLUMNS = [
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

INPUT_EVENTS_COLUMNS = [
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

CHART_EVENT_COLUMNS = ["stay_id", "itemid", "event_time_from_admit", "valuenum"]


def save_diag_features(cohort_output: str, use_icu: bool) -> pd.DataFrame:
    print("[EXTRACTING DIAGNOSIS DATA]")
    hosp_diagnose = load_hosp_diagnosis_icd()
    admission_cohort = pd.read_csv(
        COHORT_PATH / (cohort_output + ".csv.gz"), compression="gzip"
    )
    admissions_cohort_cols = (
        ["hadm_id", "stay_id", "label"] if use_icu else ["hadm_id", "label"]
    )
    diag = hosp_diagnose.merge(
        admission_cohort[admissions_cohort_cols],
        how="inner",
        left_on="hadm_id",
        right_on="hadm_id",
    )
    cols = DIAGNOSIS_ICU_COLUMNS if use_icu else DIAGNOSIS_NON_ICU_COLUMNS
    diag = standardize_icd(diag)[cols]
    diag.to_csv(FEATURE_PATH / "preproc_diag_icu.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    return diag


def save_output_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING OUPTPUT EVENTS DATA]")
    out = preproc_output_events(COHORT_PATH / (cohort_output + ".csv.gz"))
    out = out[OUTPUT_ICU_COLUNMS]
    out.to_csv(FEATURE_PATH / "preproc_out_icu.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    return out


def save_chart_events_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING CHART EVENTS DATA]")
    chart = preproc_chartevents(COHORT_PATH / (cohort_output + ".csv.gz"))
    chart = drop_wrong_uom(chart, 0.95)
    chart = chart[CHART_EVENT_COLUMNS]
    chart.to_csv(FEATURE_PATH / "preproc_chart_icu.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
    return chart


def save_icu_procedures_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING PROCEDURES DATA]")
    proc = preproc_icu_procedure_events(COHORT_PATH / (cohort_output + ".csv.gz"))
    cols = PROCEDURES_ICD_ICU_COLUMNS
    proc = proc[cols]
    proc.to_csv(FEATURE_PATH / "preproc_proc_icu.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    return proc


def save_hosp_procedures_icd_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING PROCEDURES DATA]")
    proc = preproc_hosp_procedures_icd(COHORT_PATH / (cohort_output + ".csv.gz"))
    cols = PROCEDURES_ICD_NON_ICU_COLUMNS
    proc = proc[cols]
    proc.to_csv(FEATURE_PATH / "preproc_proc_icu.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    return proc


def save_icu_input_events_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING MEDICATIONS DATA]")
    med = preprocess_icu_input_events(COHORT_PATH / (cohort_output + ".csv.gz"))
    med = med[INPUT_EVENTS_COLUMNS]
    med.to_csv(FEATURE_PATH / "preproc_med_icu.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    return med


def save_lab_events_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING LABS DATA]")
    labevents = preproc_labs_events_features(COHORT_PATH / (cohort_output + ".csv.gz"))
    labevents = drop_wrong_uom(labevents, 0.95)
    labevents = labevents[LAB_EVENTS_COLUNMS]
    labevents.to_csv(FEATURE_PATH / "preproc_labs.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED LABS DATA]")
    return labevents


def save_hosp_prescriptions_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING MEDICATIONS DATA]")
    prescriptions = preprocess_hosp_prescriptions(
        COHORT_PATH / (cohort_output + ".csv.gz")
    )
    prescriptions = prescriptions[PRESCRIPTIONS_COLUMNS]
    prescriptions.to_csv(FEATURE_PATH / "preproc_med.csv.gz", compression="gzip")
    print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    return prescriptions


def feature_icu(
    cohort_output,
    diag_flag=True,
    out_flag=True,
    chart_flag=True,
    proc_flag=True,
    med_flag=True,
):
    diag, out, chart, proc, med = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    if diag_flag:
        diag = save_diag_features(cohort_output, use_icu=True)

    if out_flag:
        out = save_output_features(cohort_output)

    if chart_flag:
        chart = save_chart_events_features(cohort_output)

    if proc_flag:
        proc = save_icu_procedures_features(cohort_output)

    if med_flag:
        med = save_icu_input_events_features(cohort_output)
    return diag, out, chart, proc, med


def feature_non_icu(
    cohort_output, diag_flag=True, lab_flag=True, proc_flag=True, med_flag=True
):
    diag, lab, proc, med = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    if diag_flag:
        diag = save_diag_features(cohort_output, use_icu=False)

    if lab_flag:
        lab = save_lab_events_features(cohort_output)

    if proc_flag:
        proc = save_hosp_procedures_icd_features(cohort_output)

    if med_flag:
        med = save_hosp_prescriptions_features(cohort_output)
    return diag, lab, proc, med
