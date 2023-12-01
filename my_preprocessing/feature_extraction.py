import pandas as pd
import logging
from my_preprocessing.icd_conversion import standardize_icd
from my_preprocessing.uom_conversion import drop_wrong_uom
from my_preprocessing.raw_file_info import load_hosp_diagnosis_icd
from my_preprocessing.preproc_file_info import *
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
from pathlib import Path

logger = logging.getLogger()


def save_data(data: pd.DataFrame, path: Path, data_name: str) -> pd.DataFrame:
    """Save DataFrame to specified path."""
    data.to_csv(path, compression="gzip")
    logger.info(f"[SUCCESSFULLY SAVED {data_name} DATA]")
    return data


def save_diag_features(cohort: pd.DataFrame, use_icu: bool) -> pd.DataFrame:
    logger.info("[EXTRACTING DIAGNOSIS DATA]")
    hosp_diagnose = load_hosp_diagnosis_icd()
    admission_cohort = cohort
    admissions_cohort_cols = (
        [CohortHeader.HOSPITAL_ADMISSION_ID, CohortHeader.STAY_ID, CohortHeader.LABEL]
        if use_icu
        else [CohortHeader.HOSPITAL_ADMISSION_ID, CohortHeader.LABEL]
    )
    diag = hosp_diagnose.merge(
        admission_cohort[admissions_cohort_cols],
        on=DiagnosesHeader.HOSPITAL_ADMISSION_ID,
    )
    cols = [h.value for h in DiagnosesHeader]
    if use_icu:
        cols = cols + [h.value for h in DiagnosesIcuHeader]

    diag = standardize_icd(diag)[cols]
    return save_data(diag, PREPROC_DIAG_ICU_PATH, "DIAGNOSES")


def save_procedures_features(cohort: pd.DataFrame, use_icu: bool) -> pd.DataFrame:
    logger.info("[EXTRACTING PROCEDURES DATA]")
    proc = (
        make_icu_procedure_events(cohort)
        if use_icu
        else make_hosp_procedures_icd(cohort)
    )
    cols = [h.value for h in ProceduresHeader] + [
        h.value for h in (IcuProceduresHeader if use_icu else NonIcuProceduresHeader)
    ]
    proc = proc[cols]
    return save_data(
        proc, PREPROC_PROC_ICU_PATH if use_icu else PREPROC_PROC_PATH, "PROCEDURES"
    )


def save_medications_features(cohort: pd.DataFrame, use_icu: bool) -> pd.DataFrame:
    logger.info("[EXTRACTING MEDICATIONS DATA]")
    med = make_icu_input_events(cohort) if use_icu else make_hosp_prescriptions(cohort)

    cols = [h.value for h in MedicationsHeader] + [
        h.value for h in (IcuMedicationHeader if use_icu else NonIcuMedicationHeader)
    ]

    med = med[cols]
    return save_data(
        med, PREPROC_MED_ICU_PATH if use_icu else PREPROC_MED_PATH, "MEDICATIONS"
    )


def save_output_features(cohort: pd.DataFrame) -> pd.DataFrame:
    logger.info("[EXTRACTING OUPTPUT EVENTS DATA]")
    out = make_output_events(cohort)
    out = out[[h.value for h in OutputEventsHeader]]
    return save_data(out, PREPROC_OUT_ICU_PATH, "OUTPUT")


def save_chart_events_features(cohort: pd.DataFrame) -> pd.DataFrame:
    logger.info("[EXTRACTING CHART EVENTS DATA]")
    chart = make_chart_events(cohort)
    chart = drop_wrong_uom(chart, 0.95)
    chart = chart[[h.value for h in ChartEventsHeader]]
    return save_data(chart, PREPROC_CHART_ICU_PATH, "CHART")


def save_lab_events_features(cohort: pd.DataFrame) -> pd.DataFrame:
    logger.info("[EXTRACTING LABS DATA]")
    labevents = make_labs_events_features(cohort)
    labevents = drop_wrong_uom(labevents, 0.95)
    labevents = labevents[[h.value for h in LabEventsHeader]]
    return save_data(labevents, PREPROC_LABS_PATH, "LABS")
