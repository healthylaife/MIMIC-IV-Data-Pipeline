from enum import StrEnum
import pandas as pd
from pipeline.file_info.common import RAW_PATH

""" 
The Hosp module provides all data acquired from the hospital wide electronic health record
"""

HOSP = "hosp"

HOSP_DIAGNOSES_ICD_PATH = RAW_PATH / HOSP / "diagnoses_icd.csv.gz"
HOSP_PATIENTS_PATH = RAW_PATH / HOSP / "patients.csv.gz"
HOSP_LAB_EVENTS_PATH = RAW_PATH / HOSP / "labevents.csv.gz"
HOSP_ADMISSIONS_PATH = RAW_PATH / HOSP / "admissions.csv.gz"
HOSP_PREDICTIONS_PATH = RAW_PATH / HOSP / "prescriptions.csv.gz"
HOSP_PROCEDURES_ICD_PATH = RAW_PATH / HOSP / "procedures_icd.csv.gz"


# information regarding a patient
class HospPatients(StrEnum):
    ID = "subject_id"  # patient id
    ANCHOR_YEAR = "anchor_year"  # shifted year for the patient
    ANCHOR_AGE = "anchor_age"  # patient’s age in the anchor_year
    ANCHOR_YEAR_GROUP = "anchor_year_group"  # anchor_year occurred during this range
    DOD = "dod"  # de-identified date of death for the patient
    GENDER = "gender"


def load_hosp_patients() -> pd.DataFrame:
    return pd.read_csv(
        HOSP_PATIENTS_PATH,
        compression="gzip",
        parse_dates=[HospPatients.DOD],
    )


# information regarding a patient’s admission to the hospital
class HospAdmissions(StrEnum):
    ID = "hadm_id"  # hospitalization id
    PATIENT_ID = "subject_id"  # patient id
    ADMITTIME = "admittime"  # datetime the patient was admitted to the hospital
    DISCHTIME = "dischtime"  # datetime the patient was discharged from the hospital
    HOSPITAL_EXPIRE_FLAG = "hospital_expire_flag"  # whether the patient died within the given hospitalization
    LOS = "los"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    INSURANCE = "insurance"
    RACE = "race"


def load_hosp_admissions() -> pd.DataFrame:
    return pd.read_csv(
        HOSP_ADMISSIONS_PATH,
        compression="gzip",
        parse_dates=[HospAdmissions.ADMITTIME.value, HospAdmissions.DISCHTIME.value],
    )


class HospDiagnosesIcd(StrEnum):
    SUBJECT_ID = "subject_id"  # patient id
    HOSPITAL_ADMISSION_ID = "hadm_id"  # patient hospitalization id
    SEQ_NUM = "seq_num"  #  priority assigned to the diagnoses
    ICD_CODE = "icd_code"  #  International Coding Definitions code
    ICD_VERSION = "icd_version"  # version for the coding system
    # added
    ICD10 = "root_icd10_convert"
    ROOT = "root"


def load_hosp_diagnosis_icd() -> pd.DataFrame:
    return pd.read_csv(HOSP_DIAGNOSES_ICD_PATH, compression="gzip")


class HospLabEvents(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    CHART_TIME = "charttime"
    ITEM_ID = "itemid"
    ADMIT_TIME = "admittime"
    LAB_TIME_FROM_ADMIT = "lab_time_from_admit"
    VALUE_NUM = "valuenum"
    VALUE_UOM = "valueuom"


def load_hosp_lab_events(chunksize: int, use_cols=None) -> pd.DataFrame:
    return pd.read_csv(
        HOSP_LAB_EVENTS_PATH,
        compression="gzip",
        parse_dates=["charttime"],
        chunksize=chunksize,
        usecols=use_cols,
    )


class HospProceduresIcd(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    SEQ_NUM = "seq_num"
    CHART_DATE = "chartdate"
    ICD_CODE = "icd_code"
    ICD_VERSION = "icd_version"


def load_hosp_procedures_icd() -> pd.DataFrame:
    return pd.read_csv(
        HOSP_PROCEDURES_ICD_PATH,
        compression="gzip",
        parse_dates=[HospProceduresIcd.CHART_DATE.value],
    ).drop_duplicates()


class HospPrescriptions(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    DRUG = "drug"
    START_TIME = "starttime"
    STOP_TIME = "stoptime"
    NDC = "ndc"
    DOSE_VAL_RX = "dose_val_rx"


def load_hosp_prescriptions() -> pd.DataFrame:
    return pd.read_csv(
        HOSP_PREDICTIONS_PATH,
        compression="gzip",
        usecols=[
            HospPrescriptions.PATIENT_ID,
            HospPrescriptions.HOSPITAL_ADMISSION_ID,
            HospPrescriptions.DRUG,
            HospPrescriptions.START_TIME,
            HospPrescriptions.STOP_TIME,
            HospPrescriptions.NDC,
            HospPrescriptions.DOSE_VAL_RX,
        ],
        parse_dates=[HospPrescriptions.START_TIME, HospPrescriptions.STOP_TIME],
    )
