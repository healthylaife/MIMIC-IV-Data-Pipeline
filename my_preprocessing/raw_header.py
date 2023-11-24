from enum import StrEnum
from pathlib import Path
import pandas as pd

# ADD RAW PATH?

# The Hosp module provides all data acquired from the hospital wide electronic health record

RAW_PATH = Path("raw_data") / "mimiciv_2_0"
MAP_PATH = Path("utils") / "mappings" / "ICD9_to_ICD10_mapping.txt"


def load_hosp_patients():
    patients = pd.read_csv(
        RAW_PATH / "hosp" / "patients.csv.gz",
        compression="gzip",
        parse_dates=["dod"],
    )
    return patients


def load_hosp_admissions():
    hosp_admissions = pd.read_csv(
        RAW_PATH / "hosp" / "admissions.csv.gz",
        compression="gzip",
        parse_dates=["admittime", "dischtime"],
    )
    return hosp_admissions


def load_icu_icustays():
    visits = pd.read_csv(
        RAW_PATH / "icu" / "icustays.csv.gz",
        compression="gzip",
        parse_dates=["intime", "outtime"],
    )
    return visits


# information regarding a patient
class HospPatients(StrEnum):
    ID = "subject_id"
    YEAR = "anchor_year"  # shifted year for the patient
    AGE = "anchor_age"  # patient’s age in the anchor_year
    YEAR_GROUP = "anchor_year_group"  # anchor_year occurred during this range
    DOD = "dod"  # de-identified date of death for the patient
    GENDER = "gender"


# information regarding a patient’s admission to the hospital
class HospAdmissions(StrEnum):
    ID = "hadm_id"
    PATIENT_ID = "subject_id"
    ADMITTIME = "admittime"  # datetime the patient was admitted to the hospital
    DISCHTIME = "dischtime"  # datetime the patient was discharged from the hospital
    HOSPITAL_EXPIRE_FLAG = "hospital_expire_flag"  # whether the patient died within the given hospitalization
    LOS = "los"


# The ICU module contains information collected from the clinical information system used within the ICU


# information regarding ICU stays
class IcuStays(StrEnum):
    PATIENT_ID = "subject_id"
    ID = "stay_id"
    HOSPITAL_AMISSION_ID = "hadm_id"
    "admittime"
    "intime"
    "outtime"
    "los"
