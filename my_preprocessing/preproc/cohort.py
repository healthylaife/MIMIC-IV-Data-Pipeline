from enum import StrEnum
from my_preprocessing.file_info import PREPROC_PATH

COHORT_PATH = PREPROC_PATH / "cohort"


# split common header icu header non icu header
class CohortHeader(StrEnum):
    PATIENT_ID = "subject_id"
    LABEL = "label"
    AGE = "age"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    INSURANCE = "insurance"
    ETHICITY = "ethnicity"
    STAY_ID = "stay_id"
    FIRST_CARE_UNIT = "first_careunit"
    LAST_CARE_UNIT = "last_careunit"
    IN_TIME = "intime"
    OUT_TIME = "outtime"
    LOS = "los"
    MIN_VALID_YEAR = "min_valid_year"
    DOD = "dod"
    GENDER = "gender"
    ADMIT_TIME = "admittime"
    DISCH_TIME = "dischtime"
