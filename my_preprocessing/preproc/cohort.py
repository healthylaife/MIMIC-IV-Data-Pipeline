from enum import StrEnum
from my_preprocessing.file_info import PREPROC_PATH
import pandas as pd
import logging

logger = logging.getLogger()
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


def load_cohort(use_icu: bool, cohort_ouput: str) -> pd.DataFrame:
    """Load cohort data from a CSV file."""
    cohort_path = COHORT_PATH / f"{cohort_ouput}.csv.gz"
    try:
        return pd.read_csv(
            cohort_path,
            compression="gzip",
            parse_dates=[CohortHeader.IN_TIME if use_icu else CohortHeader.ADMIT_TIME],
        )
    except FileNotFoundError:
        logger.error(f"Cohort file not found at {cohort_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading cohort file: {e}")
        raise
