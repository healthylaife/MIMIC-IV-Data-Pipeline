from enum import StrEnum
from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = Path("raw_data") / "mimiciv_2_0"
MAP_PATH = Path("utils") / "mappings" / "ICD9_to_ICD10_mapping.txt"

HOSP_DIAGNOSES_ICD_PATH = RAW_PATH / "hosp" / "diagnoses_icd.csv.gz"


def load_icd_map() -> pd.DataFrame:
    return pd.read_csv(MAP_PATH, header=0, delimiter="\t")


def extract_dictionary(df) -> dict:
    """
    Extracts a dictionary from the given dataframe where keys are values from column 'a' with length 3,
    and the values are the first occurrence of corresponding values in column 'b'.

    :param dataframe: pandas DataFrame with columns 'a' and 'b'.
    :return: Dictionary with specified keys and values.
    """
    # Filter rows where the length of values in column 'a' is 3
    filtered_df = df[df["diagnosis_code"].str.len() == 3]

    # Drop duplicates in 'a' to keep only the first occurrence
    filtered_df = filtered_df.drop_duplicates(subset="diagnosis_code")

    filtered_df["icd10cm"] = filtered_df["icd10cm"].apply(
        lambda x: x[:3] if isinstance(x, str) else np.nan
    )

    # Convert the filtered dataframe to a dictionary
    result_dict = dict(zip(filtered_df["diagnosis_code"], filtered_df["icd10cm"]))

    return result_dict


def load_hosp_patients() -> pd.DataFrame:
    return pd.read_csv(
        RAW_PATH / "hosp" / "patients.csv.gz",
        compression="gzip",
        parse_dates=["dod"],
    )


def load_hosp_admissions() -> pd.DataFrame:
    return pd.read_csv(
        RAW_PATH / "hosp" / "admissions.csv.gz",
        compression="gzip",
        parse_dates=["admittime", "dischtime"],
    )


def load_icu_icustays() -> pd.DataFrame:
    return pd.read_csv(
        RAW_PATH / "icu" / "icustays.csv.gz",
        compression="gzip",
        parse_dates=["intime", "outtime"],
    )


def load_diagnosis_icd() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH / "hosp" / "diagnoses_icd.csv.gz", compression="gzip")


# icd mapping
class IcdMap(StrEnum):
    DIAGNOISIS_TYPE = "diagnosis_type"
    DIAGNOISIS_CODE = "diagnosis_code"
    DIAGNOISIS_DESCRIPTION = "diagnosis_description"
    ICD9CM = "icd9cm"
    ICD10CM = "icd10cm"
    FLAGS = "flags"


# The Hosp module provides all data acquired from the hospital wide electronic health record


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


class HospDiagnoses(StrEnum):
    SUBJECT_ID = "subject_id"
    HADM_ID = "hadm_id"
    SEQ_NUM = "seq_num"
    ICD_CODE = "icd_code"
    ICD_VERSION = "icd_version"
    # added columns
    ICD10 = "root_icd10_convert"


# The ICU module contains information collected from the clinical information system used within the ICU


# information regarding ICU stays
class IcuStays(StrEnum):
    PATIENT_ID = "subject_id"
    ID = "stay_id"
    HOSPITAL_AMISSION_ID = "hadm_id"
    ADMITTIME = "admittime"
    INTIME = "intime"
    OUTTIME = "outtime"
    LOS = "los"
