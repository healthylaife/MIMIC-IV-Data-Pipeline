from enum import StrEnum
from pathlib import Path
import pandas as pd

RAW_PATH = Path("raw_data") / "mimiciv_2_0"
MAP_PATH = Path("utils") / "mappings" / "ICD9_to_ICD10_mapping.txt"
MAP_NDC_PATH = Path("utils") / "mappings" / "ndc_product.txt"

HOSP_DIAGNOSES_ICD_PATH = RAW_PATH / "hosp" / "diagnoses_icd.csv.gz"
HOSP_PATIENTS_PATH = RAW_PATH / "hosp" / "patients.csv.gz"
HOSP_LAB_EVENTS_PATH = RAW_PATH / "hosp" / "labevents.csv.gz"
HOSP_ADMISSIONS_PATH = RAW_PATH / "hosp" / "admissions.csv.gz"
HOSP_PREDICTIONS_PATH = RAW_PATH / "hosp" / "prescriptions.csv.gz"
HOSP_PROCEDURES_ICD_PATH = RAW_PATH / "hosp" / "procedures_icd.csv.gz"


ICU_ICUSTAY_PATH = RAW_PATH / "icu" / "icustays.csv.gz"
ICU_INPUT_EVENT_PATH = RAW_PATH / "icu" / "inputevents.csv.gz"
ICU_OUTPUT_EVENT_PATH = RAW_PATH / "icu" / "outputevents.csv.gz"
ICU_CHART_EVENTS_PATH = RAW_PATH / "icu" / "chartevents.csv.gz"
ICU_PROCEDURE_EVENTS_PATH = RAW_PATH / "icu" / "procedureevents.csv.gz"

PREPROC_PATH = Path("preproc_data")
COHORT_PATH = PREPROC_PATH / "cohort"
FEATURE_PATH = PREPROC_PATH / "features"


# icd mapping
class IcdMap(StrEnum):
    DIAGNOISIS_TYPE = "diagnosis_type"
    DIAGNOISIS_CODE = "diagnosis_code"
    DIAGNOISIS_DESCRIPTION = "diagnosis_description"
    ICD9CM = "icd9cm"
    ICD10CM = "icd10cm"
    FLAGS = "flags"


def load_static_icd_map() -> pd.DataFrame:
    return pd.read_csv(MAP_PATH, delimiter="\t")


# The Hosp module provides all data acquired from the hospital wide electronic health record


# information regarding a patient
class HospPatients(StrEnum):
    ID = "subject_id"  # patient id
    YEAR = "anchor_year"  # shifted year for the patient
    AGE = "anchor_age"  # patient’s age in the anchor_year
    YEAR_GROUP = "anchor_year_group"  # anchor_year occurred during this range
    DOD = "dod"  # de-identified date of death for the patient
    GENDER = "gender"


# information regarding a patient’s admission to the hospital
class HospAdmissions(StrEnum):
    ID = "hadm_id"  # hospitalization id
    PATIENT_ID = "subject_id"  # patient id
    ADMITTIME = "admittime"  # datetime the patient was admitted to the hospital
    DISCHTIME = "dischtime"  # datetime the patient was discharged from the hospital
    HOSPITAL_EXPIRE_FLAG = "hospital_expire_flag"  # whether the patient died within the given hospitalization
    LOS = "los"


class HospDiagnosesIcd(StrEnum):
    SUBJECT_ID = "subject_id"  # patient id
    HOSPITAL_AMISSION_ID = "hadm_id"  # patient hospitalization id
    SEQ_NUM = "seq_num"  #  priority assigned to the diagnoses
    ICD_CODE = "icd_code"  #  International Coding Definitions code
    ICD_VERSION = "icd_version"  # version for the coding system
    # added
    ICD10 = "root_icd10_convert"
    ROOT = "root"


class HospLabEvents(StrEnum):
    SUNJECT_ID = "subject_id"
    HOSPITAL_AMISSION_ID = "hadm_id"
    CHARTTIME = "charttime"
    ITEMID = "itemid"
    ADMITTIME = "admittime"
    LAB_TIME_FROM_ADMIT = "lab_time_from_admit"
    VALUENUM = "valuenum"


class HospProceduresIcd(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_AMISSION_ID = "hadm_id"
    SEQ_NUM = "seq_num"
    CHART_DATE = "chartdate"
    ICD_CODE = "icd_code"
    ICD_VERSION = "icd_version"


def load_hosp_patients() -> pd.DataFrame:
    return pd.read_csv(
        HOSP_PATIENTS_PATH,
        compression="gzip",
        parse_dates=[HospPatients.DOD.value],
    )


def load_hosp_admissions() -> pd.DataFrame:
    return pd.read_csv(
        HOSP_ADMISSIONS_PATH,
        compression="gzip",
        parse_dates=[HospAdmissions.ADMITTIME.value, HospAdmissions.DISCHTIME.value],
    )


def load_hosp_diagnosis_icd() -> pd.DataFrame:
    return pd.read_csv(HOSP_DIAGNOSES_ICD_PATH, compression="gzip")


def load_hosp_lab_events(chunksize, use_cols=None) -> pd.DataFrame:
    return pd.read_csv(
        HOSP_LAB_EVENTS_PATH,
        compression="gzip",
        parse_dates=["charttime"],
        chunksize=chunksize,
        usecols=use_cols,
    )


def load_hosp_procedures_icd() -> pd.DataFrame:
    return pd.read_csv(
        HOSP_PROCEDURES_ICD_PATH,
        compression="gzip",
        parse_dates=[HospProceduresIcd.CHART_DATE.value],
    ).drop_duplicates()


# The ICU module contains information collected from the clinical information system used within the ICU.


# information regarding ICU stays
class IcuStays(StrEnum):
    PATIENT_ID = "subject_id"  # patient id
    ID = "stay_id"  # icu stay id
    HOSPITAL_AMISSION_ID = "hadm_id"  # patient hospitalization id
    INTIME = "intime"  #  datetime the patient was transferred into the ICU.
    OUTTIME = "outtime"  #  datetime the patient was transferred out the ICU.
    LOS = "los"  # length of stay for the patient for the given ICU stay in fractional days.
    # added?
    ADMITTIME = "admittime"


# Information regarding patient outputs including urine, drainage...
class OuputputEvents(StrEnum):
    SUBJECT_ID = "subject_id"  # patient id
    HOSPITAL_AMISSION_ID = "hadm_id"  # patient hospitalization id
    ICU_STAY_ID = "stay_id"  # patient icu stay id
    ITEM_ID = "itemid"  # single measurement type id
    CHARTTIME = "charttime"  # time of an output event


class ChartEvents(StrEnum):
    STAY_ID = "stay_id"
    CHARTTIME = "charttime"
    ITEMID = "itemid"
    VALUENUM = "valuenum"
    VALUEOM = "valueuom"


class InputEvents(StrEnum):
    SUBJECT_ID = "subject_id"
    STAY_ID = "stay_id"
    ITEMID = "itemid"
    STARTTIME = "starttime"
    ENDTIME = "endtime"
    RATE = "rate"
    AMOUNT = "amount"
    ORDERID = "orderid"


def load_icu_icustays() -> pd.DataFrame:
    return pd.read_csv(
        ICU_ICUSTAY_PATH,
        compression="gzip",
        parse_dates=[IcuStays.INTIME.value, IcuStays.OUTTIME.value],
    )


def load_icu_outputevents() -> pd.DataFrame:
    return pd.read_csv(
        ICU_OUTPUT_EVENT_PATH,
        compression="gzip",
        parse_dates=[OuputputEvents.CHARTTIME.value],
    ).drop_duplicates()
