from enum import StrEnum
from pathlib import Path
import pandas as pd
from tqdm import tqdm

RAW_PATH = Path("raw_data") / "mimiciv_2_0"
MAP_PATH = Path("utils") / "mappings" / "ICD9_to_ICD10_mapping.txt"

HOSP_DIAGNOSES_ICD_PATH = RAW_PATH / "hosp" / "diagnoses_icd.csv.gz"
HOSP_PATIENTS_PATH = RAW_PATH / "hosp" / "patients.csv.gz"


ICU_ICUSTAY_PATH = RAW_PATH / "icu" / "icustays.csv.gz"
ICU_INPUT_EVENT_PATH = RAW_PATH / "icu" / "inputevents.csv.gz"
ICU_OUTPUT_EVENT_PATH = RAW_PATH / "icu" / "outputevents.csv.gz"
HOSP_ADMISSIONS_PATH = RAW_PATH / "hosp" / "admissions.csv.gz"
ICU_CHART_EVENTS_PATH = RAW_PATH / "icu" / "chartevents.csv.gz"
ICU_PROCEDURE_EVENTS_PATH = RAW_PATH / "icu" / "procedureevents.csv.gz"

PREPROC_PATH = Path("preproc_data")
COHORT_PATH = PREPROC_PATH / "cohort"


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


def preproc_chartevents(cohort_path: str, chunksize=10000000) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    # Only consider values in our cohort
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["intime"])
    df_cohort = pd.DataFrame()
    # read module w/ custom params
    for chunk in tqdm(
        pd.read_csv(
            ICU_CHART_EVENTS_PATH,
            compression="gzip",
            usecols=[
                ChartEvents.STAY_ID.value,
                ChartEvents.CHARTTIME.value,
                ChartEvents.ITEMID.value,
                ChartEvents.VALUENUM.value,
                ChartEvents.VALUEOM.value,
            ],
            parse_dates=[ChartEvents.CHARTTIME.value],
            chunksize=chunksize,
        )
    ):
        chunk = chunk.dropna(subset=["valuenum"])
        chunk_merged = chunk.merge(
            cohort[["stay_id", "intime"]],
            how="inner",
            left_on="stay_id",
            right_on="stay_id",
        )
        chunk_merged["event_time_from_admit"] = (
            chunk_merged["charttime"] - chunk_merged["intime"]
        )

        del chunk_merged["charttime"]
        del chunk_merged["intime"]
        chunk_merged = chunk_merged.dropna()
        chunk_merged = chunk_merged.drop_duplicates()
        if df_cohort.empty:
            df_cohort = chunk_merged
        else:
            df_cohort = df_cohort.append(chunk_merged, ignore_index=True)

    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    return df_cohort


def preproc_output_events(cohort_path: str) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort.
    Function is structured to save memory when reading and transforming data."""
    outputevents = load_icu_outputevents()
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["intime"])
    df_cohort = outputevents.merge(
        cohort[["stay_id", "intime", "outtime"]],
        how="inner",
        left_on="stay_id",
        right_on="stay_id",
    )
    df_cohort["event_time_from_admit"] = df_cohort["charttime"] - df_cohort["intime"]
    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort


def preproc_procedure_events(cohort_path: str) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    module = pd.read_csv(
        ICU_PROCEDURE_EVENTS_PATH,
        compression="gzip",
        usecols=["stay_id", "starttime", "itemid"],
        parse_dates=["starttime"],
    ).drop_duplicates()
    # Only consider values in our cohort
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["intime"])
    df_cohort = module.merge(
        cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
        how="inner",
        left_on="stay_id",
        right_on="stay_id",
    )
    df_cohort["event_time_from_admit"] = df_cohort["starttime"] - df_cohort["intime"]

    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.dropna().nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    return df_cohort


def preprocess_input_events(cohort_path: str) -> pd.DataFrame:
    adm = pd.read_csv(
        cohort_path,
        usecols=["hadm_id", "stay_id", "intime"],
        parse_dates=["intime"],
    )
    med = pd.read_csv(
        ICU_INPUT_EVENT_PATH,
        compression="gzip",
        usecols=[
            "subject_id",
            "stay_id",
            "itemid",
            "starttime",
            "endtime",
            "rate",
            "amount",
            "orderid",
        ],
        parse_dates=["starttime", "endtime"],
    )
    med = med.merge(adm, left_on="stay_id", right_on="stay_id", how="inner")
    med["start_hours_from_admit"] = med["starttime"] - med["intime"]
    med["stop_hours_from_admit"] = med["endtime"] - med["intime"]
    med = med.dropna()
    print("# of unique type of drug: ", med.itemid.nunique())
    print("# Admissions:  ", med.stay_id.nunique())
    print("# Total rows", med.shape[0])

    return med

    return
