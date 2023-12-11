from enum import StrEnum
import pandas as pd
from pipeline.file_info.common import RAW_PATH

"""
The ICU module contains information collected from the clinical information system used within the ICU.

"""
ICU = "icu"

ICU_ICUSTAY_PATH = RAW_PATH / ICU / "icustays.csv.gz"
ICU_INPUT_EVENT_PATH = RAW_PATH / ICU / "inputevents.csv.gz"
ICU_OUTPUT_EVENT_PATH = RAW_PATH / ICU / "outputevents.csv.gz"
ICU_CHART_EVENTS_PATH = RAW_PATH / ICU / "chartevents.csv.gz"
ICU_PROCEDURE_EVENTS_PATH = RAW_PATH / ICU / "procedureevents.csv.gz"


# information regarding ICU stays
class IcuStays(StrEnum):
    PATIENT_ID = "subject_id"  # patient id
    ID = "stay_id"  # icu stay id
    HOSPITAL_ADMISSION_ID = "hadm_id"  # patient hospitalization id
    INTIME = "intime"  #  datetime the patient was transferred into the ICU.
    OUTTIME = "outtime"  #  datetime the patient was transferred out the ICU.
    LOS = "los"  # length of stay for the patient for the given ICU stay in fractional days.
    # added?
    ADMITTIME = "admittime"


def load_icu_icustays() -> pd.DataFrame:
    return pd.read_csv(
        ICU_ICUSTAY_PATH,
        compression="gzip",
        parse_dates=[IcuStays.INTIME, IcuStays.OUTTIME],
    )


# Information regarding patient outputs including urine, drainage...
class OuputputEvents(StrEnum):
    SUBJECT_ID = "subject_id"  # patient id
    HOSPITAL_ADMISSION_ID = "hadm_id"  # patient hospitalization id
    STAY_ID = "stay_id"  # patient icu stay id
    ITEM_ID = "itemid"  # single measurement type id
    CHART_TIME = "charttime"  # time of an output event


def load_icu_output_events() -> pd.DataFrame:
    return pd.read_csv(
        ICU_OUTPUT_EVENT_PATH,
        compression="gzip",
        parse_dates=[OuputputEvents.CHART_TIME],
    ).drop_duplicates()


class ChartEvents(StrEnum):
    STAY_ID = "stay_id"
    CHARTTIME = "charttime"
    ITEMID = "itemid"
    VALUENUM = "valuenum"
    VALUEOM = "valueuom"


def load_icu_chart_events(chunksize: int) -> pd.DataFrame:
    return pd.read_csv(
        ICU_CHART_EVENTS_PATH,
        compression="gzip",
        usecols=[c for c in ChartEvents],
        parse_dates=[ChartEvents.CHARTTIME],
        chunksize=chunksize,
    )


def load_icu_chart_events(chunksize: int) -> pd.DataFrame:
    return pd.read_csv(
        ICU_CHART_EVENTS_PATH,
        compression="gzip",
        usecols=[c for c in ChartEvents],
        parse_dates=[ChartEvents.CHARTTIME.value],
        chunksize=chunksize,
    )


class InputEvents(StrEnum):
    SUBJECT_ID = "subject_id"
    STAY_ID = "stay_id"
    ITEMID = "itemid"
    STARTTIME = "starttime"
    ENDTIME = "endtime"
    RATE = "rate"
    AMOUNT = "amount"
    ORDERID = "orderid"


def load_input_events() -> pd.DataFrame:
    return pd.read_csv(
        ICU_INPUT_EVENT_PATH,
        compression="gzip",
        usecols=[f for f in InputEvents],
        parse_dates=[InputEvents.STARTTIME, InputEvents.ENDTIME],
    )


class ProceduresEvents(StrEnum):
    STAY_ID = "stay_id"
    START_TIME = "starttime"
    ITEM_ID = "itemid"


def load_icu_procedure_events() -> pd.DataFrame:
    return pd.read_csv(
        ICU_PROCEDURE_EVENTS_PATH,
        compression="gzip",
        usecols=[h for h in ProceduresEvents],
        parse_dates=[ProceduresEvents.START_TIME],
    ).drop_duplicates()
