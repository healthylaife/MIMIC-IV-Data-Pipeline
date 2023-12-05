from enum import StrEnum
from pathlib import Path

from my_preprocessing.file_info import PREPROC_PATH


FEATURE_PATH = PREPROC_PATH / "features"

PREPROC_DIAG_PATH = FEATURE_PATH / "preproc_diag.csv.gz"
PREPROC_DIAG_ICU_PATH = FEATURE_PATH / "preproc_diag_icu.csv.gz"


class DiagnosesHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    ICD_CODE = "icd_code"
    ROOT_ICD10 = "root_icd10_convert"
    ROOT = "root"


class DiagnosesIcuHeader(StrEnum):
    STAY_ID = "stay_id"


class PreprocDiagnosesHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    NEW_ICD_CODE = "new_icd_code"


PREPROC_PROC_PATH = FEATURE_PATH / "preproc_proc.csv.gz"
PREPROC_PROC_ICU_PATH = FEATURE_PATH / "preproc_proc_icu.csv.gz"


class ProceduresHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"


class IcuProceduresHeader(StrEnum):
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    START_TIME = "starttime"
    IN_TIME = "intime"
    EVENT_TIME_FROM_ADMIT = "event_time_from_admit"


class NonIcuProceduresHeader(StrEnum):
    ICD_CODE = "icd_code"
    ICD_VERSION = "icd_version"
    CHART_DATE = "chartdate"
    ADMIT_TIME = "admittime"
    PROC_TIME_FROM_ADMIT = "proc_time_from_admit"


PREPROC_MED_ICU_PATH = FEATURE_PATH / "preproc_med_icu.csv.gz"
PREPROC_MED_PATH = FEATURE_PATH / "preproc_med.csv.gz"


class MedicationsHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    START_TIME = "starttime"
    START_HOURS_FROM_ADMIT = "start_hours_from_admit"
    STOP_HOURS_FROM_ADMIT = "stop_hours_from_admit"


class IcuMedicationHeader(StrEnum):
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    END_TIME = "endtime"
    RATE = "rate"
    AMOUNT = "amount"
    ORDER_ID = "orderid"


class NonIcuMedicationHeader(StrEnum):
    STOP_TIME = "stoptime"
    DRUG = "drug"
    NON_PROPRIEATARY_NAME = "nonproprietaryname"
    DOSE_VAL_RX = "dose_val_rx"
    EPC = "EPC"


class PreprocMedicationHeader(StrEnum):
    DRUG_NAME = "drug_name"


PREPROC_OUT_ICU_PATH = FEATURE_PATH / "preproc_out_icu.csv.gz"


class OutputEventsHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    CHART_TIME = "charttime"
    IN_TIME = "intime"
    EVENT_TIME_FROM_ADMIT = "event_time_from_admit"


PREPROC_LABS_PATH = FEATURE_PATH / "preproc_labs.csv.gz"


class LabEventsHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    ITEM_ID = "itemid"
    CHART_TIME = "charttime"
    ADMIT_TIME = "admittime"
    LAB_TIME_FROM_ADMIT = "lab_time_from_admit"
    VALUE_NUM = "valuenum"


PREPROC_CHART_ICU_PATH = FEATURE_PATH / "preproc_chart_icu.csv.gz"


class ChartEventsHeader(StrEnum):
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    VALUE_NUM = "valuenum"
    EVENT_TIME_FROM_ADMIT = "event_time_from_admit"
