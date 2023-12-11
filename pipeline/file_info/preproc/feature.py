from enum import StrEnum

from pipeline.file_info.common import PREPROC_PATH


FEATURE_PATH = PREPROC_PATH / "features"
FEATURE_EXTRACT_PATH = FEATURE_PATH / "extract"
FEATURE_PREPROC_PATH = FEATURE_PATH / "preproc"
FEATURE_SUMMARY_PATH = FEATURE_PATH / "summary"


EXTRACT_DIAG_PATH = FEATURE_EXTRACT_PATH / "diag.csv.gz"
EXTRACT_DIAG_ICU_PATH = FEATURE_EXTRACT_PATH / "diag_icu.csv.gz"


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


EXTRACT_PROC_PATH = FEATURE_EXTRACT_PATH / "proc.csv.gz"
EXTRACT_PROC_ICU_PATH = FEATURE_EXTRACT_PATH / "proc_icu.csv.gz"
PREPROC_PROC_PATH = FEATURE_PREPROC_PATH / "proc.csv.gz"
PREPROC_PROC_ICU_PATH = FEATURE_PREPROC_PATH / "proc_icu.csv.gz"


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


EXTRACT_MED_ICU_PATH = FEATURE_EXTRACT_PATH / "med_icu.csv.gz"
EXTRACT_MED_PATH = FEATURE_EXTRACT_PATH / "med.csv.gz"
PREPROC_MED_ICU_PATH = FEATURE_PREPROC_PATH / "med_icu.csv.gz"
PREPROC_MED_PATH = FEATURE_PREPROC_PATH / "med.csv.gz"


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


EXTRACT_OUT_ICU_PATH = FEATURE_EXTRACT_PATH / "out_icu.csv.gz"
PREPROC_OUT_ICU_PATH = FEATURE_PREPROC_PATH / "out_icu.csv.gz"


class OutputEventsHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    CHART_TIME = "charttime"
    IN_TIME = "intime"
    EVENT_TIME_FROM_ADMIT = "event_time_from_admit"


EXTRACT_LABS_PATH = FEATURE_EXTRACT_PATH / "labs.csv.gz"
PREPROC_LABS_ICU_PATH = FEATURE_PREPROC_PATH / "labs.csv.gz"


class LabEventsHeader(StrEnum):
    PATIENT_ID = "subject_id"
    HOSPITAL_ADMISSION_ID = "hadm_id"
    ITEM_ID = "itemid"
    CHART_TIME = "charttime"
    ADMIT_TIME = "admittime"
    LAB_TIME_FROM_ADMIT = "lab_time_from_admit"
    VALUE_NUM = "valuenum"


EXTRACT_CHART_ICU_PATH = FEATURE_EXTRACT_PATH / "chart_icu.csv.gz"
PREPROC_CHART_ICU_PATH = FEATURE_PREPROC_PATH / "chart_icu.csv.gz"


class ChartEventsHeader(StrEnum):
    STAY_ID = "stay_id"
    ITEM_ID = "itemid"
    VALUE_NUM = "valuenum"
    EVENT_TIME_FROM_ADMIT = "event_time_from_admit"
