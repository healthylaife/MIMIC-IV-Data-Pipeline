import pandas as pd
import numpy as np
from my_preprocessing.raw_file_info import (
    HospPatients,
    IcuStays,
    HospAdmissions,
)
from my_preprocessing.preproc_file_info import CohortHeader
from my_preprocessing.prediction_task import TargetType
import my_preprocessing.icd_conversion as icd_conversion
from my_preprocessing.prediction_task import DiseaseCode
import logging

MIN_VALID_YEAR_HEADER = "min_valid_year"

logger = logging.getLogger()


def make_patients(hosp_patients: pd.DataFrame) -> pd.DataFrame:
    patients = hosp_patients[
        [
            HospPatients.ID,
            HospPatients.ANCHOR_YEAR_GROUP,
            HospPatients.ANCHOR_AGE,
            HospPatients.DOD,
            HospPatients.GENDER,
        ]
    ]
    max_anchor_year_group = (
        patients[HospPatients.ANCHOR_YEAR_GROUP].str.slice(start=-4).astype(int)
    )
    # To identify visits with prediction windows outside the range 2008-2019.
    patients[MIN_VALID_YEAR_HEADER] = (
        hosp_patients[HospPatients.ANCHOR_YEAR] + 2019 - max_anchor_year_group
    )
    return patients.rename(columns={HospPatients.ANCHOR_AGE: CohortHeader.AGE})[
        [
            HospPatients.ID,
            CohortHeader.AGE,
            MIN_VALID_YEAR_HEADER,
            HospPatients.DOD,
            HospPatients.GENDER,
        ]
    ]


def make_icu_visits(
    icu_icustays: pd.DataFrame, hosp_patients: pd.DataFrame, target_type: TargetType
) -> pd.DataFrame:
    if target_type != TargetType.READMISSION:
        return icu_icustays

    visits = icu_icustays
    # remove such stay_ids with a death for readmission labels
    patients_dod = hosp_patients[[HospPatients.ID, HospPatients.DOD]]
    visits = icu_icustays.merge(patients_dod, on=IcuStays.PATIENT_ID)
    visits = visits.loc[
        (visits[HospPatients.DOD].isna())
        | (visits[HospPatients.DOD] >= visits[IcuStays.OUTTIME])
    ]
    return visits[
        [
            CohortHeader.PATIENT_ID,
            "stay_id",
            "hadm_id",
            "intime",
            IcuStays.OUTTIME,
            IcuStays.LOS,
        ]
    ]


def make_no_icu_visits(
    hosp_admissions: pd.DataFrame, target_type: TargetType
) -> pd.DataFrame:
    hosp_admissions[HospAdmissions.LOS] = (
        hosp_admissions[HospAdmissions.DISCHTIME]
        - hosp_admissions[HospAdmissions.ADMITTIME]
    ).dt.days

    if target_type == TargetType.READMISSION:
        # remove hospitalizations with a death
        hosp_admissions = hosp_admissions[
            hosp_admissions[HospAdmissions.HOSPITAL_EXPIRE_FLAG] == 0
        ]
    return hosp_admissions[
        [
            HospAdmissions.PATIENT_ID,
            HospAdmissions.HOSPITAL_AMISSION_ID,
            HospAdmissions.ADMITTIME,
            HospAdmissions.DISCHTIME,
            HospAdmissions.LOS,
        ]
    ]


def filter_visits(
    visits,
    disease_readmission: DiseaseCode | None,
    disease_selection: DiseaseCode | None,
):
    diag = icd_conversion.preproc_icd_module()
    if disease_readmission:
        hids = icd_conversion.get_pos_ids(diag, disease_readmission)
        visits = visits[visits["hadm_id"].isin(hids["hadm_id"])]
        logger.info("[ READMISSION DUE TO " + disease_readmission + " ]")

    if disease_selection:
        hids = icd_conversion.get_pos_ids(diag, disease_selection)
        visits = visits[visits["hadm_id"].isin(hids["hadm_id"])]

    return visits


def partition_by_mort(
    df: pd.DataFrame,
    group_col: str,
    admit_col: str,
    discharge_col: str,
    death_col: str,
):
    """
    Partition data based on mortality events occurring between admission and discharge.

    Parameters:
    df (pd.DataFrame): The dataframe to partition.
    group_col (str): Column to group by.
    admit_col (str): Admission date column.
    discharge_col (str): Discharge date column.
    death_col (str): Death date column.
    """
    valid_entries = df.dropna(subset=[admit_col, discharge_col])
    valid_entries[death_col] = pd.to_datetime(valid_entries[death_col])
    valid_entries["label"] = np.where(
        (valid_entries[death_col] >= valid_entries[admit_col])
        & (valid_entries[death_col] <= valid_entries[discharge_col]),
        1,
        0,
    )
    sorted_cohort = valid_entries.sort_values(by=[group_col, admit_col])
    logger.info("[ MORTALITY LABELS FINISHED ]")
    return sorted_cohort


def partition_by_readmit(
    df: pd.DataFrame,
    gap: pd.Timedelta,
    group_col: str,
    admit_col: str,
    disch_col: str,
):
    """
    Partition data based on readmission within a specified gap.

    Parameters:
    df (pd.DataFrame): The dataframe to partition.
    gap (pd.Timedelta): Time gap to consider for readmission.
    group_col (str): Column to group by.
    admit_col (str): Admission date column.
    disch_col (str): Discharge date column.
    """

    df_sorted = df.sort_values(by=[group_col, admit_col])
    df_sorted["next_admit"] = df_sorted.groupby(group_col)[admit_col].shift(-1)
    df_sorted["time_to_next"] = df_sorted["next_admit"] - df_sorted[disch_col]
    # Identify readmission cases
    df_sorted["readmit"] = df_sorted["time_to_next"].notnull() & (
        df_sorted["time_to_next"] <= gap
    )
    temp_columns = ["next_admit", "time_to_next", "readmit"]
    case, ctrl = df_sorted[df_sorted["readmit"]], df_sorted[~df_sorted["readmit"]]
    case, ctrl = case.drop(columns=temp_columns), ctrl.drop(columns=temp_columns)
    case["label"], ctrl["label"] = np.ones(len(case)), np.zeros(len(ctrl))
    return pd.concat([case, ctrl], axis=0)


def partition_by_los(
    df: pd.DataFrame,
    los: int,
    group_col: str,
    admit_col: str,
    disch_col: str,
) -> pd.DataFrame:
    """
    Partition data based on length of stay (LOS).

    Parameters:
    df (pd.DataFrame): The dataframe to partition.
    los (int): Length of stay threshold.
    group_col (str): Column to group by.
    admit_col (str): Admission date column.
    disch_col (str): Discharge date column.
    """
    valid_cohort = df.dropna(subset=[admit_col, disch_col, "los"])
    valid_cohort["label"] = (valid_cohort["los"] > los).astype(int)
    return valid_cohort.sort_values(by=[group_col, admit_col])
