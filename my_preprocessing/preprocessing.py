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
from typing import Optional
import logging


logger = logging.getLogger()


def make_patients(hosp_patients: pd.DataFrame) -> pd.DataFrame:
    patients = hosp_patients[
        [
            HospPatients.ID,
            HospPatients.ANCHOR_YEAR,
            HospPatients.ANCHOR_YEAR_GROUP,
            HospPatients.ANCHOR_AGE,
            HospPatients.DOD,
            HospPatients.GENDER,
        ]
    ].copy()
    max_anchor_year_group = (
        patients[HospPatients.ANCHOR_YEAR_GROUP].str.slice(start=-4).astype(int)
    )
    # To identify visits with prediction windows outside the range 2008-2019.
    patients[CohortHeader.MIN_VALID_YEAR] = (
        hosp_patients[HospPatients.ANCHOR_YEAR] + 2019 - max_anchor_year_group
    )
    return patients.rename(columns={HospPatients.ANCHOR_AGE: CohortHeader.AGE})[
        [
            HospPatients.ID,
            CohortHeader.AGE,
            CohortHeader.MIN_VALID_YEAR,
            HospPatients.DOD,
            HospPatients.GENDER,
        ]
    ]


def make_icu_visits(
    icu_icustays: pd.DataFrame, hosp_patients: pd.DataFrame, target_type: TargetType
) -> pd.DataFrame:
    if target_type != TargetType.READMISSION:
        return icu_icustays
    # Filter out stays where either there is no death or the death occurred after ICU discharge
    patients_dod = hosp_patients[[HospPatients.ID, HospPatients.DOD]]
    visits = icu_icustays.merge(patients_dod, on=IcuStays.PATIENT_ID)
    filtered_visits = visits.loc[
        (visits[HospPatients.DOD].isna())
        | (visits[HospPatients.DOD] >= visits[IcuStays.OUTTIME])
    ]
    return filtered_visits[
        [
            CohortHeader.PATIENT_ID,
            CohortHeader.STAY_ID,
            CohortHeader.HOSPITAL_ADMISSION_ID,
            CohortHeader.IN_TIME,
            CohortHeader.OUT_TIME,
            CohortHeader.LOS,
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
        # Filter out hospitalizations where the patient expired
        hosp_admissions = hosp_admissions[
            hosp_admissions[HospAdmissions.HOSPITAL_EXPIRE_FLAG] == 0
        ]
    return hosp_admissions[
        [
            CohortHeader.PATIENT_ID,
            CohortHeader.HOSPITAL_ADMISSION_ID,
            CohortHeader.ADMIT_TIME,
            CohortHeader.DISCH_TIME,
            CohortHeader.LOS,
        ]
    ]


def filter_visits(
    visits,
    disease_readmission: Optional[DiseaseCode],
    disease_selection: Optional[DiseaseCode],
) -> pd.DataFrame:
    """# Filter visits based on readmission due to a specific disease and on disease selection"""
    diag = icd_conversion.preproc_icd_module()
    if disease_readmission:
        hids = icd_conversion.get_pos_ids(diag, disease_readmission)
        visits = visits[visits[CohortHeader.HOSPITAL_ADMISSION_ID].isin(hids)]
        logger.info(f"[ READMISSION DUE TO {disease_readmission} ]")

    if disease_selection:
        hids = icd_conversion.get_pos_ids(diag, disease_selection)
        visits = visits[visits[CohortHeader.HOSPITAL_ADMISSION_ID].isin(hids)]

    return visits


def partition_by_mort(
    df: pd.DataFrame,
    group_col: str,
    admit_col: str,
    discharge_col: str,
    death_col: str,
) -> pd.DataFrame:
    """
    Partition data based on mortality events occurring between admission and discharge.

    Parameters:
    df (pd.DataFrame): The dataframe to partition.
    group_col (str): Column to group by.
    admit_col (str): Admission date column.
    discharge_col (str): Discharge date column.
    death_col (str): Death date column.
    """
    df = df.dropna(subset=[admit_col, discharge_col])
    df[death_col] = pd.to_datetime(df[death_col])
    df[CohortHeader.LABEL] = np.where(
        (df[death_col] >= df[admit_col]) & (df[death_col] <= df[discharge_col]),
        1,
        0,
    )
    logger.info(
        f"[ MORTALITY LABELS FINISHED: {df[CohortHeader.LABEL].sum()} Mortality Cases ]"
    )
    return df.sort_values(by=[group_col, admit_col])


def partition_by_readmit(
    df: pd.DataFrame,
    gap: pd.Timedelta,
    group_col: str,
    admit_col: str,
    disch_col: str,
) -> pd.DataFrame:
    """
    Partition data based on readmission within a specified gap.
    """
    df["next_admit"] = (
        df.sort_values(by=[admit_col]).groupby(group_col)[admit_col].shift(-1)
    )
    df["time_to_next"] = df["next_admit"] - df[disch_col]

    df[CohortHeader.LABEL] = (
        df["time_to_next"].notnull() & (df["time_to_next"] <= gap)
    ).astype(int)

    readmit_cases = df[CohortHeader.LABEL].sum()
    logger.info(f"[ READMISSION LABELS FINISHED: {readmit_cases} Readmission Cases ]")
    return df.drop(columns=["next_admit", "time_to_next"]).sort_values(
        by=[group_col, admit_col]
    )


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
    df = df.dropna(subset=[admit_col, disch_col, CohortHeader.LOS])
    df[CohortHeader.LABEL] = (df[CohortHeader.LOS] > los).astype(int)
    logger.info(f"[ LOS LABELS FINISHED: {df[CohortHeader.LABEL].sum()} LOS Cases ]")
    return df.sort_values(by=[group_col, admit_col])
