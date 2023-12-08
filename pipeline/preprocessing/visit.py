import pandas as pd
from pipeline.conversion.icd import IcdConverter
from pipeline.file_info.raw.hosp import (
    HospPatients,
    HospAdmissions,
)
from pipeline.file_info.raw.icu import IcuStays

from pipeline.file_info.preproc.cohort import (
    CohortHeader,
    IcuCohortHeader,
    NonIcuCohortHeader,
)
from pipeline.prediction_task import TargetType
from pipeline.prediction_task import DiseaseCode
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
            IcuCohortHeader.STAY_ID,
            CohortHeader.HOSPITAL_ADMISSION_ID,
            IcuCohortHeader.IN_TIME,
            IcuCohortHeader.OUT_TIME,
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
            NonIcuCohortHeader.ADMIT_TIME,
            NonIcuCohortHeader.DISCH_TIME,
            CohortHeader.LOS,
        ]
    ]


def filter_visits(
    visits,
    disease_readmission: Optional[DiseaseCode],
    disease_selection: Optional[DiseaseCode],
) -> pd.DataFrame:
    """# Filter visits based on readmission due to a specific disease and on disease selection"""
    icd_converter = IcdConverter()
    diag = icd_converter.preproc_icd_module()
    if disease_readmission:
        hids = icd_converter.get_pos_ids(diag, disease_readmission)
        visits = visits[visits[CohortHeader.HOSPITAL_ADMISSION_ID].isin(hids)]
        logger.info(f"[ READMISSION DUE TO {disease_readmission} ]")

    if disease_selection:
        hids = icd_converter.get_pos_ids(diag, disease_selection)
        visits = visits[visits[CohortHeader.HOSPITAL_ADMISSION_ID].isin(hids)]

    return visits
