import pandas as pd
import numpy as np
from pipeline.file_info.common import load_static_icd_map, IcdMap
from pipeline.file_info.raw.hosp import HospDiagnosesIcd, load_hosp_diagnosis_icd

ROOT_ICD_CONVERT = "root_icd10_convert"


def get_conversions_icd_9_10() -> dict:
    """Create mapping dictionary ICD9 -> ICD10"""
    icd_map_df = load_static_icd_map()
    # Filter rows where the length of diagnosis_code is 3
    filtered_df = icd_map_df[icd_map_df[IcdMap.DIAGNOISIS_CODE].str.len() == 3]

    # Drop duplicated diagnosis_code to keep only the first occurrence
    filtered_df = filtered_df.drop_duplicates(subset=IcdMap.DIAGNOISIS_CODE)

    return dict(zip(filtered_df[IcdMap.DIAGNOISIS_CODE], filtered_df[IcdMap.ICD10]))


def get_pos_ids(diag: pd.DataFrame, ICD10_code: str) -> pd.Series:
    """Extracts unique hospital admission IDs where 'root' contains a specific ICD-10 code."""
    return diag[diag[HospDiagnosesIcd.ROOT].str.contains(ICD10_code, na=False)][
        HospDiagnosesIcd.HOSPITAL_ADMISSION_ID
    ].unique()


def standardize_icd(df: pd.DataFrame):
    conversions_icd_9_10 = get_conversions_icd_9_10()
    # add the converted ICD10 column
    df[ROOT_ICD_CONVERT] = df.apply(
        lambda row: conversions_icd_9_10.get(row[HospDiagnosesIcd.ICD_CODE][:3], np.nan)
        if row[HospDiagnosesIcd.ICD_VERSION] == 9
        else row[HospDiagnosesIcd.ICD_CODE],
        axis=1,
    )
    # add the roots of the converted ICD10 column
    df[HospDiagnosesIcd.ROOT] = df[ROOT_ICD_CONVERT].apply(
        lambda x: x[:3] if type(x) is str else np.nan
    )

    return df


def preproc_icd_module() -> tuple:
    """Takes an module dataset with ICD codes and puts it in long_format,
    mapping ICD-codes by a mapping table path"""
    diag = load_hosp_diagnosis_icd()[
        [
            HospDiagnosesIcd.ICD_CODE,
            HospDiagnosesIcd.ICD_VERSION,
            HospDiagnosesIcd.HOSPITAL_ADMISSION_ID,
        ]
    ]
    diag = standardize_icd(diag)
    # Keep patient ids that have at least 1 record of the given ICD10 code category
    diag.dropna(subset=[HospDiagnosesIcd.ROOT], inplace=True)
    return diag
