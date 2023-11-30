import pandas as pd
import numpy as np
from my_preprocessing.raw_file_info import (
    load_static_icd_map,
    load_hosp_diagnosis_icd,
)


def get_conversions_icd_9_10() -> dict:
    """Create mapping dictionary ICD9 -> ICD10"""
    icd_map_df = load_static_icd_map()
    # Filter rows where the length of diagnosis_code is 3
    filtered_df = icd_map_df[icd_map_df["diagnosis_code"].str.len() == 3]

    # Drop duplicated diagnosis_code to keep only the first occurrence
    filtered_df = filtered_df.drop_duplicates(subset="diagnosis_code")

    return dict(zip(filtered_df["diagnosis_code"], filtered_df["icd10cm"]))


def get_pos_ids(diag: pd.DataFrame, ICD10_code: str) -> pd.Series:
    """Extracts unique hospital admission IDs (hadm_id) where 'root' contains a specific ICD-10 code."""
    return diag[diag["root"].str.contains(ICD10_code, na=False)]["hadm_id"].unique()


def standardize_icd(df: pd.DataFrame):
    conversions_icd_9_10 = get_conversions_icd_9_10()
    # add the converted ICD10 column
    df["root_icd10_convert"] = df.apply(
        lambda row: conversions_icd_9_10.get(row["icd_code"][:3], np.nan)
        if row["icd_version"] == 9
        else row["icd_code"],
        axis=1,
    )
    # add the roots of the converted ICD10 column
    df["root"] = df["root_icd10_convert"].apply(
        lambda x: x[:3] if type(x) is str else np.nan
    )

    return df


def preproc_icd_module() -> tuple:
    """Takes an module dataset with ICD codes and puts it in long_format,
    mapping ICD-codes by a mapping table path"""
    diag = load_hosp_diagnosis_icd()[["icd_code", "icd_version", "hadm_id"]]
    diag = standardize_icd(diag)
    # Keep patient ids that have at least 1 record of the given ICD10 code category
    diag.dropna(subset=["root"], inplace=True)
    return diag
