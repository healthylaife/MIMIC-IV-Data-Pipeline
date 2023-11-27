import pandas as pd
import numpy as np
from my_preprocessing.raw_files import (
    load_static_icd_map,
    load_hosp_diagnosis_icd,
    extract_dictionary,
)


def preproc_icd_module(ICD10_code: str) -> tuple:
    """Takes an module dataset with ICD codes and puts it in long_format,
    mapping ICD-codes by a mapping table path"""
    diag = load_hosp_diagnosis_icd()[["icd_code", "icd_version", "hadm_id"]]
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe;
    adds column with converted ICD10 column"""

    # Create mapping dictionary  ICD9 -> ICD10
    mapping = load_static_icd_map()
    mapping_dico = extract_dictionary(mapping)
    # convert all ICD9 codes to ICD10
    diag["root"] = diag.apply(
        lambda row: mapping_dico.get(row["icd_code"][:3], np.nan)
        if row["icd_version"] == 9
        else row["icd_code"][:3],
        axis=1,
    )
    # patient ids that have at least 1 record of the given ICD10 code category
    diag.dropna(subset=["root"], inplace=True)
    pos_ids = pd.DataFrame(
        diag.loc[diag["root"].str.contains(ICD10_code)]["hadm_id"].unique(),
        columns=["hadm_id"],
    )

    return pos_ids
