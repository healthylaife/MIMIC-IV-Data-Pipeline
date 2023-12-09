import pandas as pd
import numpy as np

from pipeline.file_info.common import load_static_icd_map, IcdMap
from pipeline.file_info.raw.hosp import HospDiagnosesIcd

ROOT_ICD_CONVERT = "root_icd10_convert"


class IcdConverter:
    def __init__(self):
        self.conversions_icd_9_10 = self._get_conversions_icd_9_10()

    def _get_conversions_icd_9_10(self) -> dict:
        """Create mapping dictionary ICD9 -> ICD10"""
        icd_map_df = load_static_icd_map()
        filtered_df = icd_map_df[icd_map_df[IcdMap.DIAGNOISIS_CODE].str.len() == 3]
        filtered_df = filtered_df.drop_duplicates(subset=IcdMap.DIAGNOISIS_CODE)
        return dict(zip(filtered_df[IcdMap.DIAGNOISIS_CODE], filtered_df[IcdMap.ICD10]))

    def standardize_icd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes ICD codes in a DataFrame."""
        df[ROOT_ICD_CONVERT] = df.apply(
            lambda row: self.conversions_icd_9_10.get(
                row[HospDiagnosesIcd.ICD_CODE][:3], np.nan
            )
            if row[HospDiagnosesIcd.ICD_VERSION] == 9
            else row[HospDiagnosesIcd.ICD_CODE],
            axis=1,
        )
        df[HospDiagnosesIcd.ROOT] = df[ROOT_ICD_CONVERT].apply(
            lambda x: x[:3] if type(x) is str else np.nan
        )
        return df

    def get_pos_ids(self, diag: pd.DataFrame, ICD10_code: str) -> pd.Series:
        """Extracts unique hospital admission IDs where 'root' contains a specific ICD-10 code."""
        return diag[diag[HospDiagnosesIcd.ROOT].str.contains(ICD10_code, na=False)][
            HospDiagnosesIcd.HOSPITAL_ADMISSION_ID
        ].unique()
