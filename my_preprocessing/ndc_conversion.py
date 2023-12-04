import pandas as pd
import numpy as np

from my_preprocessing.file_info import MAP_NDC_PATH, HospPrescriptions
from enum import StrEnum


class NdcMappingHeader(StrEnum):
    PRODUCT_NDC = "productndc"
    NON_PROPRIETARY_NAME = "nonproprietaryname"
    PHARM_CLASSES = "pharm_classes"
    NEW_NDC = "new_ndc"


def prepare_ndc_mapping() -> pd.DataFrame:
    ndc_map = read_ndc_mapping()[
        [
            NdcMappingHeader.PRODUCT_NDC,
            NdcMappingHeader.NON_PROPRIETARY_NAME,
            NdcMappingHeader.PHARM_CLASSES,
        ]
    ]
    ndc_map[NdcMappingHeader.NON_PROPRIETARY_NAME] = (
        ndc_map[NdcMappingHeader.NON_PROPRIETARY_NAME].fillna("").str.lower()
    )
    # Normalize the NDC codes in the mapping table so that they can be merged
    ndc_map.loc[:, NdcMappingHeader.NEW_NDC] = ndc_map[
        NdcMappingHeader.PRODUCT_NDC
    ].apply(format_ndc_table)
    ndc_map = ndc_map.drop_duplicates(
        subset=[NdcMappingHeader.NEW_NDC, NdcMappingHeader.NON_PROPRIETARY_NAME]
    )
    return ndc_map


def ndc_to_str(ndc: int) -> str:
    """Converts NDC code to a string with leading zeros restored, keeping only the first 9 digits."""
    if ndc < 0:  # Handling dummy values
        return np.nan
    ndc_str = str(ndc).zfill(11)
    return ndc_str[:-2]


def format_ndc_table(ndc: str) -> str:
    """Formats NDC code from the mapping table to the standard 11-digit format, taking only the first 9 digits."""
    parts = ndc.split("-")
    formatted_ndc = "".join(
        part.zfill(length) for part, length in zip(parts, [5, 4, 2])
    )
    return formatted_ndc[:9]  # Taking only the manufacturer and product sections


def read_ndc_mapping() -> pd.DataFrame:
    """Reads and processes NDC mapping table from a file."""
    ndc_map = pd.read_csv(MAP_NDC_PATH, delimiter="\t", encoding="latin1")
    ndc_map.columns = ndc_map.columns.str.lower()
    return ndc_map


def get_EPC(s: str) -> list:
    """Extracts the Established Pharmacologic Class (EPC) from a string."""
    if not isinstance(s, str):
        return np.nan

    return [phrase for phrase in s.split(",") if "[EPC]" in phrase]
