import pandas as pd
import numpy as np


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


def read_ndc_mapping(map_path: str) -> pd.DataFrame:
    """Reads and processes NDC mapping table from a file."""
    ndc_map = pd.read_csv(map_path, header=0, delimiter="\t", encoding="latin1")
    ndc_map["NONPROPRIETARYNAME"] = ndc_map["NONPROPRIETARYNAME"].fillna("").str.lower()
    ndc_map.columns = ndc_map.columns.str.lower()
    return ndc_map


def get_EPC(s: str) -> list:
    """Extracts the Established Pharmacologic Class (EPC) from a string."""
    if not isinstance(s, str):
        return np.nan

    return [phrase for phrase in s.split(",") if "[EPC]" in phrase]
