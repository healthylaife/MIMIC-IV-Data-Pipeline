from enum import StrEnum
from pathlib import Path
import pandas as pd
import logging

RAW_PATH = Path("raw_data") / "mimiciv_2_0"
MAP_PATH = Path("utils") / "mappings" / "ICD9_to_ICD10_mapping.txt"
MAP_NDC_PATH = Path("utils") / "mappings" / "ndc_product.txt"
PREPROC_PATH = Path("preproc_data")

logger = logging.getLogger()


# icd mapping
class IcdMap(StrEnum):
    DIAGNOISIS_TYPE = "diagnosis_type"
    DIAGNOISIS_CODE = "diagnosis_code"
    DIAGNOISIS_DESCRIPTION = "diagnosis_description"
    ICD9CM = "icd9cm"
    ICD10CM = "icd10cm"
    FLAGS = "flags"


def load_static_icd_map() -> pd.DataFrame:
    return pd.read_csv(MAP_PATH, delimiter="\t")


class NdcMap(StrEnum):
    NON_PROPRIETARY_NAME = "NONPROPRIETARYNAME"


def load_ndc_mapping() -> pd.DataFrame:
    return pd.read_csv(MAP_NDC_PATH, delimiter="\t")


def save_data(data: pd.DataFrame, path: Path, data_name: str) -> pd.DataFrame:
    """Save DataFrame to specified path."""
    data.to_csv(path, compression="gzip")
    logger.info(f"[SUCCESSFULLY SAVED {data_name} DATA]")
    return data
