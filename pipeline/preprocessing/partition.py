import pandas as pd
import numpy as np

from pipeline.file_info.preproc.cohort import CohortHeader
import logging

logger = logging.getLogger()


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
