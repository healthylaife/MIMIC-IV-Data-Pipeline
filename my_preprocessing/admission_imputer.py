import pandas as pd
from uuid import uuid1
from collections import defaultdict
from typing import Union, List, Tuple
from functools import partial
from multiprocessing import Pool
import glob
import os


def hadm_imputer(
    charttime: pd.Timestamp,
    hadm_old: Union[str, float],
    hadm_ids_w_timestamps: List[Tuple[str, pd.Timestamp, pd.Timestamp]],
) -> Tuple[str, pd.Timestamp, pd.Timestamp]:
    """
    Impute hospital admission ID based on the chart time and a list of admission IDs with timestamps.
    """

    # If old HADM ID exists and is valid, use that
    if not pd.isna(hadm_old):
        hadm_old = str(int(hadm_old))
        for h_id, adm_time, disch_time in hadm_ids_w_timestamps:
            if h_id == hadm_old:
                return hadm_old, adm_time, disch_time

    # Filter and sort HADM IDs based on their proximity to the lab event charttime
    valid_hadm_ids = [
        (hadm_id, admittime, dischtime)
        for hadm_id, admittime, dischtime in hadm_ids_w_timestamps
        if admittime <= charttime <= dischtime
    ]
    valid_hadm_ids.sort(key=lambda x: abs(charttime - x[1]))

    # Return the most relevant HADM ID or None if no valid ID is found
    return valid_hadm_ids[0] if valid_hadm_ids else (None, None, None)


def impute_missing_hadm_ids(
    lab_table: pd.DataFrame, subject_hadm_admittime_tracker: dict
) -> None:
    """Impute missing HADM IDs in the lab table using subject_hadm_admittime_tracker."""

    def impute_row(row):
        """Helper function to impute data for a single row."""
        new_hadm_id, new_admittime, new_dischtime = hadm_imputer(
            row["charttime"],
            row["hadm_id"],
            subject_hadm_admittime_tracker.get(row["subject_id"], []),
        )
        return pd.Series(
            [new_hadm_id, new_admittime, new_dischtime],
            index=["hadm_id_new", "admittime", "dischtime"],
        )

    # Apply the imputation function to each row
    imputed_data = lab_table.apply(impute_row, axis=1)

    # Combine the original data with the imputed data
    result = pd.concat([lab_table, imputed_data], axis=1)

    # Generate a unique table name and save the result
    tab_name = str(uuid1())
    result.to_csv(f"{tab_name}.csv", index=False)


def impute_hadm_ids(
    lab_table: pd.DataFrame, admission_table: pd.DataFrame
) -> pd.DataFrame:
    # Convert columns to datetime
    lab_table["charttime"] = pd.to_datetime(lab_table["charttime"])
    admission_table["admittime"] = pd.to_datetime(admission_table["admittime"])
    admission_table["dischtime"] = pd.to_datetime(admission_table["dischtime"])

    # Create a tracker dictionary from the admission table
    subject_hadm_admittime_tracker = defaultdict(list)
    for row in admission_table.itertuples():
        subject_hadm_admittime_tracker[row.subject_id].append(
            (row.hadm_id, row.admittime, row.dischtime)
        )

    # Split the lab table into chunks
    chunks = [lab_table[i : i + 100] for i in range(0, lab_table.shape[0], 100)]

    # Function for processing each chunk
    impute_func = partial(
        impute_missing_hadm_ids,
        subject_hadm_admittime_tracker=subject_hadm_admittime_tracker,
    )

    # Process chunks in parallel
    with Pool(8) as p:
        p.map(impute_func, chunks)

    # Consolidate processed chunks into a single DataFrame
    all_csvs = glob.glob("*.csv")
    lab_tab = pd.concat([pd.read_csv(csv) for csv in all_csvs])

    # Clean up temporary files
    for csv in all_csvs:
        os.remove(csv)

    return lab_tab
