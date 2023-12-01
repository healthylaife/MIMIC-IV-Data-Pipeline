import pandas as pd
from tqdm import tqdm
from my_preprocessing.raw_file_info import (
    ICU_CHART_EVENTS_PATH,
    ChartEvents,
    load_icu_output_events,
    ICU_PROCEDURE_EVENTS_PATH,
    ICU_INPUT_EVENT_PATH,
    InputEvents,
)
import logging
from my_preprocessing.preproc_file_info import CohortHeader, OutputEventsHeader


def make_chart_events(cohort: pd.DataFrame, chunksize=10000000) -> pd.DataFrame:
    """Function for processing hospital observations from a pickled cohort. optimized for memory efficiency"""

    # Only consider values in our cohort TODO: filter?
    processed_chunks = []
    for chunk in tqdm(
        pd.read_csv(
            ICU_CHART_EVENTS_PATH,
            compression="gzip",
            usecols=[
                ChartEvents.STAY_ID.value,
                ChartEvents.CHARTTIME.value,
                ChartEvents.ITEMID.value,
                ChartEvents.VALUENUM.value,
                ChartEvents.VALUEOM.value,
            ],
            parse_dates=[ChartEvents.CHARTTIME.value],
            chunksize=chunksize,
        )
    ):
        chunk = chunk.dropna(subset=["valuenum"])
        chunk_merged = chunk.merge(
            cohort[["stay_id", "intime"]],
            how="inner",
            left_on="stay_id",
            right_on="stay_id",
        )
        chunk_merged["event_time_from_admit"] = (
            chunk_merged["charttime"] - chunk_merged["intime"]
        )
        chunk_merged.drop(["charttime", "intime"], axis=1, inplace=True)
        chunk_merged.dropna(inplace=True)
        chunk_merged.drop_duplicates(inplace=True)
        processed_chunks.append(chunk_merged)
    df_cohort = pd.concat(processed_chunks, ignore_index=True)
    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    return df_cohort


def make_output_events(cohort: pd.DataFrame) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort.
    Function is structured to save memory when reading and transforming data."""
    outputevents = load_icu_output_events()
    df_cohort = outputevents.merge(
        cohort[[CohortHeader.STAY_ID, CohortHeader.IN_TIME, CohortHeader.OUT_TIME]],
        on=OutputEventsHeader.STAY_ID,
    )
    df_cohort["event_time_from_admit"] = df_cohort["charttime"] - df_cohort["intime"]
    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort


def make_icu_procedure_events(cohort: pd.DataFrame) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    module = pd.read_csv(
        ICU_PROCEDURE_EVENTS_PATH,
        compression="gzip",
        usecols=["stay_id", "starttime", "itemid"],
        parse_dates=["starttime"],
    ).drop_duplicates()
    # Only consider values in our cohort
    df_cohort = module.merge(
        cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
        how="inner",
        left_on="stay_id",
        right_on="stay_id",
    )
    df_cohort["event_time_from_admit"] = df_cohort["starttime"] - df_cohort["intime"]

    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.dropna().nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])
    return df_cohort


def make_icu_input_events(cohort: pd.DataFrame) -> pd.DataFrame:
    adm = cohort[["hadm_id", "stay_id", "intime"]]
    med = pd.read_csv(
        ICU_INPUT_EVENT_PATH,
        compression="gzip",
        usecols=[
            InputEvents.SUBJECT_ID.value,
            InputEvents.STAY_ID.value,
            InputEvents.ITEMID.value,
            InputEvents.STARTTIME.value,
            InputEvents.ENDTIME.value,
            InputEvents.RATE.value,
            InputEvents.AMOUNT.value,
            InputEvents.ORDERID.value,
        ],
        parse_dates=[InputEvents.STARTTIME.value, InputEvents.ENDTIME.value],
    )
    med = med.merge(
        adm, left_on=InputEvents.STAY_ID.value, right_on="stay_id", how="inner"
    )
    med["start_hours_from_admit"] = med[InputEvents.STARTTIME.value] - med["intime"]
    med["stop_hours_from_admit"] = med[InputEvents.ENDTIME.value] - med["intime"]
    med = med.dropna()
    print("# of unique type of drug: ", med[InputEvents.ITEMID.value].nunique())
    print("# Admissions:  ", med[InputEvents.STAY_ID.value].nunique())
    print("# Total rows", med.shape[0])

    return med
