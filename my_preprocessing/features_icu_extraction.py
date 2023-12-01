import pandas as pd
from tqdm import tqdm
from my_preprocessing.raw_file_info import (
    ICU_CHART_EVENTS_PATH,
    ChartEvents,
    load_icu_output_events,
    load_icu_procedure_events,
    ICU_INPUT_EVENT_PATH,
    InputEvents,
)
import logging
from my_preprocessing.preproc_file_info import (
    CohortHeader,
    OutputEventsHeader,
    IcuProceduresHeader,
)
from my_preprocessing.uom_conversion import drop_wrong_uom


logger = logging.getLogger()


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
    df_cohort = drop_wrong_uom(df_cohort, 0.95)
    logger.info("# Unique Events:  ", df_cohort.itemid.nunique())
    logger.info("# Admissions:  ", df_cohort.stay_id.nunique())
    logger.info("Total rows", df_cohort.shape[0])

    return df_cohort


def make_output_events(cohort: pd.DataFrame) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort.
    Function is structured to save memory when reading and transforming data."""
    outputevents = load_icu_output_events()
    df_cohort = outputevents.merge(
        cohort[[CohortHeader.STAY_ID, CohortHeader.IN_TIME, CohortHeader.OUT_TIME]],
        on=OutputEventsHeader.STAY_ID,
    )
    df_cohort[OutputEventsHeader.EVENT_TIME_FROM_ADMIT] = (
        df_cohort[OutputEventsHeader.CHART_TIME] - df_cohort[OutputEventsHeader.IN_TIME]
    )
    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    logger.info("# Unique Events:  ", df_cohort[OutputEventsHeader.ITEM_ID].nunique())
    logger.info("# Admissions:  ", df_cohort[OutputEventsHeader.STAY_ID].nunique())
    logger.info("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort


def make_procedures_feature_icu(cohort: pd.DataFrame) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    module = load_icu_procedure_events()
    # Only consider values in our cohort
    df_cohort = module.merge(
        cohort[
            [
                CohortHeader.PATIENT_ID,
                CohortHeader.HOSPITAL_ADMISSION_ID,
                CohortHeader.STAY_ID,
                CohortHeader.IN_TIME,
                CohortHeader.OUT_TIME,
            ]
        ],
        on=CohortHeader.STAY_ID,
    )
    df_cohort[IcuProceduresHeader.EVENT_TIME_FROM_ADMIT] = (
        df_cohort[IcuProceduresHeader.START_TIME]
        - df_cohort[IcuProceduresHeader.IN_TIME]
    )

    df_cohort = df_cohort.dropna()
    logger.info(
        "# Unique Events:  ", df_cohort[IcuProceduresHeader.ITEM_ID].dropna().nunique()
    )
    logger.info("# Admissions:  ", df_cohort[IcuProceduresHeader.STAY_ID].nunique())
    logger.info("Total rows", df_cohort.shape[0])
    return df_cohort


def make_icu_input_events(cohort: pd.DataFrame) -> pd.DataFrame:
    adm = cohort[["hadm_id", "stay_id", "intime"]]
    med = pd.read_csv(
        ICU_INPUT_EVENT_PATH,
        compression="gzip",
        usecols=[f for f in InputEvents],
        parse_dates=[InputEvents.STARTTIME, InputEvents.ENDTIME],
    )
    med = med.merge(adm, left_on=InputEvents.STAY_ID, right_on="stay_id", how="inner")
    med["start_hours_from_admit"] = med[InputEvents.STARTTIME] - med["intime"]
    med["stop_hours_from_admit"] = med[InputEvents.ENDTIME] - med["intime"]
    med = med.dropna()
    logger.info("# of unique type of drug: ", med[InputEvents.ITEMID].nunique())
    logger.info("# Admissions:  ", med[InputEvents.STAY_ID].nunique())
    logger.info("# Total rows", med.shape[0])
    return med
