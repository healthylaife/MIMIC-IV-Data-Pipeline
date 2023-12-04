import pandas as pd
from tqdm import tqdm
from my_preprocessing.raw.icu import (
    ChartEvents,
    load_icu_output_events,
    load_icu_procedure_events,
    load_input_events,
    load_icu_chart_events,
    InputEvents,
)
import logging
from my_preprocessing.preproc.cohort import CohortHeader
from my_preprocessing.preproc_file_info import (
    OutputEventsHeader,
    IcuProceduresHeader,
    ChartEventsHeader,
    MedicationsHeader,
)
from my_preprocessing.uom_conversion import drop_wrong_uom


logger = logging.getLogger()


def process_chunk_chart_events(
    chunk: pd.DataFrame, cohort: pd.DataFrame
) -> pd.DataFrame:
    """Process a single chunk of chart events."""
    chunk = chunk.dropna(subset=[ChartEvents.VALUENUM]).merge(
        cohort, on=ChartEvents.STAY_ID
    )
    chunk[ChartEventsHeader.EVENT_TIME_FROM_ADMIT] = (
        chunk[ChartEvents.CHARTTIME] - chunk[CohortHeader.IN_TIME]
    )
    return chunk.drop(["charttime", "intime"], axis=1).dropna().drop_duplicates()


def make_chart_events_features(
    cohort: pd.DataFrame, chunksize=10000000
) -> pd.DataFrame:
    """Function for processing hospital observations from a pickled cohort, optimized for memory efficiency."""
    cohort_columns = [CohortHeader.STAY_ID, CohortHeader.IN_TIME]
    processed_chunks = [
        process_chunk_chart_events(chunk, cohort[cohort_columns])
        for chunk in tqdm(load_icu_chart_events(chunksize))
    ]
    df_cohort = pd.concat(processed_chunks, ignore_index=True)
    df_cohort = drop_wrong_uom(df_cohort, 0.95)
    """Log statistics about the chart events."""
    logger.info(f"# Unique Events: {df_cohort[ChartEventsHeader.ITEM_ID].nunique()}")
    logger.info(f"# Admissions: {df_cohort[ChartEventsHeader.STAY_ID].nunique()}")
    logger.info(f"Total rows: {df_cohort.shape[0]}")
    return df_cohort


def make_output_events_features(cohort: pd.DataFrame) -> pd.DataFrame:
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


def make_procedures_features_icu(cohort: pd.DataFrame) -> pd.DataFrame:
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


def make_medications_features_icu(cohort: pd.DataFrame) -> pd.DataFrame:
    adm = cohort[
        [CohortHeader.HOSPITAL_ADMISSION_ID, CohortHeader.STAY_ID, CohortHeader.IN_TIME]
    ]
    med = load_input_events()
    med = med.merge(adm, on=InputEvents.STAY_ID)
    med[MedicationsHeader.START_HOURS_FROM_ADMIT] = (
        med[InputEvents.STARTTIME] - med[CohortHeader.IN_TIME]
    )
    med[MedicationsHeader.STOP_HOURS_FROM_ADMIT] = (
        med[InputEvents.ENDTIME] - med[CohortHeader.IN_TIME]
    )
    med = med.dropna()
    logger.info("# of unique type of drug: ", med[InputEvents.ITEMID].nunique())
    logger.info("# Admissions:  ", med[InputEvents.STAY_ID].nunique())
    logger.info("# Total rows", med.shape[0])
    return med
