from tqdm import tqdm
from my_preprocessing.feature.feature import Feature
import logging
import pandas as pd
from my_preprocessing.preproc.feature import (
    PREPROC_CHART_ICU_PATH,
    ChartEventsHeader,
)
from my_preprocessing.preproc.cohort import CohortHeader
from my_preprocessing.raw.icu import (
    load_icu_chart_events,
    ChartEvents,
)
from my_preprocessing.file_info import save_data
from pathlib import Path

from my_preprocessing.uom_conversion import drop_wrong_uom

logger = logging.getLogger()


class Chart(Feature):
    def __init__(self, cohort: pd.DataFrame, chunksize: int = 10000000):
        self.cohort = cohort
        self.chunksize = chunksize

    def summary_path(self):
        pass

    def feature_path(self) -> Path:
        return PREPROC_CHART_ICU_PATH

    def make(self) -> pd.DataFrame:
        """Function for processing hospital observations from a pickled cohort, optimized for memory efficiency."""
        processed_chunks = [
            self.process_chunk_chart_events(chunk)
            for chunk in tqdm(load_icu_chart_events(self.chunksize))
        ]

        df_cohort = pd.concat(processed_chunks, ignore_index=True)
        df_cohort = drop_wrong_uom(df_cohort, 0.95)
        """Log statistics about the chart events."""
        logger.info(
            f"# Unique Events: {df_cohort[ChartEventsHeader.ITEM_ID].nunique()}"
        )
        logger.info(f"# Admissions: {df_cohort[ChartEventsHeader.STAY_ID].nunique()}")
        logger.info(f"Total rows: {df_cohort.shape[0]}")
        return df_cohort

    def process_chunk_chart_events(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of chart events."""
        chunk = chunk.dropna(subset=[ChartEvents.VALUENUM]).merge(
            self.cohort, on=ChartEvents.STAY_ID
        )
        chunk[ChartEventsHeader.EVENT_TIME_FROM_ADMIT] = (
            chunk[ChartEvents.CHARTTIME] - chunk[CohortHeader.IN_TIME]
        )
        return chunk.drop(["charttime", "intime"], axis=1).dropna().drop_duplicates()

    def save(self) -> pd.DataFrame:
        logger.info("[EXTRACTING CHART EVENTS DATA]")
        out = self.make()
        out = out[[h.value for h in ChartEventsHeader]]
        return save_data(out, self.feature_path(), "OUTPUT")

    def preproc(self):
        pass
