from tqdm import tqdm
from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
from pipeline.preprocessing.outlier_removal import outlier_imputation
from pipeline.file_info.preproc.feature import (
    PREPROC_CHART_ICU_PATH,
    ChartEventsHeader,
)
from pipeline.file_info.preproc.cohort import CohortHeader
from pipeline.file_info.preproc.summary import CHART_FEATURES_PATH, CHART_SUMMARY_PATH
from pipeline.file_info.raw.icu import (
    load_icu_chart_events,
    ChartEvents,
)
from pipeline.file_info.common import save_data
from pathlib import Path

from pipeline.conversion.uom import drop_wrong_uom

logger = logging.getLogger()


class Chart(Feature):
    def __init__(
        self,
        cohort: pd.DataFrame,
        chunksize: int = 10000000,
        thresh=1,
        left_thresh=0,
        impute_outlier=False,
    ):
        self.cohort = cohort
        self.chunksize = chunksize
        self.thresh = thresh
        self.left_thresh = left_thresh
        self.impute_outlier = impute_outlier

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

    def summary(self):
        chart = pd.read_csv(self.feature_path(), compression="gzip")
        freq = (
            chart.groupby([ChartEventsHeader.STAY_ID, ChartEventsHeader.ITEM_ID])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = (
            freq.groupby([ChartEventsHeader.ITEM_ID])["mean_frequency"]
            .mean()
            .reset_index()
        )

        missing = (
            chart[chart[ChartEventsHeader.VALUE_NUM] == 0]
            .groupby(ChartEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="missing_count")
        )
        total = (
            chart.groupby(ChartEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(missing, total, on=ChartEventsHeader.ITEM_ID, how="right")
        summary = pd.merge(freq, summary, on=ChartEventsHeader.ITEM_ID, how="right")
        summary = summary.fillna(0)
        summary.to_csv(CHART_SUMMARY_PATH, index=False)
        summary[ChartEventsHeader.ITEM_ID].to_csv(CHART_FEATURES_PATH, index=False)

    def preproc(self):
        logger.info("[PROCESSING CHART EVENTS DATA]")
        chart = pd.read_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
        chart = outlier_imputation(
            chart,
            "itemid",
            "valuenum",
            self.thresh,
            self.left_thresh,
            self.impute_outlier,
        )

        logger.info("Total number of rows", chart.shape[0])
        chart.to_csv(
            PREPROC_CHART_ICU_PATH,
            compression="gzip",
            index=False,
        )
        logger.info("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
