from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
from pipeline.file_info.preproc.feature import (
    PREPROC_OUT_ICU_PATH,
    OutputEventsHeader,
)
from pipeline.file_info.preproc.cohort import CohortHeader
from pipeline.file_info.preproc.summary import OUT_FEATURES_PATH, OUT_SUMMARY_PATH
from pipeline.file_info.raw.icu import load_icu_output_events, OuputputEvents
from pipeline.file_info.common import save_data
from pathlib import Path

logger = logging.getLogger()


class OutputEvents(Feature):
    def __init__(self, cohort: pd.DataFrame):
        self.cohort = cohort

    def summary_path(self):
        pass

    def feature_path(self) -> Path:
        return PREPROC_OUT_ICU_PATH

    def make(self) -> pd.DataFrame:
        """Function for getting hosp observations pertaining to a pickled cohort.
        Function is structured to save memory when reading and transforming data."""
        raw_out = load_icu_output_events()
        out = raw_out.merge(
            self.cohort[
                [CohortHeader.STAY_ID, CohortHeader.IN_TIME, CohortHeader.OUT_TIME]
            ],
            on=CohortHeader.STAY_ID,
        )
        out[OutputEventsHeader.EVENT_TIME_FROM_ADMIT] = (
            out[OuputputEvents.CHART_TIME] - out[CohortHeader.IN_TIME]
        )
        out = out.dropna()

        # Print unique counts and value_counts
        logger.info(f"# Unique Events: {out[OuputputEvents.ITEM_ID].nunique()}")
        logger.info(f"# Admissions: {out[OuputputEvents.STAY_ID].nunique()}")
        logger.info(f"Total rows: {out.shape[0]}")

        return out

    def save(self) -> pd.DataFrame:
        logger.info("[EXTRACTING OUTPUT EVENTS DATA]")
        out = self.make()
        out = out[[h.value for h in OutputEventsHeader]]
        return save_data(out, PREPROC_OUT_ICU_PATH, "OUTPUT")

    def preproc(self):
        pass

    def summary(self):
        out = pd.read_csv(self.feature_path(), compression="gzip")
        freq = (
            out.groupby([OutputEventsHeader.STAY_ID, OutputEventsHeader.ITEM_ID])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = (
            out.groupby(OutputEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(freq, total, on=OutputEventsHeader.ITEM_ID, how="right")
        summary = summary.fillna(0)
        summary.to_csv(OUT_SUMMARY_PATH, index=False)
        summary[OutputEventsHeader.ITEM_ID].to_csv(OUT_FEATURES_PATH, index=False)
        return summary[OutputEventsHeader.ITEM_ID]
