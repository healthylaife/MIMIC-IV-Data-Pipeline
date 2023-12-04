from my_preprocessing.feature.feature import Feature
import logging
import pandas as pd
from my_preprocessing.preproc.feature import (
    PREPROC_OUT_ICU_PATH,
    OutputEventsHeader,
)
from my_preprocessing.preproc.cohort import CohortHeader
from my_preprocessing.raw.icu import load_icu_output_events, OuputputEvents
from my_preprocessing.file_info import save_data
from pathlib import Path

logger = logging.getLogger()


class OutputEvents(Feature):
    def summary_path(self):
        pass

    def feature_path(self) -> Path:
        return PREPROC_OUT_ICU_PATH

    def make(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Function for getting hosp observations pertaining to a pickled cohort.
        Function is structured to save memory when reading and transforming data."""
        raw_out = load_icu_output_events()
        out = raw_out.merge(
            cohort[[CohortHeader.STAY_ID, CohortHeader.IN_TIME, CohortHeader.OUT_TIME]],
            on=CohortHeader.STAY_ID,
        )
        out[OutputEventsHeader.EVENT_TIME_FROM_ADMIT] = (
            out[OuputputEvents.CHART_TIME] - out[CohortHeader.IN_TIME]
        )
        out = out.dropna()
        # Print unique counts and value_counts
        logger.info("# Unique Events:  ", out[OuputputEvents.ITEM_ID].nunique())
        logger.info("# Admissions:  ", out[OuputputEvents.STAY_ID].nunique())
        logger.info("Total rows", out.shape[0])

        return out

    def save(self) -> pd.DataFrame:
        logger.info("[EXTRACTING OUTPUT EVENTS DATA]")
        out = self.make()
        out = out[[h.value for h in OutputEventsHeader]]
        return save_data(out, PREPROC_OUT_ICU_PATH, "OUTPUT")

    def preproc(self):
        pass
