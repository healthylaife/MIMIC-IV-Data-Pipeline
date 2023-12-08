import pandas as pd
import numpy as np
import datetime
from pipeline.file_info.preproc.cohort import COHORT_PATH, CohortHeader
import logging
from pathlib import Path
from pipeline.file_info.raw.hosp import HospAdmissions

from pipeline.prediction_task import PredictionTask, TargetType

logger = logging.getLogger()


class Cohort:
    def __init__(
        self,
        icu: bool,
        name: str,
        df: pd.DataFrame = pd.DataFrame(),
    ):
        self.df = df
        self.icu = icu
        self.name = name
        self.summary_name = f"summary_{name}"
        self.admit_col = CohortHeader.IN_TIME if self.icu else CohortHeader.ADMIT_TIME
        self.disch_col = CohortHeader.OUT_TIME if self.icu else CohortHeader.DISCH_TIME

    def prepare_mort_labels(self, visits: pd.DataFrame):
        visits = visits.dropna(subset=[self.admit_col, self.disch_col])
        visits[CohortHeader.DOD] = pd.to_datetime(visits[CohortHeader.DOD])
        visits[CohortHeader.LABEL] = np.where(
            (visits[CohortHeader.DOD] >= visits[self.admit_col])
            & (visits[CohortHeader.DOD] <= visits[self.disch_col]),
            1,
            0,
        )
        logger.info(
            f"[ MORTALITY LABELS FINISHED: {visits[CohortHeader.LABEL].sum()} Mortality Cases ]"
        )
        return visits

    def prepare_read_labels(self, visits: pd.DataFrame, nb_days: int):
        gap = datetime.timedelta(days=nb_days)
        visits["next_admit"] = (
            visits.sort_values(by=[self.admit_col])
            .groupby(CohortHeader.PATIENT_ID)[self.admit_col]
            .shift(-1)
        )
        visits["time_to_next"] = visits["next_admit"] - visits[self.disch_col]
        visits[CohortHeader.LABEL] = (
            visits["time_to_next"].notnull() & (visits["time_to_next"] <= gap)
        ).astype(int)
        readmit_cases = visits[CohortHeader.LABEL].sum()
        logger.info(
            f"[ READMISSION LABELS FINISHED: {readmit_cases} Readmission Cases ]"
        )
        return visits.drop(columns=["next_admit", "time_to_next"])

    def prepare_los_labels(self, visits: pd.DataFrame, nb_days):
        visits = visits.dropna(
            subset=[self.admit_col, self.disch_col, CohortHeader.LOS]
        )
        visits[CohortHeader.LABEL] = (visits[CohortHeader.LOS] > nb_days).astype(int)
        logger.info(
            f"[ LOS LABELS FINISHED: {visits[CohortHeader.LABEL].sum()} LOS Cases ]"
        )
        return visits

    def prepare_labels(self, visits: pd.DataFrame, prediction_task: PredictionTask):
        if prediction_task.target_type == TargetType.MORTALITY:
            df = self.prepare_mort_labels(visits)
        elif prediction_task.target_type == TargetType.READMISSION:
            df = self.prepare_read_labels(visits, prediction_task.nb_days)
        elif prediction_task.target_type == TargetType.LOS:
            df = self.prepare_los_labels(visits, prediction_task.nb_days)
        df = df.sort_values(by=[CohortHeader.PATIENT_ID, self.admit_col])
        self.df = df.rename(columns={HospAdmissions.RACE: CohortHeader.ETHICITY})

    def save(self):
        self.df.to_csv(
            COHORT_PATH / f"{self.name}.csv.gz",
            index=False,
            compression="gzip",
        )
        logger.info(f"[ COHORT {self.name} SAVED ]")

    def save_summary(self):
        self.df.to_csv(
            COHORT_PATH / f"{self.summary_name}.csv.gz",
            index=False,
            compression="gzip",
        )
        logger.info(f"[ SUMMARY {self.summary_name} SAVED ]")

    def read_output(self) -> pd.DataFrame:
        data = pd.read_csv(
            COHORT_PATH / f"{self.name}.csv.gz",
            compression="gzip",
        )
        for col in [CohortHeader.ADMIT_TIME, CohortHeader.DISCH_TIME]:
            data[col] = pd.to_datetime(data[col])

        data[CohortHeader.LOS] = (
            (
                data[CohortHeader.DISCH_TIME] - data[CohortHeader.ADMIT_TIME]
            ).dt.total_seconds()
            / 3600
        ).astype(int)
        data = data[data[CohortHeader.LOS] > 0]
        data[CohortHeader.AGE] = data[CohortHeader.AGE].astype(int)

        logger.info("[ READ COHORT ]")
        self.df = data
