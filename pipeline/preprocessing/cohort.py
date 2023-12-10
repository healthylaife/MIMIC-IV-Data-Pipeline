import pandas as pd
import numpy as np
import datetime
from pipeline.file_info.common import save_data
from pipeline.file_info.preproc.cohort import (
    COHORT_PATH,
    CohortHeader,
    NonIcuCohortHeader,
    IcuCohortHeader,
)
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
        self.admit_col = (
            IcuCohortHeader.IN_TIME if self.icu else NonIcuCohortHeader.ADMIT_TIME
        )
        self.disch_col = (
            IcuCohortHeader.OUT_TIME if self.icu else NonIcuCohortHeader.DISCH_TIME
        )

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
        save_data(self.df, COHORT_PATH / f"{self.name}.csv.gz", "COHORT")

    def save_summary(self):
        summary = "\n".join(
            [
                f"{self.df} FOR {' ICU' if self.icu else ''} DATA",
                f"# Admission Records: {self.df.shape[0]}",
                f"# Patients: {self.df[CohortHeader.PATIENT_ID].nunique()}",
                f"# Positive cases: {self.df[self.df[CohortHeader.LABEL]==1].shape[0]}",
                f"# Negative cases: {self.df[self.df[CohortHeader.LABEL]==0].shape[0]}",
            ]
        )
        with open(COHORT_PATH / f"{self.summary_name}.txt", "w") as f:
            f.write(summary)


def read_cohort(name: str, use_icu: bool) -> pd.DataFrame:
    data = pd.read_csv(
        COHORT_PATH / f"{name}.csv.gz",
        compression="gzip",
    )
    start_time = IcuCohortHeader.IN_TIME if use_icu else NonIcuCohortHeader.ADMIT_TIME
    stop_time = IcuCohortHeader.OUT_TIME if use_icu else NonIcuCohortHeader.DISCH_TIME
    for col in [start_time, stop_time]:
        data[col] = pd.to_datetime(data[col])
    data[CohortHeader.LOS] = (
        (data[stop_time] - data[start_time]).dt.total_seconds() / 3600
    ).astype(int)
    data = data[data[CohortHeader.LOS] > 0]
    data[CohortHeader.AGE] = data[CohortHeader.AGE].astype(int)

    logger.info("[ READ COHORT ]")

    return data
