from pathlib import Path
import pandas as pd
import datetime
import logging
from my_preprocessing.raw.hosp import (
    load_hosp_patients,
    load_hosp_admissions,
    HospAdmissions,
)
from my_preprocessing.raw.icu import load_icu_icustays


from my_preprocessing.preproc.cohort import COHORT_PATH, CohortHeader
from my_preprocessing.prediction_task import PredictionTask, TargetType

from my_preprocessing.preprocessing import (
    make_patients,
    make_icu_visits,
    make_no_icu_visits,
    filter_visits,
    partition_by_mort,
    partition_by_los,
    partition_by_readmit,
)

logger = logging.getLogger()


class CohortExtractor:
    def __init__(
        self,
        prediction_task: PredictionTask,
        preproc_dir: Path,
        cohort_output: Path,
        summary_output: Path,
    ):
        self.prediction_task = prediction_task
        self.preproc_dir = preproc_dir
        self.cohort_output = cohort_output
        self.summary_output = summary_output

    def get_icu_status(self) -> str:
        return "ICU" if self.prediction_task.use_icu else "Non-ICU"

    def generate_extract_log(self) -> str:
        icu_log = self.get_icu_status()
        task_info = f"{icu_log} | {self.prediction_task.target_type}"
        if self.prediction_task.disease_readmission:
            task_info += f" DUE TO {self.prediction_task.disease_readmission}"

        if self.prediction_task.disease_selection:
            task_info += f" ADMITTED DUE TO {self.prediction_task.disease_selection}"

        return f"EXTRACTING FOR: {task_info} | {self.prediction_task.nb_days} |".upper()

    def generate_output_suffix(self) -> str:
        return (
            self.get_icu_status()  # .lower()
            + "_"
            + self.prediction_task.target_type.lower().replace(" ", "_")
            + "_"
            + str(self.prediction_task.nb_days)
            + "_"
            + self.prediction_task.disease_readmission
            if self.prediction_task.disease_readmission
            else ""
        )

    def fill_outputs(self) -> None:
        output_suffix = self.generate_output_suffix()
        disease_selection = (
            f"_{self.prediction_task.disease_selection}"
            if self.prediction_task.disease_selection
            else ""
        )
        self.cohort_output = (
            self.cohort_output or f"cohort_{output_suffix}{disease_selection}"
        )

        self.summary_output = (
            self.summary_output or f"summary_{output_suffix}{disease_selection}"
        )

    def get_case_ctrls(
        self,
        df: pd.DataFrame,
        gap: int,
        group_col: str,
        admit_col: str,
        disch_col: str,
        death_col: str,
    ) -> pd.DataFrame:
        """Handles logic for creating the labelled cohort based on the specified target

        Parameters:
        df (pd.DataFrame): The dataframe to partition.
        gap (int): Time gap for readmissions or LOS threshold.
        group_col (str): Column to group by.
        admit_col (str), disch_col (str), death_col (str): Relevant date columns.
        """
        if self.prediction_task.target_type == TargetType.MORTALITY:
            return partition_by_mort(df, group_col, admit_col, disch_col, death_col)
        elif self.prediction_task.target_type == TargetType.READMISSION:
            gap = datetime.timedelta(days=gap)
            return partition_by_readmit(df, gap, group_col, admit_col, disch_col)
        elif self.prediction_task.target_type == TargetType.LOS:
            return partition_by_los(df, gap, group_col, admit_col, disch_col)

    def save_cohort(self, cohort: pd.DataFrame) -> None:
        cohort.to_csv(
            COHORT_PATH / f"{self.cohort_output}.csv.gz",
            index=False,
            compression="gzip",
        )
        logger.info(f"[ COHORT {self.cohort_output} SAVED ]")

    def load_hospital_data(self):
        return load_hosp_patients(), load_hosp_admissions()

    def create_visits(self, hosp_patients, hosp_admissions):
        if self.prediction_task.use_icu:
            icu_icustays = load_icu_icustays()
            return make_icu_visits(
                icu_icustays, hosp_patients, self.prediction_task.target_type
            )
        else:
            return make_no_icu_visits(hosp_admissions, self.prediction_task.target_type)

    def prepare_cohort(self, visits):
        visits = self.filter_and_merge_visits(visits)
        admit_col = (
            CohortHeader.IN_TIME
            if self.prediction_task.use_icu
            else CohortHeader.ADMIT_TIME
        )
        disch_col = (
            CohortHeader.OUT_TIME
            if self.prediction_task.use_icu
            else CohortHeader.DISCH_TIME
        )

        return self.get_case_ctrls(
            df=visits,
            gap=self.prediction_task.nb_days,
            group_col=CohortHeader.PATIENT_ID,
            admit_col=admit_col,
            disch_col=disch_col,
            death_col=CohortHeader.DOD,
        ).rename(columns={HospAdmissions.RACE: CohortHeader.ETHICITY})

    def filter_and_merge_visits(self, visits):
        visits = filter_visits(
            visits,
            self.prediction_task.disease_readmission,
            self.prediction_task.disease_selection,
        )
        patients_data = make_patients(load_hosp_patients())
        patients_filtered = patients_data.loc[patients_data["age"] >= 18]
        admissions_info = load_hosp_admissions()[
            [
                HospAdmissions.HOSPITAL_ADMISSION_ID,
                HospAdmissions.INSURANCE,
                HospAdmissions.RACE,
            ]
        ]
        visits = visits.merge(patients_filtered, on=CohortHeader.PATIENT_ID)
        visits = visits.merge(admissions_info, on=CohortHeader.HOSPITAL_ADMISSION_ID)
        return visits

    def extract(self) -> pd.DataFrame:
        logger.info("===========MIMIC-IV v2.0============")
        self.fill_outputs()
        logger.info(self.generate_extract_log())

        hosp_patients, hosp_admissions = self.load_hospital_data()
        visits = self.create_visits(hosp_patients, hosp_admissions)
        cohort = self.prepare_cohort(visits)

        self.save_cohort(cohort)
        logger.info("[ COHORT SUCCESSFULLY SAVED ]")
        logger.info(self.cohort_output)

        self.save_summary(cohort)

        return cohort

    def save_summary(self, cohort: pd.DataFrame) -> None:
        summary = "\n".join(
            [
                f"{self.prediction_task.disease_readmission} FOR {self.get_icu_status()} DATA",
                f"# Admission Records: {cohort.shape[0]}",
                f"# Patients: {cohort[CohortHeader.PATIENT_ID].nunique()}",
                f"# Positive cases: {cohort[cohort['label']==1].shape[0]}",
                f"# Negative cases: {cohort[cohort['label']==0].shape[0]}",
            ]
        )

        with open(f"./data/cohort/{self.summary_output}.txt", "w") as f:
            f.write(summary)

        print("[ SUMMARY SUCCESSFULLY SAVED ]")
