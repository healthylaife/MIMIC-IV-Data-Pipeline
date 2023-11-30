from pathlib import Path
import pandas as pd
import datetime
import logging
from my_preprocessing.raw_file_info import (
    load_hosp_patients,
    load_hosp_admissions,
    load_icu_icustays,
    HospAdmissions,
)
from my_preprocessing.preproc_file_info import COHORT_PATH
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

    def generate_icu_log(self) -> str:
        return "ICU" if self.prediction_task.use_icu else "Non-ICU"

    def generate_extract_log(self) -> str:
        if not (self.prediction_task.disease_selection):
            if self.prediction_task.disease_readmission:
                return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.prediction_task.target_type} DUE TO {self.prediction_task.disease_readmission} | {str(self.prediction_task.nb_days)} | ".upper()
            return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.prediction_task.target_type} | {str(self.prediction_task.nb_days)} |".upper()
        else:
            if self.prediction_task.disease_readmission:
                return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.prediction_task.target_type} DUE TO {self.prediction_task.disease_readmission} | ADMITTED DUE TO {self.prediction_task.disease_selection} | {str(self.prediction_task.nb_days)} |".upper()
        return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.prediction_task.target_type} | ADMITTED DUE TO {self.prediction_task.disease_selection} | {str(self.prediction_task.nb_days)} |".upper()

    def generate_output_suffix(self) -> str:
        return (
            self.generate_icu_log()  # .lower()
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
        if not self.cohort_output:
            self.cohort_output = "cohort_" + self.generate_output_suffix()
        if not self.summary_output:
            self.summary_output = "summary_" + self.generate_output_suffix()
        if self.prediction_task.disease_selection:
            self.cohort_output = (
                self.cohort_output + "_" + self.prediction_task.disease_selection
            )
            self.summary_output = (
                self.summary_output + "_" + self.prediction_task.disease_selection
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
            COHORT_PATH / (self.cohort_output + ".csv.gz"),
            index=False,
            compression="gzip",
        )
        logger.info("[ COHORT " + self.cohort_output + " SAVED ]")

    def extract(self) -> None:
        logger.info("===========MIMIC-IV v2.0============")
        self.fill_outputs()
        logger.info(self.generate_extract_log())

        hosp_patients = load_hosp_patients()
        hosp_admissions = load_hosp_admissions()
        visits = pd.DataFrame()
        if self.prediction_task.use_icu:
            icu_icustays = load_icu_icustays()
            visits = make_icu_visits(
                icu_icustays, hosp_patients, self.prediction_task.target_type
            )
        else:
            visits = make_no_icu_visits(
                hosp_admissions, self.prediction_task.target_type
            )

        visits = filter_visits(
            visits,
            self.prediction_task.disease_readmission,
            self.prediction_task.disease_selection,
        )
        patients = make_patients(hosp_patients)
        patients = patients.loc[patients["age"] >= 18]
        admissions_info = hosp_admissions[
            [
                HospAdmissions.HOSPITAL_AMISSION_ID,
                HospAdmissions.INSURANCE,
                HospAdmissions.RACE,
            ]
        ]
        visits = visits.merge(patients, how="inner", on="subject_id")
        visits = visits.merge(admissions_info, how="inner", on="hadm_id")

        cohort = self.get_case_ctrls(
            df=visits,
            gap=self.prediction_task.nb_days,
            group_col="subject_id",
            admit_col="intime" if self.prediction_task.use_icu else "admittime",
            disch_col="outtime" if self.prediction_task.use_icu else "dischtime",
            death_col="dod",
        )

        cohort = cohort.rename(columns={"race": "ethnicity"})

        self.save_cohort(cohort)
        logger.info("[ COHORT SUCCESSFULLY SAVED ]")
        logger.info(self.cohort_output)
        return cohort
