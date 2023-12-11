from typing import Tuple
import pandas as pd
import logging
from pipeline.file_info.raw.hosp import (
    load_hosp_patients,
    load_hosp_admissions,
    HospAdmissions,
)
from pipeline.file_info.raw.icu import load_icu_icustays
from pipeline.file_info.preproc.cohort import CohortHeader
from pipeline.prediction_task import PredictionTask
from pipeline.preprocessing.visit import (
    make_patients,
    make_icu_visits,
    make_no_icu_visits,
    filter_visits,
)
from pipeline.preprocessing.cohort import Cohort

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class CohortExtractor:
    """
    Extracts cohort data based on specified prediction tasks and ICU status.

    Attributes:
        prediction_task (PredictionTask): The prediction task to be used for cohort extraction.
        cohort_output (Path): The path for the output of the cohort data.
    """

    def __init__(
        self,
        prediction_task: PredictionTask,
        cohort_output: str = None,
    ):
        self.prediction_task = prediction_task
        self.cohort_output = cohort_output

    def get_icu_status(self) -> str:
        """Determines the ICU status based on the prediction task."""
        return "ICU" if self.prediction_task.use_icu else "Non-ICU"

    def generate_extract_log(self) -> str:
        """Generates a log message for the extraction process."""
        icu_log = self.get_icu_status()
        task_info = f"{icu_log} | {self.prediction_task.target_type}"
        if self.prediction_task.disease_readmission:
            task_info += f" DUE TO {self.prediction_task.disease_readmission}"

        if self.prediction_task.disease_selection:
            task_info += f" ADMITTED DUE TO {self.prediction_task.disease_selection}"

        return f"EXTRACTING FOR: {task_info} | {self.prediction_task.nb_days} |".upper()

    def generate_output_suffix(self) -> str:
        """Generates a suffix for the output file based on the task details."""
        return (
            self.get_icu_status()  # .lower()
            + "_"
            + self.prediction_task.target_type.lower().replace(" ", "_")
            + "_"
            + str(self.prediction_task.nb_days)
            + "_"
            + (
                self.prediction_task.disease_readmission
                if self.prediction_task.disease_readmission
                else ""
            )
        )

    def fill_outputs(self) -> None:
        """Fills in the output details based on the prediction task."""
        disease_selection = (
            f"_{self.prediction_task.disease_selection}"
            if self.prediction_task.disease_selection
            else ""
        )
        self.cohort_output = (
            self.cohort_output
            or f"cohort_{self.generate_output_suffix()}{disease_selection}"
        )

    def load_hospital_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads hospital patient and admission data."""
        return load_hosp_patients(), load_hosp_admissions()

    def create_visits(self, hosp_patients, hosp_admissions):
        if self.prediction_task.use_icu:
            icu_icustays = load_icu_icustays()
            return make_icu_visits(
                icu_icustays, hosp_patients, self.prediction_task.target_type
            )
        else:
            return make_no_icu_visits(hosp_admissions, self.prediction_task.target_type)

    def filter_and_merge_visits(
        self,
        visits: pd.DataFrame,
        hosp_patients: pd.DataFrame,
        hosp_admissions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filters and merges visit records with patient and admission data."""
        visits = filter_visits(
            visits,
            self.prediction_task.disease_readmission,
            self.prediction_task.disease_selection,
        )
        patients_data = make_patients(hosp_patients)
        patients_filtered = patients_data.loc[patients_data["age"] >= 18]
        admissions_info = hosp_admissions[
            [
                HospAdmissions.HOSPITAL_ADMISSION_ID,
                HospAdmissions.INSURANCE,
                HospAdmissions.RACE,
            ]
        ]
        visits = visits.merge(patients_filtered, on=CohortHeader.PATIENT_ID)
        visits = visits.merge(admissions_info, on=CohortHeader.HOSPITAL_ADMISSION_ID)
        return visits

    def extract(self) -> Cohort:
        """
        Extracts the cohort data based on specified criteria and saves it.

        Returns:
            Cohort: The extracted and processed cohort data.
        """
        logger.info("===========MIMIC-IV v2.0============")
        self.fill_outputs()
        logger.info(self.generate_extract_log())

        hosp_patients, hosp_admissions = self.load_hospital_data()
        visits = self.create_visits(hosp_patients, hosp_admissions)
        visits = self.filter_and_merge_visits(visits, hosp_patients, hosp_admissions)
        self.fill_outputs()
        cohort = Cohort(
            icu=self.prediction_task.use_icu,
            name=self.cohort_output,
        )
        cohort.prepare_labels(visits, self.prediction_task)
        cohort.save()
        cohort.save_summary()
        return cohort
