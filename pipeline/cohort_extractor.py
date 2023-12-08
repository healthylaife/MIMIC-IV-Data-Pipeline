from pathlib import Path
import pandas as pd
import datetime
import logging
from pipeline.file_info.raw.hosp import (
    load_hosp_patients,
    load_hosp_admissions,
    HospAdmissions,
)
from pipeline.file_info.raw.icu import load_icu_icustays
from pipeline.file_info.preproc.cohort import COHORT_PATH, CohortHeader
from pipeline.prediction_task import PredictionTask, TargetType

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
    def __init__(
        self,
        prediction_task: PredictionTask,
        cohort_output: Path = None,
    ):
        self.prediction_task = prediction_task
        self.cohort_output = cohort_output

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
            + (
                self.prediction_task.disease_readmission
                if self.prediction_task.disease_readmission
                else ""
            )
        )

    def fill_outputs(self) -> None:
        disease_selection = (
            f"_{self.prediction_task.disease_selection}"
            if self.prediction_task.disease_selection
            else ""
        )
        self.cohort_output = (
            self.cohort_output
            or f"cohort_{self.generate_output_suffix()}{disease_selection}"
        )

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

    def filter_and_merge_visits(self, visits, hosp_patients, hosp_admissions):
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
