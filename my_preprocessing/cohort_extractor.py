from pathlib import Path
import pandas as pd
import my_preprocessing.icd_conversion as icd_conversion
import datetime
import logging
import numpy as np
from my_preprocessing.raw_files import (
    load_hosp_patients,
    load_hosp_admissions,
    load_icu_icustays,
    HospPatients,
    IcuStays,
    HospAdmissions,
)
from my_preprocessing.preproc_files import COHORT_PATH
from my_preprocessing.prediction_task import PredictionTask, TargetType

logger = logging.getLogger()

MIN_VALID_YEAR_HEADER = "min_valid_year"


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

    def make_no_icu_visits(self) -> pd.DataFrame:
        hosp_admissions = load_hosp_admissions()
        hosp_admissions[HospAdmissions.LOS] = (
            hosp_admissions[HospAdmissions.DISCHTIME]
            - hosp_admissions[HospAdmissions.ADMITTIME]
        ).dt.days

        if self.prediction_task.target_type == TargetType.READMISSION:
            # remove hospitalizations with a death
            hosp_admissions = hosp_admissions[
                hosp_admissions[HospAdmissions.HOSPITAL_EXPIRE_FLAG] == 0
            ]
        return hosp_admissions[
            [
                HospAdmissions.PATIENT_ID,
                HospAdmissions.HOSPITAL_AMISSION_ID,
                HospAdmissions.ADMITTIME,
                HospAdmissions.DISCHTIME,
                HospAdmissions.LOS,
            ]
        ]

    def make_icu_visits(self) -> pd.DataFrame:
        icu_icustays = load_icu_icustays()
        if self.prediction_task.target_type != TargetType.READMISSION:
            return icu_icustays
        # remove such stay_ids with a death for readmission labels
        hosp_patient = load_hosp_patients()[[HospPatients.ID, HospPatients.DOD]]
        visits = icu_icustays.merge(hosp_patient, on=IcuStays.PATIENT_ID)
        visits = visits.loc[
            (visits[HospPatients.DOD].isna())
            | (visits[HospPatients.DOD] >= visits[IcuStays.OUTTIME])
        ]
        return visits[
            ["subject_id", "stay_id", "hadm_id", "intime", IcuStays.OUTTIME, "los"]
        ]

    def make_visits(self) -> pd.DataFrame:
        return (
            self.make_icu_visits()
            if self.prediction_task.use_icu
            else self.make_no_icu_visits()
        )

    def make_patients(self) -> pd.DataFrame:
        hosp_patients = load_hosp_patients()[
            [
                HospPatients.ID,
                HospPatients.ANCHOR_YEAR,
                HospPatients.ANCHOR_AGE,
                HospPatients.ANCHOR_YEAR_GROUP,
                HospPatients.DOD,
                HospPatients.GENDER,
            ]
        ]
        max_anchor_year_group = (
            hosp_patients[HospPatients.ANCHOR_YEAR_GROUP]
            .str.slice(start=-4)
            .astype(int)
        )
        hosp_patients[MIN_VALID_YEAR_HEADER] = (
            hosp_patients[HospPatients.ANCHOR_YEAR] + 2019 - max_anchor_year_group
        )

        hosp_patients["age"] = hosp_patients[HospPatients.ANCHOR_AGE]  # ro remove...
        # Define anchor_year corresponding to the anchor_year_group 2017-2019.
        # To identify visits with prediction windows outside the range 2008-2019.
        return hosp_patients[
            [
                HospPatients.ID,
                "age",
                MIN_VALID_YEAR_HEADER,
                HospPatients.DOD,
                HospPatients.GENDER,
            ]
        ]

    # PARTITION BY TARGET

    def partition_by_los(
        self,
        df: pd.DataFrame,
        los: int,
        group_col: str,
        admit_col: str,
        disch_col: str,
    ) -> pd.DataFrame:
        """
        Partition data based on length of stay (LOS).

        Parameters:
        df (pd.DataFrame): The dataframe to partition.
        los (int): Length of stay threshold.
        group_col (str): Column to group by.
        admit_col (str): Admission date column.
        disch_col (str): Discharge date column.
        """
        valid_cohort = df.dropna(subset=[admit_col, disch_col, "los"])
        valid_cohort["label"] = (valid_cohort["los"] > los).astype(int)
        return valid_cohort.sort_values(by=[group_col, admit_col])

    def partition_by_readmit(
        self,
        df: pd.DataFrame,
        gap: pd.Timedelta,
        group_col: str,
        admit_col: str,
        disch_col: str,
    ):
        """
        Partition data based on readmission within a specified gap.

        Parameters:
        df (pd.DataFrame): The dataframe to partition.
        gap (pd.Timedelta): Time gap to consider for readmission.
        group_col (str): Column to group by.
        admit_col (str): Admission date column.
        disch_col (str): Discharge date column.
        """

        df_sorted = df.sort_values(by=[group_col, admit_col])
        df_sorted["next_admit"] = df_sorted.groupby(group_col)[admit_col].shift(-1)
        df_sorted["time_to_next"] = df_sorted["next_admit"] - df_sorted[disch_col]
        # Identify readmission cases
        df_sorted["readmit"] = df_sorted["time_to_next"].notnull() & (
            df_sorted["time_to_next"] <= gap
        )
        temp_columns = ["next_admit", "time_to_next", "readmit"]
        case, ctrl = df_sorted[df_sorted["readmit"]], df_sorted[~df_sorted["readmit"]]
        case, ctrl = case.drop(columns=temp_columns), ctrl.drop(columns=temp_columns)
        case["label"], ctrl["label"] = np.ones(len(case)), np.zeros(len(ctrl))
        return pd.concat([case, ctrl], axis=0)

    def partition_by_mort(
        self,
        df: pd.DataFrame,
        group_col: str,
        admit_col: str,
        discharge_col: str,
        death_col: str,
    ):
        """
        Partition data based on mortality events occurring between admission and discharge.

        Parameters:
        df (pd.DataFrame): The dataframe to partition.
        group_col (str): Column to group by.
        admit_col (str): Admission date column.
        discharge_col (str): Discharge date column.
        death_col (str): Death date column.
        """
        valid_entries = df.dropna(subset=[admit_col, discharge_col])
        valid_entries[death_col] = pd.to_datetime(valid_entries[death_col])
        valid_entries["label"] = np.where(
            (valid_entries[death_col] >= valid_entries[admit_col])
            & (valid_entries[death_col] <= valid_entries[discharge_col]),
            1,
            0,
        )
        sorted_cohort = valid_entries.sort_values(by=[group_col, admit_col])
        logger.info("[ MORTALITY LABELS FINISHED ]")
        return sorted_cohort

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
            return self.partition_by_mort(
                df, group_col, admit_col, disch_col, death_col
            )
        elif self.prediction_task.target_type == TargetType.READMISSION:
            gap = datetime.timedelta(days=gap)
            return self.partition_by_readmit(df, gap, group_col, admit_col, disch_col)
        elif self.prediction_task.target_type == TargetType.LOS:
            return self.partition_by_los(df, gap, group_col, admit_col, disch_col)

    def filter_visits(self, visits):
        diag = icd_conversion.preproc_icd_module()
        if self.prediction_task.disease_readmission:
            hids = icd_conversion.get_pos_ids(
                diag, self.prediction_task.disease_readmission
            )
            visits = visits[visits["hadm_id"].isin(hids["hadm_id"])]
            logger.info(
                "[ READMISSION DUE TO "
                + self.prediction_task.disease_readmission
                + " ]"
            )

        if self.prediction_task.disease_selection:
            hids = icd_conversion.get_pos_ids(
                diag, self.prediction_task.disease_selection
            )
            visits = visits[visits["hadm_id"].isin(hids["hadm_id"])]

            self.cohort_output = (
                self.cohort_output + "_" + self.prediction_task.disease_selection
            )
            self.summary_output = (
                self.summary_output + "_" + self.prediction_task.disease_selection
            )
        return visits

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

        visits = self.make_visits()
        visits = self.filter_visits(visits)
        patients = self.make_patients()
        patients = patients.loc[patients["age"] >= 18]
        admissions_info = load_hosp_admissions()[["hadm_id", "insurance", "race"]]

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
