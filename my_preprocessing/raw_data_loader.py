from pathlib import Path
import pandas as pd
import my_preprocessing.disease_cohort as disease_cohort
import datetime
import logging
import numpy as np
from tqdm import tqdm
from my_preprocessing.raw_files import (
    load_hosp_patients,
    load_hosp_admissions,
    load_icu_icustays,
)

logger = logging.getLogger()


class RawDataLoader:
    def __init__(
        self,
        use_icu: bool,
        label: str,
        time: int,
        icd_code: str,
        preproc_dir: Path,
        disease_label: str,
        cohort_output: Path,
        summary_output: Path,
    ):
        self.use_icu = use_icu
        self.label = label
        self.time = time
        self.icd_code = icd_code
        self.preproc_dir = preproc_dir
        self.disease_label = disease_label
        self.cohort_output = cohort_output
        self.summary_output = summary_output

        self.visit_col = "stay_id" if use_icu else "hadm_id"
        self.admit_col = "intime" if use_icu else "admittime"
        self.dish_col = "outtime" if use_icu else "hadm_id"
        self.admit_col = "intime" if use_icu else "dischtime"
        self.adm_visit_col = "hadm_id" if use_icu else ""

    def generate_icu_log(self) -> str:
        return "ICU" if self.use_icu else "Non-ICU"

    # LOG

    def generate_extract_log(self) -> str:
        if self.icd_code == "No Disease Filter":
            if len(self.disease_label):
                return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.label.upper()} DUE TO {self.disease_label.upper()} | {str(self.time)} | "
            return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.label.upper()} | {str(self.time)} |"
        else:
            if len(self.disease_label):
                return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.label.upper()} DUE TO {self.disease_label.upper()} | ADMITTED DUE TO {self.icd_code.upper()} | {str(self.time)} |"
        return f"EXTRACTING FOR: | {self.generate_icu_log()} | {self.label.upper()} | ADMITTED DUE TO {self.icd_code.upper()} | {str(self.time)} |"

    def generate_output_suffix(self) -> str:
        return (
            self.generate_icu_log()
            + "_"
            + self.label.lower().replace(" ", "_")
            + "_"
            + str(self.time)
            + "_"
            + self.disease_label
        )

    def fill_outputs(self) -> None:
        if not self.cohort_output:
            self.cohort_output = "cohort_" + self.generate_output_suffix()
        if not self.summary_output:
            self.summary_output = "summary_" + self.generate_output_suffix()

    # COLUMNS

    def get_visits_columns(self) -> list:
        # Return the list of columns to be used for visits data
        if self.use_icu:
            return ["subject_id", "stay_id", "hadm_id", "intime", "outtime", "los"]
        return ["subject_id", "hadm_id", "admittime", "dischtime", "los"]

    def get_patients_columns(self) -> list:
        # Return the list of columns to be used for patients data
        return [
            "subject_id",
            "anchor_year",
            "anchor_age",
            "min_valid_year",
            "dod",
            "gender",
        ]

    def get_eth_columns(self) -> list:
        # Return the list of columns to be used for patients data
        return ["hadm_id", "insurance", "race"]

    # VISITS AND PATIENTS

    def load_no_icu_visits(self) -> pd.DataFrame:
        hosp_admissions = load_hosp_admissions()
        dischtimes = hosp_admissions["dischtime"]
        admittimes = hosp_admissions["admittime"]
        hosp_admissions["los"] = dischtimes - admittimes
        hosp_admissions["admittime"] = pd.to_datetime(admittimes)
        hosp_admissions["dischtime"] = pd.to_datetime(dischtimes)

        # simplify....
        hosp_admissions["los"] = pd.to_timedelta(dischtimes - admittimes, unit="h")
        hosp_admissions["los"] = hosp_admissions["los"].astype(str)
        hosp_admissions["los"] = pd.to_numeric(
            hosp_admissions["los"].str.split(expand=True)[0]
        )
        if self.label == "Readmission":
            # remove hospitalizations with a death; impossible for readmission for such visits
            hosp_admissions = hosp_admissions.loc[
                hosp_admissions.hospital_expire_flag == 0
            ]

            print("[ READMISSION DUE TO " + self.disease_label + " ]")
        return hosp_admissions

    def load_icu_visits(self) -> pd.DataFrame:
        icu_icustays = load_icu_icustays()
        if self.label != "Readmission":
            return icu_icustays
        # icustays doesn't have a way to identify if patient died during visit; must
        # use core/patients to remove such stay_ids for readmission labels
        hosp_patient = load_hosp_patients()[["subject_id", "dod"]]
        visits = icu_icustays.merge(hosp_patient, how="inner", on="subject_id")
        visits = visits.loc[(visits.dod.isna()) | (visits["dod"] >= visits["outtime"])]
        return visits

    def load_visits(self) -> pd.DataFrame:
        return self.load_icu_visits() if self.use_icu else self.load_no_icu_visits()

    def load_patients(self) -> pd.DataFrame:
        hosp_patients = load_hosp_patients()[
            [
                "subject_id",
                "anchor_year",
                "anchor_age",
                "anchor_year_group",
                "dod",
                "gender",
            ]
        ]
        hosp_patients["min_valid_year"] = hosp_patients["anchor_year"] + (
            2019 - hosp_patients["anchor_year_group"].str.slice(start=-4).astype(int)
        )

        # Define anchor_year corresponding to the anchor_year_group 2017-2019. This is later used to prevent consideration
        # of visits with prediction windows outside the dataset's time range (2008-2019)
        return hosp_patients

    # PARTITION BY TARGET

    def partition_by_los(
        self,
        df: pd.DataFrame,
        los: int,
        group_col: str,
        admit_col: str,
        disch_col: str,
    ):
        valid_cohort = df.loc[
            ~df[admit_col].isna() & ~df[disch_col].isna() & ~df["los"].isna()
        ]
        valid_cohort["label"] = (valid_cohort["los"] > los).astype(int)
        sorted_cohort = valid_cohort.sort_values(by=[group_col, admit_col])
        logger.info("[ LOS LABELS FINISHED ]")
        return sorted_cohort

    def partition_by_readmit(
        self,
        df: pd.DataFrame,
        gap: pd.Timedelta,
        group_col: str,
        admit_col: str,
        disch_col: str,
    ):
        """Applies labels to individual visits according to whether or not a readmission has occurred within the specified `gap` days."""

        df_sorted = df.sort_values(by=[group_col, admit_col])

        # Calculate the time difference between consecutive visits for each patient
        df_sorted["next_admit"] = df_sorted.groupby(group_col)[admit_col].shift(-1)
        df_sorted["time_to_next"] = df_sorted["next_admit"] - df_sorted[disch_col]

        # Identify readmission cases
        df_sorted["readmit"] = df_sorted["time_to_next"].notnull() & (
            df_sorted["time_to_next"] <= gap
        )

        # Séparer en deux groupes: cas de réadmission et contrôles
        case = df_sorted[df_sorted["readmit"]]
        ctrl = df_sorted[~df_sorted["readmit"]]

        print("[ READMISSION LABELS FINISHED ]")
        return case.drop(columns=["next_admit", "time_to_next", "readmit"]), ctrl.drop(
            columns=["next_admit", "time_to_next", "readmit"]
        )

    def partition_by_mort(
        self,
        dataframe: pd.DataFrame,
        group_col: str,
        admit_col: str,
        discharge_col: str,
        death_col: str,
    ):
        """Applies labels to individual visits according to whether a death has occurred
        between the specified admit_col and disch_col times."""
        valid_entries = dataframe.loc[
            ~dataframe[admit_col].isna() & ~dataframe[discharge_col].isna()
        ]
        valid_entries[death_col] = pd.to_datetime(valid_entries[death_col])
        valid_entries["label"] = np.where(
            (valid_entries[death_col] >= valid_entries[admit_col])
            & (valid_entries[death_col] <= valid_entries[discharge_col])
            & (~valid_entries[death_col].isna()),
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
        """Handles logic for creating the labelled cohort based on arguments passed to extract().

        Parameters:
        df: dataframe with patient data
        gap: specified time interval gap for readmissions
        group_col: patient identifier to group patients (normally subject_id)
        visit_col: visit identifier for individual patient visits (normally hadm_id or stay_id)
        admit_col: column for visit start date information (normally admittime or intime)
        disch_col: column for visit end date information (normally dischtime or outtime)
        valid_col: generated column containing a patient's year that corresponds to the 2017-2019 anchor time range
        dod_col: Date of death column
        """
        # breakpoint()
        if self.label == "Mortality":
            return self.partition_by_mort(
                df, group_col, admit_col, disch_col, death_col
            )
        if self.label == "Readmission":
            gap = datetime.timedelta(days=gap)
            # transform gap into a timedelta to compare with datetime columns
            case, ctrl = self.partition_by_readmit(
                df, gap, group_col, admit_col, disch_col
            )

            # case hadm_ids are labelled 1 for readmission, ctrls have a 0 label
            case["label"] = np.ones(case.shape[0]).astype(int)
            ctrl["label"] = np.zeros(ctrl.shape[0]).astype(int)

            return pd.concat([case, ctrl], axis=0)
        if self.label == "Length of Stay":
            return self.partition_by_los(df, gap, group_col, admit_col, disch_col)

    # TODO: SAVE
    def save_cohort(self, cohort: pd.DataFrame) -> None:
        # Save the processed cohort data
        # cohort[cols].to_csv(
        #     root_dir + "/data/cohort/" + cohort_output + ".csv.gz",
        #     index=False,
        #     compression="gzip",
        # )
        print("not saved yet")

    def filter_visits(self, visits):
        if len(self.disease_label):
            hids = disease_cohort.preproc_icd_module(self.disease_label)
            visits = visits[visits["hadm_id"].isin(hids["hadm_id"])]
            print("[ READMISSION DUE TO " + self.disease_label + " ]")

        if self.icd_code != "No Disease Filter":
            hids = disease_cohort.preproc_icd_module(self.icd_code)
            visits = visits[visits["hadm_id"].isin(hids["hadm_id"])]
            self.cohort_output = self.cohort_output + "_" + self.icd_code
            self.summary_output = self.summary_output + "_" + self.icd_code
        return visits

    def extract(self) -> None:
        logger.info("===========MIMIC-IV v2.0============")
        self.fill_outputs()
        logger.info(self.generate_extract_log())

        visits = self.load_visits()[self.get_visits_columns()]
        visits = self.filter_visits(visits)

        patients = self.load_patients()[self.get_patients_columns()]
        patients["age"] = patients["anchor_age"]
        patients = patients.loc[patients["age"] >= 18]
        visits_patients = visits.merge(patients, how="inner", on="subject_id")

        eth = load_hosp_admissions()[self.get_eth_columns()]
        visits_patients = visits_patients.merge(eth, how="inner", on="hadm_id")

        cols_to_keep = [
            "subject_id",
            "hadm_id",
            "los",
            "min_valid_year",
            "dod",
            "age",
            "gender",
            "race",
            "insurance",
        ]
        if self.use_icu:
            cols_to_keep = cols_to_keep + ["stay_id", "intime", "outtime"]
        else:
            cols_to_keep = cols_to_keep + ["admittime", "dischtime"]
        visits_patients = visits_patients[cols_to_keep]

        cohort = self.get_case_ctrls(
            df=visits_patients,
            gap=self.time,
            group_col="subject_id",
            admit_col="intime" if self.use_icu else "admittime",
            disch_col="outtime" if self.use_icu else "dischtime",
            death_col="dod",
        )

        cohort = cohort.rename(columns={"race": "ethnicity"})

        self.save_cohort(cohort)
        logger.info("[ COHORT SUCCESSFULLY SAVED ]")
        logger.info(self.cohort_output)
        return cohort
