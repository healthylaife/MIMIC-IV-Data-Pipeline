from pathlib import Path
import pandas as pd
import my_preprocessing.disease_cohort as disease_cohort
import datetime
import logging
import numpy as np
from tqdm import tqdm
from my_preprocessing.raw_header import (
    RAW_PATH,
    load_hosp_patients,
    load_hosp_admissions,
    load_icu_icustays,
)

logger = logging.getLogger()


# TODO
# CLARIFY LOG
# EXPLICIT OPTION CONSEQUENCE ICU... option =-> get columns...
# TRANSFORM  AND ENRICH
# simplify diseases cohort


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

    # RAW EXTRACT

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
        if len(self.disease_label):
            hids = disease_cohort.extract_diag_cohort(self.disease_label, RAW_PATH)
            hosp_admissions = hosp_admissions[
                hosp_admissions["hadm_id"].isin(hids["hadm_id"])
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
        if len(self.disease_label):
            hids = disease_cohort.extract_diag_cohort(self.disease_label, RAW_PATH)
            visits = visits[visits["hadm_id"].isin(hids["hadm_id"])]
            print("[ READMISSION DUE TO " + self.disease_label + " ]")
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
        invalid = df.loc[
            (df[admit_col].isna()) | (df[disch_col].isna()) | (df["los"].isna())
        ]
        cohort = df.loc[
            (~df[admit_col].isna()) & (~df[disch_col].isna()) & (~df["los"].isna())
        ]

        pos_cohort = cohort[cohort["los"] > los]
        neg_cohort = cohort[cohort["los"] <= los]
        neg_cohort = neg_cohort.fillna(0)
        pos_cohort = pos_cohort.fillna(0)

        pos_cohort["label"] = 1
        neg_cohort["label"] = 0

        cohort = pd.concat([pos_cohort, neg_cohort], axis=0)
        cohort = cohort.sort_values(by=[group_col, admit_col])
        print("[ LOS LABELS FINISHED ]")
        return cohort, invalid

    def partition_by_readmit(
        self,
        df: pd.DataFrame,
        gap: datetime.timedelta,
        group_col: str,
        admit_col: str,
        disch_col: str,
    ):
        """Applies labels to individual visits according to whether or not a readmission has occurred within the specified `gap` days.
        For a given visit, another visit must occur within the gap window for a positive readmission label.
        The gap window starts from the disch_col time and the admit_col of subsequent visits are considered.
        """

        case = pd.DataFrame()  # hadm_ids with readmission within the gap period
        ctrl = pd.DataFrame()  # hadm_ids without readmission within the gap period
        invalid = pd.DataFrame()  # hadm_ids that are not considered in the cohort

        # Iterate through groupbys based on group_col (subject_id). Data is sorted by subject_id and admit_col (admittime)
        # to ensure that the most current hadm_id is last in a group.
        # grouped= df[[group_col, visit_col, admit_col, disch_col, valid_col]].sort_values(by=[group_col, admit_col]).groupby(group_col)
        grouped = df.sort_values(by=[group_col, admit_col]).groupby(group_col)
        for subject, group in tqdm(grouped):
            if group.shape[0] <= 1:
                ctrl = pd.concat(
                    [ctrl, pd.DataFrame([group.iloc[0]])], ignore_index=True
                )
            else:
                for idx in range(group.shape[0] - 1):
                    visit_time = group.iloc[idx][
                        disch_col
                    ]  # For each index (a unique hadm_id), get its timestamp
                    if (
                        group.loc[
                            (group[admit_col] > visit_time)
                            & (  # Readmissions must come AFTER the current timestamp
                                group[admit_col] - visit_time <= gap
                            )  # Distance between a timestamp and readmission must be within gap
                        ].shape[0]
                        >= 1
                    ):  # If ANY rows meet above requirements, a readmission has occurred after that visit
                        case = pd.concat(
                            [case, pd.DataFrame([group.iloc[idx]])], ignore_index=True
                        )
                    else:
                        # If no readmission is found, only add to ctrl if prediction window is guaranteed to be within the
                        # time range of the dataset (2008-2019). Visits with prediction windows existing in potentially out-of-range
                        # dates (like 2018-2020) are excluded UNLESS the prediction window takes place the same year as the visit,
                        # in which case it is guaranteed to be within 2008-2019

                        ctrl = pd.concat(
                            [ctrl, pd.DataFrame([group.iloc[idx]])], ignore_index=True
                        )

                ctrl = pd.concat(
                    [ctrl, pd.DataFrame([group.iloc[-1]])], ignore_index=True
                )

        print("[ READMISSION LABELS FINISHED ]")
        return case, ctrl, invalid

    def partition_by_mort(
        self,
        df: pd.DataFrame,
        group_col: str,
        admit_col: str,
        disch_col: str,
        death_col: str,
    ):
        """Applies labels to individual visits according to whether or not a death has occurred within
        the times of the specified admit_col and disch_col"""

        invalid = df.loc[(df[admit_col].isna()) | (df[disch_col].isna())]

        cohort = df.loc[(~df[admit_col].isna()) & (~df[disch_col].isna())]

        cohort["label"] = 0
        pos_cohort = cohort[~cohort[death_col].isna()]
        neg_cohort = cohort[cohort[death_col].isna()]
        neg_cohort = neg_cohort.fillna(0)
        pos_cohort = pos_cohort.fillna(0)
        pos_cohort[death_col] = pd.to_datetime(pos_cohort[death_col])

        pos_cohort["label"] = np.where(
            (pos_cohort[death_col] >= pos_cohort[admit_col])
            & (pos_cohort[death_col] <= pos_cohort[disch_col]),
            1,
            0,
        )

        pos_cohort["label"] = pos_cohort["label"].astype("Int32")
        cohort = pd.concat([pos_cohort, neg_cohort], axis=0)
        cohort = cohort.sort_values(by=[group_col, admit_col])
        print("[ MORTALITY LABELS FINISHED ]")
        return cohort, invalid

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

        case = None  # hadm_ids with readmission within the gap period
        ctrl = None  # hadm_ids without readmission within the gap period
        invalid = None  # hadm_ids that are not considered in the cohort
        if self.label == "Mortality":
            return self.partition_by_mort(
                df, group_col, admit_col, disch_col, death_col
            )
        elif self.label == "Readmission":
            gap = datetime.timedelta(days=gap)
            # transform gap into a timedelta to compare with datetime columns
            case, ctrl, invalid = self.partition_by_readmit(
                df, gap, group_col, admit_col, disch_col
            )

            # case hadm_ids are labelled 1 for readmission, ctrls have a 0 label
            case["label"] = np.ones(case.shape[0]).astype(int)
            ctrl["label"] = np.zeros(ctrl.shape[0]).astype(int)

            return pd.concat([case, ctrl], axis=0), invalid
        elif self.label == "Length of Stay":
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

    def extract(self) -> None:
        logger.info("===========MIMIC-IV v2.0============")
        self.fill_outputs()
        logger.info(self.generate_extract_log())
        visits = self.load_visits()[self.get_visits_columns()]
        patients = self.load_patients()[self.get_patients_columns()]

        visits_patients = visits.merge(patients, how="inner", on="subject_id")
        visits_patients["Age"] = visits_patients["anchor_age"]
        visits_patients = visits_patients.loc[visits_patients["Age"] >= 18]
        ##Add Demo data
        eth = load_hosp_admissions()[self.get_eth_columns()]
        visits_patients = visits_patients.merge(eth, how="inner", on="hadm_id")

        if self.use_icu:
            visits_patients = visits_patients[
                [
                    "subject_id",
                    "stay_id",
                    "hadm_id",
                    "intime",
                    "outtime",
                    "los",
                    "min_valid_year",
                    "dod",
                    "Age",
                    "gender",
                    "race",
                    "insurance",
                ]
            ]
        else:
            visits_patients = visits_patients.dropna(subset=["min_valid_year"])[
                [
                    "subject_id",
                    "hadm_id",
                    "admittime",
                    "dischtime",
                    "los",
                    "min_valid_year",
                    "dod",
                    "Age",
                    "gender",
                    "race",
                    "insurance",
                ]
            ]

        admit_col = "intime" if self.use_icu else "admittime"
        disch_col = "outtime" if self.use_icu else "dischtime"
        cohort, invalid = self.get_case_ctrls(
            visits_patients,
            self.time,
            "subject_id",
            admit_col,
            disch_col,
            "dod",
        )

        if self.icd_code != "No Disease Filter":
            hids = disease_cohort.extract_diag_cohort(self.icd_code, RAW_PATH)
            cohort = cohort[cohort["hadm_id"].isin(hids["hadm_id"])]
            self.cohort_output = self.cohort_output + "_" + self.icd_code
            self.summary_output = self.summary_output + "_" + self.icd_code
        cohort = cohort.rename(columns={"race": "ethnicity"})
        self.save_cohort(cohort)
        logger.info("[ COHORT SUCCESSFULLY SAVED ]")
        logger.info(self.cohort_output)
        return cohort
