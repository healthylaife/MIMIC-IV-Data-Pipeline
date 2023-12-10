from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
import numpy as np
from pipeline.conversion.ndc import (
    NdcMappingHeader,
    get_EPC,
    ndc_to_str,
    prepare_ndc_mapping,
)
from pipeline.file_info.preproc.feature import (
    MedicationsHeader,
    IcuMedicationHeader,
    NonIcuMedicationHeader,
    EXTRACT_MED_ICU_PATH,
    EXTRACT_MED_PATH,
    PREPROC_MED_ICU_PATH,
    PREPROC_MED_PATH,
    PreprocMedicationHeader,
)
from pipeline.file_info.preproc.cohort import (
    CohortHeader,
    IcuCohortHeader,
    NonIcuCohortHeader,
)
from pipeline.file_info.preproc.summary import MED_FEATURES_PATH, MED_SUMMARY_PATH
from pipeline.file_info.raw.hosp import (
    HospPrescriptions,
    load_hosp_prescriptions,
)
from pipeline.file_info.raw.icu import (
    InputEvents,
    load_input_events,
)
from pipeline.file_info.common import save_data
from pathlib import Path

logger = logging.getLogger()


class Medications(Feature):
    def __init__(
        self,
        use_icu: bool,
        df: pd.DataFrame = pd.DataFrame(),
        group_code: bool = False,
    ):
        self.use_icu = use_icu
        self.group_code = group_code
        self.df = df
        self.final_df = pd.DataFrame()

    def df(self) -> pd.DataFrame:
        return self.df

    def extract_from(self, cohort: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[EXTRACTING MEDICATIONS DATA]")
        cohort_headers = (
            [
                CohortHeader.HOSPITAL_ADMISSION_ID,
                IcuCohortHeader.STAY_ID,
                IcuCohortHeader.IN_TIME,
            ]
            if self.use_icu
            else [CohortHeader.HOSPITAL_ADMISSION_ID, NonIcuCohortHeader.ADMIT_TIME]
        )
        admissions = cohort[cohort_headers]
        raw_med = load_input_events() if self.use_icu else load_hosp_prescriptions()
        medications = raw_med.merge(
            admissions,
            on=IcuCohortHeader.STAY_ID
            if self.use_icu
            else CohortHeader.HOSPITAL_ADMISSION_ID,
        )
        admit_header = (
            IcuCohortHeader.IN_TIME if self.use_icu else NonIcuCohortHeader.ADMIT_TIME
        )

        medications[MedicationsHeader.START_HOURS_FROM_ADMIT] = (
            medications[InputEvents.STARTTIME] - medications[admit_header]
        )
        medications[MedicationsHeader.STOP_HOURS_FROM_ADMIT] = (
            medications[
                InputEvents.ENDTIME if self.use_icu else HospPrescriptions.STOP_TIME
            ]
            - medications[admit_header]
        )
        medications = (
            medications.dropna()
            if self.use_icu
            else self.normalize_non_icu(medications)
        )
        self.log_medication_stats(medications)
        cols = [h.value for h in MedicationsHeader] + [
            h.value
            for h in (IcuMedicationHeader if self.use_icu else NonIcuMedicationHeader)
        ]
        medications = medications[cols]
        self.df = medications
        return medications

    def normalize_non_icu(self, med: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize medication data for non-ICU cases.

        Args:
            med (pd.DataFrame): The medication dataframe.

        Returns:
            pd.DataFrame: The normalized dataframe.
        """
        med[NonIcuMedicationHeader.DRUG] = (
            med[NonIcuMedicationHeader.DRUG]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
        )
        med[HospPrescriptions.NDC] = med[HospPrescriptions.NDC].fillna(-1)
        med[HospPrescriptions.NDC] = med[HospPrescriptions.NDC].astype("Int64")
        med[NdcMappingHeader.NEW_NDC] = med[HospPrescriptions.NDC].apply(ndc_to_str)
        ndc_map = prepare_ndc_mapping()
        med = med.merge(ndc_map, on=NdcMappingHeader.NEW_NDC, how="left")
        med[NonIcuMedicationHeader.EPC] = med["pharm_classes"].apply(get_EPC)
        return med

    def log_medication_stats(self, med: pd.DataFrame) -> None:
        """
        Log statistics for medication data.

        Args:
            med (pd.DataFrame): The medication dataframe.
        """
        unique_drug_count = med[
            InputEvents.ITEMID if self.use_icu else NonIcuMedicationHeader.DRUG
        ].nunique()
        unique_admission_count = med[
            InputEvents.STAY_ID if self.use_icu else CohortHeader.HOSPITAL_ADMISSION_ID
        ].nunique()
        logger.info(f"Number of unique types of drugs: {unique_drug_count}")
        if not self.use_icu:
            logger.info(
                f"Number of unique type of drug after grouping: {med[NonIcuMedicationHeader.NON_PROPRIEATARY_NAME].nunique()}"
            )
        logger.info(f"Number of admissions: {unique_admission_count}")
        logger.info(f"Total number of rows: {med.shape[0]}")

    def preproc(self, group_code: bool):
        logger.info("[PROCESSING MEDICATIONS DATA]")
        self.df[PreprocMedicationHeader.DRUG_NAME] = (
            self.df[NonIcuMedicationHeader.NON_PROPRIEATARY_NAME]
            if group_code
            else self.df[NonIcuMedicationHeader.DRUG]
        )
        self.df = self.df.drop(
            columns=[
                NonIcuMedicationHeader.NON_PROPRIEATARY_NAME,
                NonIcuMedicationHeader.DRUG,
            ]
        )
        self.df.dropna()
        self.group_code = group_code
        logger.info(f"Total number of rows: {self.df.shape[0]}")
        return self.df

    def summary(self) -> pd.DataFrame:
        med: pd.DataFrame = self.df
        feature_name = (
            IcuMedicationHeader.ITEM_ID.value
            if self.use_icu
            else PreprocMedicationHeader.DRUG_NAME.value
        )
        group_columns = (
            [IcuMedicationHeader.STAY_ID, IcuMedicationHeader.ITEM_ID]
            if self.use_icu
            else [
                MedicationsHeader.HOSPITAL_ADMISSION_ID,
                PreprocMedicationHeader.DRUG_NAME,
            ]
        )
        freq = med.groupby(group_columns).size().reset_index(name="mean_frequency")
        amount_column = (
            IcuMedicationHeader.AMOUNT
            if self.use_icu
            else NonIcuMedicationHeader.DOSE_VAL_RX
        )
        missing = (
            med[med[amount_column] == 0]
            .groupby(feature_name)
            .size()
            .reset_index(name="missing_count")
        )
        total = med.groupby(feature_name).size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on=feature_name, how="right")
        summary = pd.merge(freq, summary, on=feature_name, how="right")
        summary["missing%"] = 100 * (summary["missing_count"] / summary["total_count"])
        summary = summary.fillna(0)
        summary.to_csv(MED_SUMMARY_PATH, index=False)
        summary[feature_name].to_csv(MED_FEATURES_PATH, index=False)
        return summary

    def generate_fun(self):
        meds: pd.DataFrame = self.df
        meds[["start_days", "dummy", "start_hours"]] = meds[
            "start_hours_from_admit"
        ].str.split(" ", expand=True)
        meds[["start_hours", "min", "sec"]] = meds["start_hours"].str.split(
            ":", -1, expand=True
        )
        meds["start_time"] = pd.to_numeric(meds["start_days"]) * 24 + pd.to_numeric(
            meds["start_hours"]
        )
        meds[["start_days", "dummy", "start_hours"]] = meds[
            "stop_hours_from_admit"
        ].str.split(" ", expand=True)
        meds[["start_hours", "min", "sec"]] = meds["start_hours"].str.split(
            ":", expand=True
        )
        meds["stop_time"] = pd.to_numeric(meds["start_days"]) * 24 + pd.to_numeric(
            meds["start_hours"]
        )
        meds = meds.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
        #####Sanity check
        meds["sanity"] = meds["stop_time"] - meds["start_time"]
        meds = meds[meds["sanity"] > 0]
        del meds["sanity"]
        #####Select hadm_id as in main file
        meds = meds[meds["hadm_id"].isin(self.data["hadm_id"])]
        meds = pd.merge(meds, self.data[["hadm_id", "los"]], on="hadm_id", how="left")

        #####Remove where start time is after end of visit
        meds["sanity"] = meds["los"] - meds["start_time"]
        meds = meds[meds["sanity"] > 0]
        del meds["sanity"]
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds["stop_time"] > meds["los"], "stop_time"] = meds.loc[
            meds["stop_time"] > meds["los"], "los"
        ]
        del meds["los"]

        meds["dose_val_rx"] = meds["dose_val_rx"].apply(pd.to_numeric, errors="coerce")
        self.df = meds
        return meds

    # def mortality_length(self, include_time):
    #     col = "stay_id" if self.use_icu else "hadm_id"
    #     self.df = self.df[self.df[col].isin(self.cohort[col])]
    #     self.df = self.df[self.df["start_time"] <= include_time]
    #     self.df.loc[self.df["stop_time"] > include_time, "stop_time"] = include_time

    # def los_length(self, include_time):
    #     col = "stay_id" if self.use_icu else "hadm_id"
    #     self.df = self.df[self.df[col].isin(self.cohort[col])]
    #     self.df = self.df[self.df["start_time"] <= include_time]
    #     self.df.loc[self.df["stop_time"] > include_time, "stop_time"] = include_time

    # def read_length(self):
    #     col = "stay_id" if self.use_icu else "hadm_id"
    #     self.df = self.df[self.df[col].isin(self.cohort[col])]
    #     self.df = pd.merge(
    #         self.df, self.cohort[[col, "select_time"]], on=col, how="left"
    #     )
    #     self.df["stop_time"] = self.df["stop_time"] - self.df["select_time"]

    #     self.df["start_time"] = self.df["start_time"] - self.df["select_time"]
    #     self.df = self.df[self.df["stop_time"] >= 0]
    #     self.df.loc[self.df["start_time"] < 0, "start_time"] = 0

    # def smooth_meds_step(self, bucket, i, t):
    #     sub_meds = (
    #         self.df[(self.df["start_time"] >= i) & (self.df["start_time"] < i + bucket)]
    #         .groupby(
    #             ["stay_id", "itemid", "orderid"]
    #             if self.use_icu
    #             else ["hadm_id", "drug_name"]
    #         )
    #         .agg(
    #             {
    #                 "stop_time": "max",
    #                 "subject_id": "max",
    #                 "rate": np.nanmean,
    #                 "amount": np.nanmean,
    #             }
    #             if self.use_icu
    #             else {
    #                 "stop_time": "max",
    #                 "subject_id": "max",
    #                 "dose_val_rx": np.nanmean,
    #             }
    #         )
    #     )
    #     sub_meds = sub_meds.reset_index()
    #     sub_meds["start_time"] = t
    #     sub_meds["stop_time"] = sub_meds["stop_time"] / bucket
    #     if self.final_df.empty:
    #         self.final_df = sub_meds
    #     else:
    #         self.final_df = self.final_df.append(sub_meds)

    # def smooth_meds(self):
    #     f2_df = self.final_df.groupby(
    #         ["stay_id", "itemid", "orderid"]
    #         if self.use_icd
    #         else ["hadm_id", "drug_name"]
    #     ).size()
    #     df_per_adm = (
    #         f2_df.groupby("stay_id" if self.use_icd else "hadm_id")
    #         .sum()
    #         .reset_index()[0]
    #         .max()
    #     )
    #     dflength_per_adm = (
    #         self.final_df.groupby("stay_id" if self.use_icd else "hadm_id").size().max()
    #     )
    #     return f2_df, df_per_adm, dflength_per_adm

    # def dict_step(self, hid, los, dataDic):
    #     feat = self.final_df["itemid" if self.use_icu else "drug_name"].unique()
    #     df2 = self.final_df[
    #         self.final_df["stay_id" if self.use_icu else "hadm_id"] == hid
    #     ]
    #     if df2.shape[0] == 0:
    #         val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
    #         val = val.fillna(0)
    #         val.columns = pd.MultiIndex.from_product([["MEDS"], val.columns])
    #     else:
    #         if self.use_icu:
    #             rate = df2.pivot_table(
    #                 index="start_time", columns="itemid", values="rate"
    #             )
    #             amount = df2.pivot_table(
    #                 index="start_time", columns="itemid", values="amount"
    #             )
    #         else:
    #             val = df2.pivot_table(
    #                 index="start_time", columns="drug_name", values="dose_val_rx"
    #             )

    #         df2 = df2.pivot_table(
    #             index="start_time",
    #             columns="itemid" if self.use_icu else "drug_name",
    #             values="stop_time",
    #         )
    #         add_indices = pd.Index(range(los)).difference(df2.index)
    #         add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
    #         df2 = pd.concat([df2, add_df])
    #         df2 = df2.sort_index()
    #         df2 = df2.ffill()
    #         df2 = df2.fillna(0)
    #         if self.use_icu:
    #             rate = pd.concat([rate, add_df])
    #             rate = rate.sort_index()
    #             rate = rate.ffill()
    #             rate = rate.fillna(-1)
    #             amount = pd.concat([amount, add_df])
    #             amount = amount.sort_index()
    #             amount = amount.ffill()
    #             amount = amount.fillna(-1)
    #         else:
    #             val = pd.concat([val, add_df])
    #             val = val.sort_index()
    #             val = val.ffill()
    #             val = val.fillna(-1)

    #         df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
    #         df2[df2 > 0] = 1
    #         df2[df2 < 0] = 0
    #         val.iloc[:, 0:] = df2.iloc[:, 0:] * val.iloc[:, 0:]
    #         # print(df2.head())
    #         if self.use_icu:
    #             dataDic.iloc[:, 0:].to_dict(orient="list")
    #             dataDic[hid]["Med"]["rate"] = rate.iloc[:, 0:].to_dict(orient="list")
    #             dataDic[hid]["Med"]["amount"] = amount.iloc[:, 0:].to_dict(
    #                 orient="list"
    #             )
    #         else:
    #             dataDic[hid]["Med"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
    #             dataDic[hid]["Med"]["val"] = val.iloc[:, 0:].to_dict(orient="list")

    #         feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))

    #         val = pd.concat([val, feat_df], axis=1)

    #         val = val[feat]
    #         val = val.fillna(0)

    #         val.columns = pd.MultiIndex.from_product([["MEDS"], val.columns])
    #     return val
