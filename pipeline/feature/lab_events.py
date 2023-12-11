from tqdm import tqdm
from pipeline.preprocessing.admission_imputer import (
    INPUTED_HOSPITAL_ADMISSION_ID_HEADER,
    impute_hadm_ids,
)
from pipeline.feature.feature_abc import Feature, Name
import logging
import pandas as pd
from pipeline.preprocessing.outlier_removal import outlier_imputation
from pipeline.file_info.preproc.feature import LabEventsHeader
from pipeline.file_info.preproc.cohort import CohortHeader, NonIcuCohortHeader
from pipeline.file_info.raw.hosp import (
    HospAdmissions,
    HospLabEvents,
    load_hosp_admissions,
    load_hosp_lab_events,
)
from pipeline.file_info.common import save_data
from pipeline.conversion.uom import drop_wrong_uom

logger = logging.getLogger()


class Lab(Feature):
    def name() -> str:
        return Name.LAB

    def __init__(self, df: pd.DataFrame = pd.DataFrame(), chunksize: int = 10000000):
        self.df = df
        self.chunksize = chunksize
        self.final_df = pd.DataFrame()

    def df(self):
        return self.df

    def extract_from(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Process and transform lab events data."""
        logger.info("[EXTRACTING LABS DATA]")
        admissions = load_hosp_admissions()[
            [
                HospAdmissions.PATIENT_ID,
                HospAdmissions.ID,
                HospAdmissions.ADMITTIME,
                HospAdmissions.DISCHTIME,
            ]
        ]
        usecols = [
            HospLabEvents.ITEM_ID,
            HospLabEvents.PATIENT_ID,
            HospLabEvents.HOSPITAL_ADMISSION_ID,
            HospLabEvents.CHART_TIME,
            HospLabEvents.VALUE_NUM,
            HospLabEvents.VALUE_UOM,
        ]
        processed_chunks = [
            self.process_lab_chunk(chunk, admissions, cohort)
            for chunk in tqdm(
                load_hosp_lab_events(chunksize=self.chunksize, use_cols=usecols)
            )
        ]
        labevents = pd.concat(processed_chunks, ignore_index=True)
        labevents = labevents[[h.value for h in LabEventsHeader]]
        self.df = labevents
        return labevents

    def process_lab_chunk(
        self, chunk: pd.DataFrame, admissions: pd.DataFrame, cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """Process a single chunk of lab events."""
        chunk = chunk.dropna(subset=[HospLabEvents.VALUE_NUM]).fillna(
            {HospLabEvents.VALUE_UOM: 0}
        )
        chunk = chunk[
            chunk[LabEventsHeader.PATIENT_ID].isin(cohort[CohortHeader.PATIENT_ID])
        ]
        chunk_with_hadm, chunk_no_hadm = (
            chunk[chunk[HospLabEvents.HOSPITAL_ADMISSION_ID].notna()],
            chunk[chunk[HospLabEvents.HOSPITAL_ADMISSION_ID].isna()],
        )
        chunk_imputed = impute_hadm_ids(chunk_no_hadm.copy(), admissions)
        chunk_imputed[HospLabEvents.HOSPITAL_ADMISSION_ID] = chunk_imputed[
            INPUTED_HOSPITAL_ADMISSION_ID_HEADER
        ]
        chunk_imputed = chunk_imputed[
            [
                HospLabEvents.PATIENT_ID,
                HospLabEvents.HOSPITAL_ADMISSION_ID,
                HospLabEvents.ITEM_ID,
                HospLabEvents.CHART_TIME,
                HospLabEvents.VALUE_NUM,
                HospLabEvents.VALUE_UOM,
            ]
        ]
        merged_chunk = pd.concat([chunk_with_hadm, chunk_imputed], ignore_index=True)
        return self.merge_with_cohort_and_calculate_lab_time(merged_chunk, cohort)

    # in utils?
    def merge_with_cohort_and_calculate_lab_time(
        self, chunk: pd.DataFrame, cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge chunk with cohort data and calculate the lab time from admit time."""
        chunk = chunk.merge(
            cohort[
                [
                    CohortHeader.HOSPITAL_ADMISSION_ID,
                    NonIcuCohortHeader.ADMIT_TIME,
                    NonIcuCohortHeader.DISCH_TIME,
                ]
            ],
            on=LabEventsHeader.HOSPITAL_ADMISSION_ID,
        )
        chunk[LabEventsHeader.CHART_TIME] = pd.to_datetime(
            chunk[LabEventsHeader.CHART_TIME]
        )
        chunk[LabEventsHeader.LAB_TIME_FROM_ADMIT] = (
            chunk[LabEventsHeader.CHART_TIME] - chunk[LabEventsHeader.ADMIT_TIME]
        )
        return chunk.dropna()

    def preproc(self):
        pass

    def impute_outlier(self, impute, thresh, left_thresh):
        print("[PROCESSING LABS DATA]")
        self.df = outlier_imputation(
            self.df,
            HospLabEvents.ITEM_ID,
            HospLabEvents.VALUE_NUM,
            thresh,
            left_thresh,
            impute,
        )
        print("Total number of rows", self.df.shape[0])
        print("[SUCCESSFULLY SAVED LABS DATA]")
        return self.df

    def summary(self):
        labs: pd.DataFrame = self.df
        freq = (
            labs.groupby(
                [LabEventsHeader.HOSPITAL_ADMISSION_ID, LabEventsHeader.ITEM_ID]
            )
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = (
            freq.groupby([LabEventsHeader.ITEM_ID])["mean_frequency"]
            .mean()
            .reset_index()
        )

        missing = (
            labs[labs[LabEventsHeader.VALUE_NUM] == 0]
            .groupby(LabEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="missing_count")
        )
        total = (
            labs.groupby(LabEventsHeader.ITEM_ID).size().reset_index(name="total_count")
        )
        summary = pd.merge(missing, total, on=LabEventsHeader.ITEM_ID, how="right")
        summary = pd.merge(freq, summary, on=LabEventsHeader.ITEM_ID, how="right")
        summary["missing%"] = 100 * (summary["missing_count"] / summary["total_count"])
        summary = summary.fillna(0)

        return summary

    def generate_fun(self, cohort):
        processed_chunks = []
        for labs in tqdm(self.df):
            labs = labs[labs["hadm_id"].isin(cohort["hadm_id"])].copy()
            # Process 'lab_time_from_admit' to numeric total hours
            time_parts = labs["lab_time_from_admit"].str.extract(
                r"(\d+) days (\d+):(\d+):(\d+)"
            )
            labs["start_time"] = pd.to_numeric(time_parts[0]) * 24 + pd.to_numeric(
                time_parts[1]
            )
            labs = pd.merge(labs, cohort[["hadm_id", "los"]], on="hadm_id", how="left")
            labs = labs[labs["los"] - labs["start_time"] > 0]
            labs = labs.drop(columns=["lab_time_from_admit", "los"])
            processed_chunks.append(labs)
        final = pd.concat(processed_chunks, ignore_index=True)
        self.df = final
        return final

    def mortality_length(self, cohort, include_time):
        self.df = self.df[self.df["hadm_id"].isin(cohort["hadm_id"])]
        self.df = self.df[self.df["start_time"] <= include_time]

    def los_length(self, cohort, include_time):
        self.df = self.df[self.df["hadm_id"].isin(cohort["hadm_id"])]
        self.df = self.df[self.df["start_time"] <= include_time]

    def read_length(self, cohort):
        self.df = self.df[self.df["hadm_id"].isin(cohort["hadm_id"])]

    def smooth_meds_step(self, bucket, i, t):
        sub_labs = (
            self.df[(self.df["start_time"] >= i) & (self.df["start_time"] < i + bucket)]
            .groupby(["hadm_id", "itemid"])
            .agg({"subject_id": "max", "valuenum": "mean"})
        )
        sub_labs = sub_labs.reset_index()
        sub_labs["start_time"] = t
        return sub_labs

    # def smooth_meds(self):
    #     f2_df = self.final_df.groupby(["hadm_id", "itemid"]).size()
    #     df_per_adm = f2_df.groupby("hadm_id").sum().reset_index()[0].max()
    #     dflength_per_adm = self.final_df.groupby("hadm_id").size().max()
    #     return f2_df, df_per_adm, dflength_per_adm

    # def dict_step(self, hid, los, dataDic):
    #     feat = self.final_df["itemid"].unique()
    #     df2 = self.final_df[self.final_df["hadm_id"] == hid]
    #     if df2.shape[0] == 0:
    #         val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
    #         val = val.fillna(0)
    #         val.columns = pd.MultiIndex.from_product([["LAB"], val.columns])
    #     else:
    #         val = df2.pivot_table(
    #             index="start_time", columns="itemid", values="valuenum"
    #         )
    #         df2["val"] = 1
    #         df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
    #         # print(df2.shape)
    #         add_indices = pd.Index(range(los)).difference(df2.index)
    #         add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
    #         df2 = pd.concat([df2, add_df])
    #         df2 = df2.sort_index()
    #         df2 = df2.fillna(0)

    #         val = pd.concat([val, add_df])
    #         val = val.sort_index()
    #         if self.impute == "Mean":
    #             val = val.ffill()
    #             val = val.bfill()
    #             val = val.fillna(val.mean())
    #         elif self.impute == "Median":
    #             val = val.ffill()
    #             val = val.bfill()
    #             val = val.fillna(val.median())
    #         val = val.fillna(0)

    #         df2[df2 > 0] = 1
    #         df2[df2 < 0] = 0

    #         # print(df2.head())
    #         dataDic[hid]["Lab"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
    #         dataDic[hid]["Lab"]["val"] = val.iloc[:, 0:].to_dict(orient="list")

    #         feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))
    #         val = pd.concat([val, feat_df], axis=1)

    #         val = val[feat]
    #         val = val.fillna(0)
    #         val.columns = pd.MultiIndex.from_product([["LAB"], val.columns])
    #     return val
