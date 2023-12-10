from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
import numpy as np
from pipeline.file_info.preproc.feature import (
    OutputEventsHeader,
)
from pipeline.file_info.preproc.cohort import IcuCohortHeader
from pipeline.file_info.raw.icu import load_icu_output_events, OuputputEvents

logger = logging.getLogger()


class OutputEvents(Feature):
    def __init__(self, df: pd.DataFrame = pd.DataFrame()):
        self.df = df
        self.final_df = pd.DataFrame()

    def df(self):
        return self.df

    def extract_from(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Function for getting hosp observations pertaining to a pickled cohort.
        Function is structured to save memory when reading and transforming data."""
        logger.info("[EXTRACTING OUTPUT EVENTS DATA]")
        raw_out = load_icu_output_events()
        out = raw_out.merge(
            cohort[
                [
                    IcuCohortHeader.STAY_ID,
                    IcuCohortHeader.IN_TIME,
                    IcuCohortHeader.OUT_TIME,
                ]
            ],
            on=IcuCohortHeader.STAY_ID,
        )
        out[OutputEventsHeader.EVENT_TIME_FROM_ADMIT] = (
            out[OuputputEvents.CHART_TIME] - out[IcuCohortHeader.IN_TIME]
        )
        out = out.dropna()

        # Print unique counts and value_counts
        logger.info(f"# Unique Events: {out[OuputputEvents.ITEM_ID].nunique()}")
        logger.info(f"# Admissions: {out[OuputputEvents.STAY_ID].nunique()}")
        logger.info(f"Total rows: {out.shape[0]}")
        out = out[[h.value for h in OutputEventsHeader]]
        self.df = out
        return out

    def preproc(self):
        pass

    def summary(self):
        out: pd.DataFrame = self.df
        freq = (
            out.groupby([OutputEventsHeader.STAY_ID, OutputEventsHeader.ITEM_ID])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = (
            out.groupby(OutputEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(freq, total, on=OutputEventsHeader.ITEM_ID, how="right")
        summary = summary.fillna(0)
        return summary

    def generate_fun(self, cohort):
        """
        Processes event times in the data, adjusting based on the cohort stay_id and length of stay (los).
        """
        out: pd.DataFrame = self.df[self.df["stay_id"].isin(cohort["stay_id"])].copy()
        time_split = out["event_time_from_admit"].str.extract(
            r"(\d+) (\d+):(\d+):(\d+)"
        )
        out["start_time"] = pd.to_numeric(time_split[0]) * 24 + pd.to_numeric(
            time_split[1]
        )
        # Removing entries where event time is after discharge time
        out = out.merge(cohort[["stay_id", "los"]], on="stay_id", how="left")
        out = out[out["los"] - out["start_time"] > 0]
        out = out.drop(columns=["los", "event_time_from_admit"])

        self.df = out
        return out

    # def mortality_length(self, include_time):
    #     self.df = self.df[self.df["stay_id"].isin(self.cohort["stay_id"])]
    #     self.df = self.df[self.df["start_time"] <= include_time]

    # def los_length(self, include_time):
    #     self.df = self.df[self.df["stay_id"].isin(self.cohort["stay_id"])]
    #     self.df = self.df[self.df["start_time"] <= include_time]

    # def read_length(self):
    #     self.df = self.df[self.df["stay_id"].isin(self.cohort["stay_id"])]
    #     self.df = pd.merge(
    #         self.df, self.cohort[["stay_id", "select_time"]], on="stay_id", how="left"
    #     )
    #     self.df["start_time"] = self.df["start_time"] - self.df["select_time"]
    #     self.df = self.df[self.df["start_time"] >= 0]

    # def smooth_meds_step(self, bucket, i, t):
    #     sub_out = (
    #         self.out[
    #             (self.out["start_time"] >= i) & (self.out["start_time"] < i + bucket)
    #         ]
    #         .groupby(["stay_id", "itemid"])
    #         .agg({"subject_id": "max"})
    #     )
    #     sub_out = sub_out.reset_index()
    #     sub_out["start_time"] = t
    #     if self.final_df.empty:
    #         self.final_df = sub_out
    #     else:
    #         self.final_df = self.final_df.append(sub_out)

    # def smooth_meds(self):
    #     f2_df = self.final_df.groupby(["stay_id", "itemid"]).size()
    #     df_per_adm = f2_df.groupby("stay_id").sum().reset_index()[0].max()
    #     dflength_per_adm = self.final_df.groupby("stay_id").size().max()
    #     return f2_df, df_per_adm, dflength_per_adm

    # def dict_step(self, hid, los, dataDic):
    #     feat = self.final_df["itemid"].unique()
    #     df2 = self.final_df[self.final_df["stay_id"] == hid]
    #     if df2.shape[0] == 0:
    #         df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
    #         df2 = df2.fillna(0)
    #         df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])
    #     else:
    #         df2["val"] = 1
    #         df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
    #         # print(df2.shape)
    #         add_indices = pd.Index(range(los)).difference(df2.index)
    #         add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
    #         df2 = pd.concat([df2, add_df])
    #         df2 = df2.sort_index()
    #         df2 = df2.fillna(0)
    #         df2[df2 > 0] = 1
    #         # print(df2.head())
    #         dataDic[hid]["Out"] = df2.to_dict(orient="list")

    #         feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
    #         df2 = pd.concat([df2, feat_df], axis=1)

    #         df2 = df2[feat]
    #         df2 = df2.fillna(0)
    #         df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])
    #         return df2
