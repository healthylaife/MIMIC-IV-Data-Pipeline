from tqdm import tqdm
import numpy as np
from pipeline.feature.feature_abc import Feature
import logging
import pandas as pd
from pipeline.preprocessing.outlier_removal import outlier_imputation
from pipeline.file_info.preproc.feature import (
    EXTRACT_CHART_ICU_PATH,
    ChartEventsHeader,
)
from pipeline.file_info.preproc.cohort import IcuCohortHeader
from pipeline.file_info.preproc.summary import CHART_FEATURES_PATH, CHART_SUMMARY_PATH
from pipeline.file_info.raw.icu import (
    load_icu_chart_events,
    ChartEvents,
)
from pipeline.file_info.common import save_data
from pathlib import Path

from pipeline.conversion.uom import drop_wrong_uom

logger = logging.getLogger()


class Chart(Feature):
    def __init__(
        self,
        df: pd.DataFrame = pd.DataFrame(),
        cohort: pd.DataFrame = pd.DataFrame(),
        chunksize: int = 10000000,
        thresh=1,
        left_thresh=0,
        impute_outlier=False,
    ):
        self.cohort = cohort
        self.chunksize = chunksize
        self.thresh = thresh
        self.left_thresh = left_thresh
        self.impute_outlier = impute_outlier
        self.df = df
        self.final_df = pd.DataFrame()
        self.feature_path = EXTRACT_CHART_ICU_PATH

    def df(self):
        return self.df

    def extract_from(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Function for processing hospital observations from a pickled cohort, optimized for memory efficiency."""
        logger.info("[EXTRACTING CHART EVENTS DATA]")

        processed_chunks = [
            self.process_chunk_chart_events(chunk, cohort)
            for chunk in tqdm(load_icu_chart_events(self.chunksize))
        ]

        out = pd.concat(processed_chunks, ignore_index=True)
        out = drop_wrong_uom(out, 0.95)
        """Log statistics about the chart events."""
        logger.info(f"# Unique Events: {out[ChartEventsHeader.ITEM_ID].nunique()}")
        logger.info(f"# Admissions: {out[ChartEventsHeader.STAY_ID].nunique()}")
        logger.info(f"Total rows: {out.shape[0]}")
        out = out[[h.value for h in ChartEventsHeader]]
        self.df = out
        return out

    def process_chunk_chart_events(
        self, chunk: pd.DataFrame, cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """Process a single chunk of chart events."""
        chunk = chunk.dropna(subset=[ChartEvents.VALUENUM]).merge(
            cohort, on=ChartEvents.STAY_ID
        )
        chunk[ChartEventsHeader.EVENT_TIME_FROM_ADMIT] = (
            chunk[ChartEvents.CHARTTIME] - chunk[IcuCohortHeader.IN_TIME]
        )
        return chunk.drop(["charttime", "intime"], axis=1).dropna().drop_duplicates()

    def summary(self):
        chart: pd.DataFrame = self.df
        freq = (
            chart.groupby([ChartEventsHeader.STAY_ID, ChartEventsHeader.ITEM_ID])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = (
            freq.groupby([ChartEventsHeader.ITEM_ID])["mean_frequency"]
            .mean()
            .reset_index()
        )

        missing = (
            chart[chart[ChartEventsHeader.VALUE_NUM] == 0]
            .groupby(ChartEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="missing_count")
        )
        total = (
            chart.groupby(ChartEventsHeader.ITEM_ID)
            .size()
            .reset_index(name="total_count")
        )
        summary = pd.merge(missing, total, on=ChartEventsHeader.ITEM_ID, how="right")
        summary = pd.merge(freq, summary, on=ChartEventsHeader.ITEM_ID, how="right")
        summary = summary.fillna(0)
        return summary

    def preproc(self):
        logger.info("[PROCESSING CHART EVENTS DATA]")
        chart = pd.read_csv(EXTRACT_CHART_ICU_PATH, compression="gzip")
        chart = outlier_imputation(
            chart,
            "itemid",
            "valuenum",
            self.thresh,
            self.left_thresh,
            self.impute_outlier,
        )

        logger.info("Total number of rows", chart.shape[0])
        chart.to_csv(
            EXTRACT_CHART_ICU_PATH,
            compression="gzip",
            index=False,
        )
        logger.info("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    def generate_fun(self):
        chunksize = 5000000
        final = pd.DataFrame()
        for chart in tqdm(
            pd.read_csv(
                EXTRACT_CHART_ICU_PATH,
                compression="gzip",
                chunksize=chunksize,
            )
        ):
            chart = chart[chart["stay_id"].isin(self.data["stay_id"])]
            chart[["start_days", "dummy", "start_hours"]] = chart[
                "event_time_from_admit"
            ].str.split(" ", -1, expand=True)
            chart[["start_hours", "min", "sec"]] = chart["start_hours"].str.split(
                ":", -1, expand=True
            )
            chart["start_time"] = pd.to_numeric(
                chart["start_days"]
            ) * 24 + pd.to_numeric(chart["start_hours"])
            chart = chart.drop(
                columns=[
                    "start_days",
                    "dummy",
                    "start_hours",
                    "min",
                    "sec",
                    "event_time_from_admit",
                ]
            )
            chart = chart[chart["start_time"] >= 0]

            ###Remove where event time is after discharge time
            chart = pd.merge(
                chart, self.data[["stay_id", "los"]], on="stay_id", how="left"
            )
            chart["sanity"] = chart["los"] - chart["start_time"]
            chart = chart[chart["sanity"] > 0]
            del chart["sanity"]
            del chart["los"]

            if final.empty:
                final = chart
            else:
                final = final.append(chart, ignore_index=True)
        self.df = final
        return final

    def mortality_length(self, include_time):
        self.df = self.df[self.df["stay_id"].isin(self.df["stay_id"])]
        self.df = self.df[self.df["start_time"] <= include_time]

    def los_length(self, include_time):
        self.df = self.df[self.df["stay_id"].isin(self.df["stay_id"])]
        self.df = self.df[self.df["start_time"] <= include_time]

    def read_length(self):
        self.df = self.df[self.df["stay_id"].isin(self.cohort["stay_id"])]
        self.df = pd.merge(
            self.df, self.cohort[["stay_id", "select_time"]], on="stay_id", how="left"
        )
        self.df["start_time"] = self.df["start_time"] - self.df["select_time"]
        self.df = self.df[self.df["start_time"] >= 0]

    def smooth_meds_step(self, bucket, i, t):
        sub_chart = (
            self.df[
                (self.chart["start_time"] >= i) & (self.df["start_time"] < i + bucket)
            ]
            .groupby(["stay_id", "itemid"])
            .agg({"valuenum": np.nanmean})
        )
        sub_chart = sub_chart.reset_index()
        sub_chart["start_time"] = t
        if self.final_df.empty:
            self.final_df = sub_chart
        else:
            self.final_df = self.final_df.append(sub_chart)

    def smooth_meds(self):
        f2_df = self.final_df.groupby(["stay_id", "itemid"]).size()
        df_per_adm = f2_df.groupby("stay_id").sum().reset_index()[0].max()
        dflength_per_adm = self.final_df.groupby("stay_id").size().max()
        return f2_df, df_per_adm, dflength_per_adm

    def dict_step(self, hid, los, dataDic):
        feat = self.final_df["itemid"].unique()
        df2 = self.final_df[self.final_df["stay_id"] == hid]
        if df2.shape[0] == 0:
            val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
            val = val.fillna(0)
            val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])
        else:
            val = df2.pivot_table(
                index="start_time", columns="itemid", values="valuenum"
            )
            df2["val"] = 1
            df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
            add_indices = pd.Index(range(los)).difference(df2.index)
            add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
            df2 = pd.concat([df2, add_df])
            df2 = df2.sort_index()
            df2 = df2.fillna(0)

            val = pd.concat([val, add_df])
            val = val.sort_index()
            if self.impute == "Mean":
                val = val.ffill()
                val = val.bfill()
                val = val.fillna(val.mean())
            elif self.impute == "Median":
                val = val.ffill()
                val = val.bfill()
                val = val.fillna(val.median())
            val = val.fillna(0)

            df2[df2 > 0] = 1
            df2[df2 < 0] = 0
            dataDic[hid]["Chart"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
            dataDic[hid]["Chart"]["val"] = val.iloc[:, 0:].to_dict(orient="list")

            feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))
            val = pd.concat([val, feat_df], axis=1)

            val = val[feat]
            val = val.fillna(0)
            val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])
        return val
