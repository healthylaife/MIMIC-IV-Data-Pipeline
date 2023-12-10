from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from pipeline.feature.lab_events import Lab
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.file_info.preproc.cohort import COHORT_PATH
from pipeline.file_info.preproc.feature import (
    EXTRACT_CHART_ICU_PATH,
    EXTRACT_DIAG_ICU_PATH,
    EXTRACT_DIAG_PATH,
    EXTRACT_LABS_PATH,
    EXTRACT_MED_ICU_PATH,
    EXTRACT_MED_PATH,
    EXTRACT_OUT_ICU_PATH,
    EXTRACT_PROC_ICU_PATH,
    EXTRACT_PROC_PATH,
)
from pipeline.prediction_task import PredictionTask, TargetType
import logging

from pipeline.features_extractor import FeatureExtractor
from pipeline.feature.chart_events import Chart, ChartEvents
from pipeline.feature.diagnoses import Diagnoses
from pipeline.preprocessing.cohort import read_cohort
from pipeline.feature.feature_abc import Feature

logger = logging.getLogger()


class DataGenerator:
    def __init__(
        self,
        cohort_output: pd.DataFrame,
        feature_extractor: FeatureExtractor,
        # impute: str,
        include_time: int = 24,
        bucket: int = 1,
        predW: int = 0,
        target_type: TargetType = TargetType.LOS,
    ):
        self.cohort_output = cohort_output
        self.feature_extractor = feature_extractor
        # self.impute = impute
        self.include_time = include_time
        self.bucket = bucket
        self.predW = predW
        self.target_type = target_type
        self.dia = pd.DataFrame()
        self.proc = pd.DataFrame()
        self.out = pd.DataFrame()
        self.chart = pd.DataFrame()
        self.med = pd.DataFrame()
        self.lab = pd.DataFrame()

    def generate_features(self):
        print("[ ======READING DIAGNOSIS ]")
        self.cohort = read_cohort(self.cohort_output, self.feature_extractor.use_icu)
        if self.feature_extractor.for_diagnoses:
            preproc_dia = pd.read_csv(
                EXTRACT_DIAG_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_DIAG_PATH,
                compression="gzip",
            )
            dia = Diagnoses(use_icu=self.feature_extractor.use_icu, df=preproc_dia)
            self.dia, self.dia_per_adm = dia.generate_fun(self.cohort)
        if self.feature_extractor.for_procedures:
            print("[ ======READING PROCEDURES ]")
            preproc_proc = pd.read_csv(
                EXTRACT_PROC_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_PROC_PATH,
                compression="gzip",
            )
            proc = Procedures(use_icu=self.feature_extractor.use_icu, df=preproc_proc)
            self.proc = proc.generate_fun(self.cohort)

        if self.feature_extractor.use_icu and self.feature_extractor.for_output_events:
            print("[ ======READING OUTPUT ]")
            preproc_out = pd.read_csv(EXTRACT_OUT_ICU_PATH, compression="gzip")
            out = OutputEvents(df=preproc_out)
            self.out = out.generate_fun(self.cohort)

        if self.feature_extractor.use_icu and self.feature_extractor.for_chart_events:
            print("[ ======READING CHART ]")
            preproc_chart = pd.read_csv(
                EXTRACT_CHART_ICU_PATH, compression="gzip", chunksize=5000000
            )
            chart = Chart(df=preproc_chart)
            self.chart = chart.generate_fun(self.cohort)

        if self.feature_extractor.for_medications:
            print("[ ======READING MEDICATIONS ]")
            preproc_med = pd.read_csv(
                EXTRACT_MED_ICU_PATH
                if self.feature_extractor.use_icu
                else EXTRACT_MED_PATH,
                compression="gzip",
            )
            med = Medications(use_icu=self.feature_extractor.use_icu, df=preproc_med)
            self.med = med.generate_fun(self.cohort)
        if not (self.feature_extractor.use_icu) and self.feature_extractor.for_labs:
            print("[ ======READING LABS ]")
            preproc_labs = pd.read_csv(
                EXTRACT_LABS_PATH, compression="gzip", chunksize=5000000
            )
            lab = Lab(df=preproc_labs)
            self.lab = lab.generate_fun(self.cohort)

    def length_by_target(self):
        if self.target_type == TargetType.MORTALITY:
            if self.feature_extractor.for_diagnoses:
                dia = Diagnoses(use_icu=self.feature_extractor.use_icu, df=self.dia)
                dia.mortality_length(self.cohort)
                self.dia = dia.df
            if self.feature_extractor.for_procedures:
                proc = Procedures(use_icu=self.feature_extractor.use_icu, df=self.proc)
                proc.mortality_length(self.cohort, self.include_time)
                self.proc = proc.df
            if (
                self.feature_extractor.use_icu
                and self.feature_extractor.for_output_events
            ):
                out = OutputEvents(df=self.out)
                out.mortality_length(self.cohort, self.include_time)
                self.out = out.df
            if (
                self.feature_extractor.use_icu
                and self.feature_extractor.for_chart_events
            ):
                chart = Chart(df=self.chart)
                chart.mortality_length(self.cohort, self.include_time)
                self.chart = chart.df
            if self.feature_extractor.for_medications:
                med = Medications(use_icu=self.feature_extractor.use_icu, df=self.chart)
                med.mortality_length(self.cohort, self.include_time)
                self.med = med.df
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif self.target_type == TargetType.READMISSION:
            if self.feature_extractor.for_diagnoses:
                dia = Diagnoses(use_icu=self.feature_extractor.use_icu, df=self.dia)
                dia.read_length()
                self.dia = dia.df
            if self.feature_extractor.for_procedures:
                proc = Procedures(use_icu=self.feature_extractor.use_icu, df=self.proc)
                proc.read_length(self.cohort)
                self.proc = proc.df
            if (
                self.feature_extractor.use_icu
                and self.feature_extractor.for_output_events
            ):
                out = OutputEvents(df=self.out)
                out.read_length(self.cohort)
                self.out = out.df
            if (
                self.feature_extractor.use_icu
                and self.feature_extractor.for_chart_events
            ):
                chart = Chart(df=self.chart)
                chart.read_length(self.cohort)
                self.chart = chart.df
            if self.feature_extractor.for_medications:
                med = Medications(use_icu=self.feature_extractor.use_icu, df=self.chart)
                med.read_length(self.cohort)
                self.med = med.df
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif self.target_type == TargetType.LOS:
            if self.feature_extractor.for_diagnoses:
                dia = Diagnoses(use_icu=self.feature_extractor.use_icu, df=self.dia)
                dia.los_length(self.cohort)
                self.dia = dia.df
            if self.feature_extractor.for_procedures:
                proc = Procedures(use_icu=self.feature_extractor.use_icu, df=self.proc)
                proc.los_length(self.cohort, self.include_time)
                self.proc = proc.df
            if (
                self.feature_extractor.use_icu
                and self.feature_extractor.for_output_events
            ):
                out = OutputEvents(df=self.out)
                out.los_length(self.cohort, self.include_time)
                self.out = out.df
            if (
                self.feature_extractor.use_icu
                and self.feature_extractor.for_chart_events
            ):
                chart = Chart(df=self.chart)
                chart.los_length(self.cohort, self.include_time)
                self.chart = chart.df
            if self.feature_extractor.for_medications:
                med = Medications(use_icu=self.feature_extractor.use_icu, df=self.med)
                med.los_length(self.cohort, self.include_time)
                self.med = med.df
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")

    def smooth_ini(self):
        if self.feature_extractor.for_medications:
            self.med = self.med.sort_values(by=["start_time"])
        if self.feature_extractor.for_procedures:
            self.proc = self.proc.sort_values(by=["start_time"])
        if self.feature_extractor.for_output_events and self.feature_extractor.use_icu:
            self.out = self.out.sort_values(by=["start_time"])
        if self.feature_extractor.for_chart_events and self.feature_extractor.use_icu:
            self.chart = self.chart.sort_values(by=["start_time"])

        return

    def smooth_tqdm(self):
        final_proc = pd.DataFrame()
        final_out = pd.DataFrame()
        final_chart = pd.DataFrame()
        final_meds = pd.DataFrame()
        final_lab = pd.DataFrame()
        t = 0
        for i in tqdm(range(0, self.include_time, self.bucket)):
            if self.feature_extractor.for_medications:
                med = Medications(use_icu=self.feature_extractor.use_icu, df=self.med)
                sub_meds = med.smooth_meds_step(self.bucket, i, t)
                if final_meds.empty:
                    final_meds = sub_meds
                else:
                    final_meds = pd.concat([final_meds, sub_meds], ignore_index=True)

            if self.feature_extractor.for_procedures:
                proc = Procedures(use_icu=self.feature_extractor.use_icu, df=self.proc)
                sub_proc = proc.smooth_meds_step(self.bucket, i, t)
                if final_proc.empty:
                    final_proc = sub_proc
                else:
                    final_proc = pd.concat([final_proc, sub_proc], ignore_index=True)

            if (
                self.feature_extractor.for_output_events
                and self.feature_extractor.use_icu
            ):
                out = OutputEvents(df=self.out)
                sub_out = out.smooth_meds_step(self.bucket, i, t)
                if final_out.empty:
                    final_out = sub_out
                else:
                    final_out = pd.concat([final_out, sub_out], ignore_index=True)

            if (
                self.feature_extractor.for_chart_events
                and self.feature_extractor.use_icu
            ):
                chart = Chart(df=self.chart)
                sub_chart = chart.smooth_meds_step(self.bucket, i, t)
                if final_chart.empty:
                    final_chart = sub_chart
                else:
                    final_chart = pd.concat([final_chart, sub_chart], ignore_index=True)

            if self.feature_extractor.for_labs and not self.feature_extractor.use_icu:
                lab = Lab(df=self.lab)
                sub_lab = lab.smooth_meds_step(self.bucket, i, t)
                if final_lab.empty:
                    final_lab = sub_lab
                else:
                    final_lab = pd.concat([final_lab, sub_lab], ignore_index=True)
