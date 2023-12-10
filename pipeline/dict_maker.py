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
from pipeline.file_info.common import PREPROC_PATH
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


class DictMaker:
    def __init__(self, feature_extractor: FeatureExtractor, hids):
        self.feature_extractor = feature_extractor
        self.hids = hids

    def create_dict(self, meds, proc, out, labs, chart, los):
        self.dataDic = {}
        self.labels_csv = pd.DataFrame(
            columns=[
                "stay_id" if self.feature_extractor.use_icu else "hadm_id",
                "label",
            ]
        )
        self.labels_csv[
            "stay_id" if self.feature_extractor.use_icu else "hadm_id"
        ] = pd.Series(self.hids)
        self.labels_csv["label"] = 0
        self.process_feature_data(meds, "Med", los)
        self.process_feature_data(proc, "Proc", los)
        self.process_feature_data(out, "Out", los)
        self.process_feature_data(labs, "Lab", los)
        self.process_feature_data(chart, "Chart", los)
        self.save_csv_files()

        return

    def process_feature_data(self, feature_df, feature_name, los):
        for hid in tqdm(self.hids):
            # Process specific feature data for each 'hid'
            # Update self.dataDic[hid][feature_name] with processed data
            return

    def save_csv_files(self):
        for hid in self.hids:
            self.save_individual_csv(hid)
        self.labels_csv.to_csv(PREPROC_PATH / "csv/labels.csv", index=False)

    def save_individual_csv(self, hid):
        # Save demographic and dynamic data to CSV files
        demo_csv_path = os.path.join(PREPROC_PATH / f"/csv/{str(hid)}demo.csv")
        dynamic_csv_path = os.path.join(PREPROC_PATH / f"/csv/{str(hid)}dynamic.csv")
        static_csv_path = os.path.join(PREPROC_PATH / f"/csv/{str(hid)}static.csv")
        return
        # Save corresponding DataFrames to these paths
