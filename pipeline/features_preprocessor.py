import pandas as pd
import logging
from pipeline.feature.diagnoses import Diagnoses, IcdGroupOption
from pipeline.feature.lab_events import Lab
from pipeline.feature.medications import Medications
from pipeline.feature.output_events import OutputEvents
from pipeline.feature.procedures import Procedures
from pipeline.feature_selector import FeatureSelector
from pipeline.features_extractor import FeatureExtractor
from typing import List
from pathlib import Path

from pipeline.feature.chart_events import Chart

logger = logging.getLogger()


class FeaturePreprocessor:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        group_diag_icd: IcdGroupOption,
        group_med_code: bool,
        keep_proc_icd9: bool,
        clean_chart: bool = False,
        impute_outlier_chart: bool = False,
        clean_labs: bool = False,
        impute_labs: bool = False,
        thresh: int = 100,
        left_thresh: int = 0,
    ):
        self.feature_extractor = feature_extractor
        self.group_diag_icd = group_diag_icd
        self.group_med_code = group_med_code
        self.keep_proc_icd9 = keep_proc_icd9
        self.clean_chart = clean_chart
        self.impute_outlier_chart = impute_outlier_chart
        self.clean_labs = clean_labs
        self.impute_labs = impute_labs
        self.thresh = thresh
        self.left_thresh = left_thresh

    def feature_selection(self) -> List[pd.DataFrame]:
        feature_selector = FeatureSelector(
            use_icu=self.feature_extractor.use_icu,
            select_dia=self.feature_extractor.for_diagnoses,
            select_med=self.feature_extractor.for_medications,
            select_proc=self.feature_extractor.for_procedures,
            select_chart=self.feature_extractor.for_chart_events,
            select_labs=self.feature_extractor.for_labs,
            select_out=self.feature_extractor.for_output_events,
        )
        return feature_selector.feature_selection()

    def preproc_events_features(self):
        event_preproc_features: List[pd.DataFrame] = []
        if self.clean_chart and self.feature_extractor.use_icu:
            chart = Chart(
                cohort=pd.DataFrame(),
                thresh=self.thresh,
                left_thresh=self.left_thresh,
                impute_outlier=self.impute_outlier_chart,
            )
            event_preproc_features.append(chart.preproc())
        if self.clean_labs and not self.feature_extractor.use_icu:
            lab = Lab(
                cohort=pd.DataFrame(),
                thresh=self.thresh,
                left_thresh=self.left_thresh,
                impute_outlier=self.impute_labs,
            )
            event_preproc_features.append(lab.preproc())
        return event_preproc_features

    def preprocess(self):
        self.preprocess_no_event_features()
        self.save_summaries()
        self.feature_selection()
        self.preproc_events_features()

    def preprocess_no_event_features(self):
        no_event_preproc_features = []
        empty_cohort = pd.DataFrame()
        if self.feature_extractor.for_diagnoses:
            dia = Diagnoses(
                cohort=empty_cohort,
                use_icu=self.feature_extractor.use_icu,
                icd_group_option=self.group_diag_icd,
            )
            no_event_preproc_features.append(dia.preproc())
        if not self.feature_extractor.use_icu:
            if self.feature_extractor.for_medications:
                med = Medications(
                    cohort=empty_cohort,
                    use_icu=self.feature_extractor.use_icu,
                    group_code=self.group_med_code,
                )
                no_event_preproc_features.append(med.preproc())
            if self.feature_extractor.for_procedures:
                proc = Procedures(
                    cohort=empty_cohort,
                    use_icu=self.feature_extractor.use_icu,
                    keep_icd9=self.keep_proc_icd9,
                )
                no_event_preproc_features.append(proc.preproc())
        return no_event_preproc_features

    def save_summaries(self):
        summaries = []
        if self.feature_extractor.for_diagnoses:
            dia = Diagnoses(
                cohort=pd.DataFrame(),
                use_icu=self.feature_extractor.use_icu,
                icd_group_option=self.group_diag_icd,
            )
            summaries.append(dia.summary())
        if self.feature_extractor.for_medications:
            med = Medications(
                cohort=pd.DataFrame(),
                use_icu=self.feature_extractor.use_icu,
                group_code=self.group_med_code,
            )
            summaries.append(med.summary())
        if self.feature_extractor.for_procedures:
            proc = Procedures(
                cohort=pd.DataFrame(),
                use_icu=self.feature_extractor.use_icu,
                keep_icd9=self.keep_proc_icd9,
            )
            summaries.append(proc.summary())
        if self.feature_extractor.for_output_events:
            out = OutputEvents(cohort=pd.DataFrame())
            summaries.append(out.summary())
        if self.feature_extractor.for_chart_events:
            chart = Chart(cohort=pd.DataFrame())
            summaries.append(chart.summary())
        if self.feature_extractor.for_labs:
            lab = Lab(cohort=pd.DataFrame())
            summaries.append(lab.summary)
        return summaries
