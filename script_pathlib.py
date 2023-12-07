from pipeline.cohort_extractor import CohortExtractor
from pipeline.feature.diagnoses import IcdGroupOption
from pipeline.feature_selector import FeatureSelector
from pipeline.features_preprocessor import FeaturePreprocessor
from pipeline.prediction_task import TargetType, PredictionTask, DiseaseCode
from pipeline.features_extractor import FeatureExtractor

if __name__ == "__main__":
    prediction_task = PredictionTask(
        target_type=TargetType.READMISSION,
        disease_readmission=DiseaseCode.CAD,
        disease_selection=None,
        nb_days=30,
        use_icu=False,
    )

    cohort_extractor = CohortExtractor(prediction_task=prediction_task)
    cohort = cohort_extractor.extract()
    feature_extractor = FeatureExtractor(
        cohort_output=cohort_extractor.cohort_output,
        use_icu=prediction_task.use_icu,
        for_diagnoses=True,
        for_labs=not prediction_task.use_icu,
        for_chart_events=prediction_task.use_icu,
        for_medications=True,
        for_output_events=prediction_task.use_icu,
        for_procedures=True,
    )
    features = feature_extractor.save_features()

    feat_preproc = FeaturePreprocessor(
        feature_extractor=feature_extractor,
        group_diag_icd=IcdGroupOption.KEEP,
        group_med_code=True,
        keep_proc_icd9=False,
        clean_chart=False,
        impute_outlier_chart=False,
        clean_labs=False,
        impute_labs=False,
    )
    preproc = feat_preproc.preprocess_no_event_features()
    summaries = feat_preproc.save_summaries()

    feat_select = FeatureSelector(
        prediction_task.use_icu,
        feature_extractor.for_diagnoses,
        feature_extractor.for_medications,
        feature_extractor.for_procedures,
        not (prediction_task.use_icu) and feature_extractor.for_labs,
        prediction_task.use_icu and feature_extractor.for_chart_events,
        prediction_task.use_icu and feature_extractor.for_output_events,
    )

    selection = feat_select.feature_selection()

    feat_preproc = FeaturePreprocessor(
        feature_extractor=feature_extractor,
        group_diag_icd=IcdGroupOption.KEEP,
        group_med_code=False,
        keep_proc_icd9=False,
        clean_chart=False,
        impute_outlier_chart=False,
        clean_labs=True,
        impute_labs=True,
        thresh=98,
        left_thresh=0,
    )
    feat_preproc.preproc_events_features()

    from pipeline.data_generator import DataGenerator
    from pipeline.preprocessing.data_gen import generate_admission_cohort

    cohort = generate_admission_cohort(feature_extractor.cohort_output)
    # data_generator = DataGenerator(
