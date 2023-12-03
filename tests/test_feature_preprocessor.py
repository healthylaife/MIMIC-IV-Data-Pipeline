from my_preprocessing.features_extractor import FeatureExtractor
from my_preprocessing.features_preprocessor import FeaturePreprocessor, IcdGroupOption


def test_feature_icu_all_true():
    extractor = FeatureExtractor(
        cohort_output="cohort_icu_mortality_0_",
        use_icu=True,
        for_diagnoses=True,
        for_output_events=True,
        for_chart_events=True,
        for_procedures=True,
        for_medications=True,
        for_labs=True,
    )
    preprocessor = FeaturePreprocessor(
        feature_extractor=extractor,
        group_diag_icd=IcdGroupOption.GROUP,
        group_med_code=True,
        keep_proc_icd9=False,
        clean_chart=True,
        impute_outlier_chart=True,
        impute_labs=True,
        thresh=98,
        left_thresh=2,
        clean_labs=True,
    )

    preprocessor.preprocess()
    assert 0 == 0


def test_feature_non_icu_all_true():
    extractor = FeatureExtractor(
        cohort_output="cohort_Non-ICU_readmission_30_I50",
        use_icu=False,
        for_diagnoses=True,
        for_output_events=True,
        for_chart_events=True,
        for_procedures=True,
        for_medications=True,
        for_labs=True,
    )
    preprocessor = FeaturePreprocessor(
        feature_extractor=extractor,
        group_diag_icd=IcdGroupOption.GROUP,
        group_med_code=True,
        keep_proc_icd9=False,
        clean_chart=True,
        impute_outlier_chart=True,
        impute_labs=True,
        thresh=95,
        left_thresh=5,
        clean_labs=True,
    )
    preprocessor.preprocess()
    assert 0 == 0
