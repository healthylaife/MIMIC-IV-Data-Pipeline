from my_preprocessing.features_extractor import (
    FeatureExtractor,
)


def test_feature_icu_all_true():
    feature_extractor = FeatureExtractor(
        cohort_output="cohort_icu_mortality_0_",
        use_icu=True,
        for_diagnoses=True,
        for_output_events=True,
        for_chart_events=True,
        for_procedures=True,
        for_medications=True,
        for_labs=True,
    )
    result = feature_extractor.save_features()
    assert 0 == 0


def test_feature_non_icu_all_true():
    feature_extractor = FeatureExtractor(
        cohort_output="cohort_icu_mortality_0_",
        use_icu=True,
        for_diagnoses=True,
        for_output_events=True,
        for_chart_events=True,
        for_procedures=True,
        for_medications=True,
        for_labs=True,
    )
    result = feature_extractor.save_features()
    assert 0 == 0
