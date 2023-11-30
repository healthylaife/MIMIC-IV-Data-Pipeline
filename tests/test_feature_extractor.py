from my_preprocessing.feature_extractor import feature_icu, feature_non_icu

# rename tests...

# add test for icd conversion


def test_feature_icu_all_true():
    result = feature_icu("cohort_icu_mortality_0_", True, True, True, True, True)
    assert 0 == 0


def test_feature_non_icu_all_true():
    result = feature_non_icu(
        "cohort_NON-ICU_readmission_30_I25", True, True, True, True
    )
    assert 0 == 0
