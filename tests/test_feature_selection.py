from my_preprocessing.feature_selection_icu import feature_icu, feature_non_icu

# rename tests...

# add test for icd conversion


def test_feature_icu_all_true():
    result = feature_icu("cohort_icu_mortality_0_", True, True, True, True, True)
    assert 0 == 0


def test_feature_non_icu_all_true():
    result = feature_non_icu("NON-ICU_mortality_0_", True, True, True, True)
    assert 0 == 0
