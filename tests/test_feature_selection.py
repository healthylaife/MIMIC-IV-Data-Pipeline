from my_preprocessing.feature_selection_icu import feature_icu

# rename tests...

# add test for icd conversion


def test_feature_icu_all_true():
    result = feature_icu("cohort_icu_mortality_0_", True, True, True, True, True)
    assert result.shape[0] == 11038
    assert result["endtime"].max().year == 2201
    assert result["starttime"].min().year == 2110
    assert int(result["rate"].mean()) == 72  # to improve
    assert int(result["amount"].mean()) == 244  # to improve
    assert result["stop_hours_from_admit"].mean().days == 3
    assert result["stop_hours_from_admit"].mean().seconds == 65190


def test_feature_icu_all_false():
    result = feature_icu("cohort_icu_mortality_0_", False, False, False, False, False)
    assert result.shape[0] == 0
    assert result.shape[1] == 0
