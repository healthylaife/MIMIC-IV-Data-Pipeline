import pytest
from my_preprocessing.feature_selection_icu import feature_icu


def test_feature_icu():
    toto = feature_icu("cohort_icu_mortality_0_", True, True, True, True, True)
    breakpoint()
    assert 0 == 0
