from preprocessing.day_intervals_preproc.day_intervals_cohort_v2 import extract_data
from preprocessing.hosp_module_preproc.feature_selection_icu import (
    feature_icu,
    preprocess_features_icu,
    generate_summary_icu,
    features_selection_icu,
)
from model.data_generation_icu import Generator

cohort_output = extract_data(
    "ICU",
    "Mortality",
    0,
    "No Disease Filter",
    "d:\\Work\\Repos\\MIMIC-IV-Data-Pipeline",
    "",
)

feature_icu("cohort_icu_mortality_0_", "mimiciv", True, True, True, True, True)

preprocess_features_icu(
    "cohort_icu_mortality_0_",
    True,
    "Convert ICD-9 to ICD-10 and group ICD-10 codes",
    False,
    False,
    False,
    0,
    0,
)

generate_summary_icu(True, True, True, True, True)
features_selection_icu(
    "cohort_icu_mortality_0_",
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
)

preprocess_features_icu(
    "cohort_icu_mortality_0_", False, False, True, True, True, 98, 0
)

gen = Generator(
    "cohort_icu_mortality_0_",
    True,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    False,
    72,
    1,
    2,
)
