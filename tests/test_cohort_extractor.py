import pytest
from my_preprocessing.cohort_extractor import CohortExtractor
from my_preprocessing.prediction_task import PredictionTask, TargetType


@pytest.mark.parametrize(
    "use_icu, target_type, nb_days, disease_readmission, disease_selection, expected_admission_records_count, expected_patients_count, expected_positive_cases_count",
    [
        (True, TargetType.MORTALITY, 0, None, None, 140, 100, 10),
        (True, TargetType.LOS, 3, None, None, 140, 100, 55),
        (True, TargetType.LOS, 7, None, None, 140, 100, 20),
        (True, TargetType.READMISSION, 30, None, None, 128, 93, 18),
        (True, TargetType.READMISSION, 90, None, None, 128, 93, 22),
        (True, TargetType.READMISSION, 30, "I50", None, 27, 20, 2),
        (True, TargetType.READMISSION, 30, "I25", None, 32, 29, 2),
        (True, TargetType.READMISSION, 30, "N18", None, 25, 18, 2),
        (True, TargetType.READMISSION, 30, "J44", None, 17, 12, 3),
        (False, TargetType.MORTALITY, 0, None, None, 275, 100, 15),
        (False, TargetType.LOS, 3, None, None, 275, 100, 163),
        (False, TargetType.LOS, 7, None, None, 275, 100, 76),
        (False, TargetType.READMISSION, 30, None, None, 260, 95, 52),
        (False, TargetType.READMISSION, 90, None, None, 260, 95, 86),
        (False, TargetType.READMISSION, 30, "I50", None, 55, 23, 13),
        # heart failure
        (False, TargetType.READMISSION, 30, "I25", None, 68, 32, 13),
        (False, TargetType.READMISSION, 30, "N18", None, 63, 22, 10),
        (False, TargetType.READMISSION, 30, "J44", None, 26, 12, 7),
        (True, TargetType.MORTALITY, 0, None, "I50", 32, 22, 5),
    ],
)
def test_cohort_extractor(
    use_icu,
    target_type,
    nb_days,
    disease_readmission,
    disease_selection,
    expected_admission_records_count,
    expected_patients_count,
    expected_positive_cases_count,
):
    prediction_task = PredictionTask(
        target_type, disease_readmission, disease_selection, nb_days, use_icu
    )
    cohort_extractor = CohortExtractor(
        prediction_task=prediction_task,
    )
    cohort = cohort_extractor.extract()
    assert len(cohort) == expected_admission_records_count
    assert cohort["subject_id"].nunique() == expected_patients_count
    assert cohort["label"].sum() == expected_positive_cases_count
