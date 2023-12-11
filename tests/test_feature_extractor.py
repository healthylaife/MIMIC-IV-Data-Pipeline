from pipeline.features_extractor import (
    FeatureExtractor,
)
from pipeline.feature.feature_abc import Name


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
    assert len(result) == 5
    assert len(result[Name.DIAGNOSES]) == 2647
    assert result[Name.DIAGNOSES].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "icd_code",
        "root_icd10_convert",
        "root",
        "stay_id",
    ]
    assert len(result[Name.PROCEDURES]) == 1435
    assert result[Name.PROCEDURES].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "intime",
        "event_time_from_admit",
    ]
    assert len(result[Name.MEDICATIONS]) == 11038
    assert result[Name.MEDICATIONS].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "starttime",
        "start_hours_from_admit",
        "stop_hours_from_admit",
        "stay_id",
        "itemid",
        "endtime",
        "rate",
        "amount",
        "orderid",
    ]
    assert len(result[Name.OUTPUT]) == 9362
    assert result[Name.OUTPUT].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "intime",
        "event_time_from_admit",
    ]
    assert len(result[Name.CHART]) == 162571
    assert result[Name.CHART].columns.tolist() == [
        "stay_id",
        "itemid",
        "valuenum",
        "event_time_from_admit",
    ]


def test_feature_non_icu_all_true():
    feature_extractor = FeatureExtractor(
        cohort_output="cohort_Non-ICU_readmission_30_I50",
        use_icu=False,
        for_diagnoses=True,
        for_output_events=True,
        for_chart_events=True,
        for_procedures=True,
        for_medications=True,
        for_labs=True,
    )
    result = feature_extractor.save_features()
    assert len(result) == 4
    assert len(result[Name.DIAGNOSES]) == 1273
    assert result[Name.DIAGNOSES].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "icd_code",
        "root_icd10_convert",
        "root",
    ]
    assert len(result[Name.PROCEDURES]) == 136
    assert result[Name.PROCEDURES].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "icd_code",
        "icd_version",
        "chartdate",
        "admittime",
        "proc_time_from_admit",
    ]
    assert len(result[Name.MEDICATIONS]) == 4803
    assert result[Name.MEDICATIONS].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "starttime",
        "start_hours_from_admit",
        "stop_hours_from_admit",
        "stoptime",
        "drug",
        "nonproprietaryname",
        "dose_val_rx",
        "EPC",
    ]
    assert len(result[Name.LAB]) == 22029
    assert result[Name.LAB].columns.tolist() == [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "admittime",
        "lab_time_from_admit",
        "valuenum",
    ]
