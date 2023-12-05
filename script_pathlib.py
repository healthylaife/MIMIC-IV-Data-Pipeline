from my_preprocessing.cohort_extractor import CohortExtractor
from my_preprocessing.features_extractor import FeatureExtractor
from my_preprocessing.prediction_task import DiseaseCode, PredictionTask, TargetType


prediction_task = PredictionTask(TargetType.LOS, None, DiseaseCode.CKD, 7, True)

cohort_extractor = CohortExtractor(
    prediction_task=prediction_task,
    cohort_output=None,
    summary_output=None,
)
cohort = cohort_extractor.extract()

feature_extractor = FeatureExtractor(
    "cohort_ICU_lenghth_of_stay_7__I25", True, True, True, True, True, True, False
)

feature_extractor.save_features()
