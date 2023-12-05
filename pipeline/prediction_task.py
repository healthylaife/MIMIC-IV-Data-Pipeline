from enum import StrEnum


class TargetType(StrEnum):
    MORTALITY = "Mortality"
    LOS = "Lenghth of stay"
    READMISSION = "Readmission"


class DiseaseCode(StrEnum):
    HEARTH_FAILURE = "I50"
    CAD = "I25"  # Coronary Artery Disease
    CKD = "N18"  # Chronic Kidney Disease
    COPD = "J44"  # Chronic obstructive pulmonary disease


class PredictionTask:
    def __init__(
        self,
        target_type: TargetType,
        disease_readmission: DiseaseCode | None,
        disease_selection: DiseaseCode | None,
        nb_days: int | None,
        use_icu: bool,
    ):
        if nb_days is not None and nb_days < 0:
            raise ValueError(
                "the number of days after a readmission should be positive."
            )
        self.target_type = target_type
        self.disease_readmission = disease_readmission
        self.disease_selection = disease_selection
        self.nb_days = nb_days
        self.use_icu = use_icu
