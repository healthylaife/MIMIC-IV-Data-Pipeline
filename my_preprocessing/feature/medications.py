from my_preprocessing.feature.feature import Feature
import logging
import pandas as pd
from my_preprocessing.ndc_conversion import (
    NdcMappingHeader,
    get_EPC,
    ndc_to_str,
    prepare_ndc_mapping,
)
from my_preprocessing.preproc.feature import (
    MedicationsHeader,
    IcuMedicationHeader,
    NonIcuMedicationHeader,
    PREPROC_MED_ICU_PATH,
    PREPROC_MED_PATH,
)
from my_preprocessing.preproc.cohort import CohortHeader
from my_preprocessing.raw.hosp import (
    HospPrescriptions,
    load_hosp_prescriptions,
)
from my_preprocessing.raw.icu import (
    InputEvents,
    load_input_events,
)
from my_preprocessing.file_info import save_data
from pathlib import Path

logger = logging.getLogger()


class Medications(Feature):
    def __init__(self, cohort: pd.DataFrame, use_icu: bool):
        self.cohort = cohort
        self.use_icu = use_icu

    def summary_path(self) -> Path:
        return  # PREPROC_MED_ICU_PATH if self.use_icu else PREPROC_MED_PATH

    def feature_path(self) -> Path:
        return PREPROC_MED_ICU_PATH if self.use_icu else PREPROC_MED_PATH

    def make(self) -> pd.DataFrame:
        logger.info("[EXTRACTING MEDICATIONS DATA]")
        cohort_headers = (
            [
                CohortHeader.HOSPITAL_ADMISSION_ID,
                CohortHeader.STAY_ID,
                CohortHeader.IN_TIME,
            ]
            if self.use_icu
            else [CohortHeader.HOSPITAL_ADMISSION_ID, CohortHeader.ADMIT_TIME]
        )
        admissions = self.cohort[cohort_headers]
        raw_med = load_input_events() if self.use_icu else load_hosp_prescriptions()
        medications = raw_med.merge(
            admissions,
            on=CohortHeader.STAY_ID
            if self.use_icu
            else CohortHeader.HOSPITAL_ADMISSION_ID,
        )
        admit_header = CohortHeader.IN_TIME if self.use_icu else CohortHeader.ADMIT_TIME

        medications[MedicationsHeader.START_HOURS_FROM_ADMIT] = (
            medications[InputEvents.STARTTIME] - medications[admit_header]
        )
        medications[MedicationsHeader.STOP_HOURS_FROM_ADMIT] = (
            medications[
                InputEvents.ENDTIME if self.use_icu else HospPrescriptions.STOP_TIME
            ]
            - medications[admit_header]
        )
        medications = (
            medications.dropna()
            if self.use_icu
            else self.normalize_non_icu(medications)
        )
        self.log_icu(medications) if self.use_icu else self.log_non_icu(medications)
        return medications

    def normalize_non_icu(self, med: pd.DataFrame):
        med[NonIcuMedicationHeader.DRUG] = (
            med[NonIcuMedicationHeader.DRUG].fillna("").astype(str)
        )
        med[NonIcuMedicationHeader.DRUG] = med[NonIcuMedicationHeader.DRUG].apply(
            lambda x: str(x).lower().strip().replace(" ", "_") if not "" else ""
        )
        med[NonIcuMedicationHeader.DRUG] = (
            med[NonIcuMedicationHeader.DRUG]
            .dropna()
            .apply(lambda x: str(x).lower().strip())
        )
        med[HospPrescriptions.NDC] = med[HospPrescriptions.NDC].fillna(-1)

        # Ensures the decimal is removed from the ndc col
        med[HospPrescriptions.NDC] = med[HospPrescriptions.NDC].astype("Int64")
        med[NdcMappingHeader.NEW_NDC] = med[HospPrescriptions.NDC].apply(ndc_to_str)
        ndc_map = prepare_ndc_mapping()
        med = med.merge(ndc_map, on=NdcMappingHeader.NEW_NDC)

        # Function generates a list of EPCs, as a drug can have multiple EPCs
        med[NonIcuMedicationHeader.EPC] = med.pharm_classes.apply(get_EPC)
        return med

    def log_icu(self, med: pd.DataFrame) -> None:
        logger.info(f"# of unique type of drug: {med[InputEvents.ITEMID].nunique()}")
        logger.info(f"# Admissions:  {med[InputEvents.STAY_ID].nunique()}")

        logger.info(f"# Total rows: {med.shape[0]}")
        return med

    def log_non_icu(self, med: pd.DataFrame) -> None:
        logger.info(
            f"Number of unique type of drug: {med[NonIcuMedicationHeader.DRUG].nunique()}"
        )
        logger.info(
            f"Number of unique type of drug (after grouping to use Non propietary names): {med[NonIcuMedicationHeader.NON_PROPRIEATARY_NAME].nunique()}"
        )
        print("# Admissions:  ", med[CohortHeader.HOSPITAL_ADMISSION_ID].nunique())
        print("Total number of rows: ", med.shape[0])

    def save(self) -> pd.DataFrame:
        cols = [h.value for h in MedicationsHeader] + [
            h.value
            for h in (IcuMedicationHeader if self.use_icu else NonIcuMedicationHeader)
        ]
        med = self.make()
        med = med[cols]
        return save_data(med, self.feature_path(), "MEDICATIONS")

    def preproc(self):
        logger.info("[PROCESSING MEDICATIONS DATA]")
        path = self.feature_path()
        med = pd.read_csv(path, compression="gzip")
        med[PreprocMedicationHeader.DRUG_NAME] = (
            med[NonIcuMedicationHeader.NON_PROPRIEATARY_NAME]
            if self.group_med_code
            else med[NonIcuMedicationHeader.DRUG]
        )
        med = med.drop(
            columns=[
                NonIcuMedicationHeader.NON_PROPRIEATARY_NAME,
                NonIcuMedicationHeader.DRUG,
            ]
        )
        med.dropna()
        print("Total number of rows", med.shape[0])
        med.to_csv(path, compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
        return med``
