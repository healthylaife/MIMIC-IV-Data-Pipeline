import pandas as pd
from tqdm import tqdm

from my_preprocessing.preproc_file_info import (
    CohortHeader,
    NonIcuProceduresHeader,
    LabEventsHeader,
    MedicationsHeader,
    NonIcuMedicationHeader,
)
from my_preprocessing.raw_file_info import (
    MAP_NDC_PATH,
    load_hosp_procedures_icd,
    load_hosp_lab_events,
    load_hosp_admissions,
    load_hosp_predictions,
    HospAdmissions,
    HospProceduresIcd,
    HospLabEvents,
    HospPrescriptions,
)
from my_preprocessing.admission_imputer import (
    impute_hadm_ids,
    INPUTED_HOSPITAL_ADMISSION_ID_HEADER,
)
from my_preprocessing.ndc_conversion import (
    format_ndc_table,
    read_ndc_mapping,
    ndc_to_str,
    get_EPC,
)
from my_preprocessing.uom_conversion import drop_wrong_uom
from typing import Tuple

CHUNKSIZE = 10000000


def make_labs_events_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Process and transform lab events data."""
    admissions = load_hosp_admissions()[
        [
            HospAdmissions.PATIENT_ID,
            HospAdmissions.ID,
            HospAdmissions.ADMITTIME,
            HospAdmissions.DISCHTIME,
        ]
    ]
    usecols = [
        HospLabEvents.ITEM_ID,
        HospLabEvents.PATIENT_ID,
        HospLabEvents.HOSPITAL_ADMISSION_ID,
        HospLabEvents.CHART_TIME,
        HospLabEvents.VALUE_NUM,
        HospLabEvents.VALUE_UOM,
    ]

    processed_chunks = [
        process_lab_chunk(chunk, cohort, admissions)
        for chunk in tqdm(load_hosp_lab_events(chunksize=CHUNKSIZE, use_cols=usecols))
    ]

    return pd.concat(processed_chunks, ignore_index=True)


def process_lab_chunk(
    chunk: pd.DataFrame, cohort: pd.DataFrame, admissions: pd.DataFrame
) -> pd.DataFrame:
    """Process a single chunk of lab events."""
    chunk = chunk.dropna(subset=[HospLabEvents.VALUE_NUM]).fillna(
        {HospLabEvents.VALUE_UOM: 0}
    )
    chunk = chunk[
        chunk[LabEventsHeader.PATIENT_ID].isin(cohort[CohortHeader.PATIENT_ID])
    ]
    chunk_with_hadm, chunk_no_hadm = (
        chunk[chunk[HospLabEvents.HOSPITAL_ADMISSION_ID].notna()],
        chunk[chunk[HospLabEvents.HOSPITAL_ADMISSION_ID].isna()],
    )
    chunk_imputed = impute_hadm_ids(chunk_no_hadm.copy(), admissions)
    chunk_imputed[HospLabEvents.HOSPITAL_ADMISSION_ID] = chunk_imputed[
        INPUTED_HOSPITAL_ADMISSION_ID_HEADER
    ]
    chunk_imputed = chunk_imputed[
        [
            HospLabEvents.PATIENT_ID,
            HospLabEvents.HOSPITAL_ADMISSION_ID,
            HospLabEvents.ITEM_ID,
            HospLabEvents.CHART_TIME,
            HospLabEvents.VALUE_NUM,
            HospLabEvents.VALUE_UOM,
        ]
    ]
    merged_chunk = pd.concat([chunk_with_hadm, chunk_imputed], ignore_index=True)
    return merge_with_cohort_and_calculate_lab_time(merged_chunk, cohort)


def merge_with_cohort_and_calculate_lab_time(
    chunk: pd.DataFrame, cohort: pd.DataFrame
) -> pd.DataFrame:
    """Merge chunk with cohort data and calculate the lab time from admit time."""
    chunk = chunk.merge(
        cohort[
            [
                CohortHeader.HOSPITAL_ADMISSION_ID,
                CohortHeader.ADMIT_TIME,
                CohortHeader.DISCH_TIME,
            ]
        ],
        on=LabEventsHeader.HOSPITAL_ADMISSION_ID,
    )
    chunk[LabEventsHeader.CHART_TIME] = pd.to_datetime(
        chunk[LabEventsHeader.CHART_TIME]
    )
    chunk[LabEventsHeader.LAB_TIME_FROM_ADMIT] = (
        chunk[LabEventsHeader.CHART_TIME] - chunk[LabEventsHeader.ADMIT_TIME]
    )
    return chunk.dropna()


def make_procedures_feature_non_icu(cohort: pd.DataFrame) -> pd.DataFrame:
    module = load_hosp_procedures_icd()
    df_cohort = module.merge(
        cohort[
            [
                CohortHeader.HOSPITAL_ADMISSION_ID,
                CohortHeader.ADMIT_TIME,
                CohortHeader.DISCH_TIME,
            ]
        ],
        on=HospProceduresIcd.HOSPITAL_ADMISSION_ID,
    )
    df_cohort[NonIcuProceduresHeader.PROC_TIME_FROM_ADMIT] = (
        df_cohort[NonIcuProceduresHeader.CHART_DATE]
        - df_cohort[NonIcuProceduresHeader.ADMIT_TIME]
    )
    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    for v in [9, 10]:
        print(
            f"# Unique ICD{v} Procedures:  ",
            df_cohort.loc[df_cohort[NonIcuProceduresHeader.ICD_VERSION] == v][
                NonIcuProceduresHeader.ICD_CODE
            ]
            .dropna()
            .nunique(),
        )

    print(
        "\nValue counts of each ICD version:\n",
        df_cohort[NonIcuProceduresHeader.ICD_VERSION].value_counts(),
    )
    print("# Admissions:  ", df_cohort[CohortHeader.HOSPITAL_ADMISSION_ID].nunique())
    print("Total number of rows: ", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort


def make_hosp_prescriptions(cohort: pd.DataFrame) -> pd.DataFrame:
    adm = cohort[[CohortHeader.HOSPITAL_ADMISSION_ID, CohortHeader.ADMIT_TIME]]
    med = load_hosp_predictions()
    med = med.merge(adm, on=MedicationsHeader.HOSPITAL_ADMISSION_ID)
    med[MedicationsHeader.START_HOURS_FROM_ADMIT] = (
        med[MedicationsHeader.START_TIME] - med[CohortHeader.ADMIT_TIME]
    )
    med[MedicationsHeader.STOP_HOURS_FROM_ADMIT] = (
        med[NonIcuMedicationHeader.STOP_TIME] - med[CohortHeader.ADMIT_TIME]
    )

    # Normalize drug strings and remove potential duplicates

    med[NonIcuMedicationHeader.DRUG] = (
        med[NonIcuMedicationHeader.DRUG].fillna("").astype(str)
    )
    med[NonIcuMedicationHeader.DRUG] = med[NonIcuMedicationHeader.DRUG].apply(
        lambda x: x.lower().strip().replace(" ", "_") if not "" else ""
    )
    med[NonIcuMedicationHeader.DRUG] = (
        med[NonIcuMedicationHeader.DRUG].dropna().apply(lambda x: x.lower().strip())
    )
    med = ndc_meds(med, MAP_NDC_PATH)

    print("Number of unique type of drug: ", med.drug.nunique())
    print(
        "Number of unique type of drug (after grouping to use Non propietary names): ",
        med[NonIcuMedicationHeader.NON_PROPRIEATARY_NAME].nunique(),
    )
    print("Total number of rows: ", med.shape[0])
    print("# Admissions:  ", med[CohortHeader.HOSPITAL_ADMISSION_ID].nunique())

    return med


def ndc_meds(med, mapping: str) -> pd.DataFrame:
    # Convert any nan values to a dummy value
    med[HospPrescriptions.NDC] = med[HospPrescriptions.NDC].fillna(-1)

    # Ensures the decimal is removed from the ndc col
    med[HospPrescriptions.NDC] = med[HospPrescriptions.NDC].astype("Int64")

    # Read in NDC mapping table
    ndc_map = read_ndc_mapping(mapping)[
        ["productndc", "nonproprietaryname", "pharm_classes"]
    ]

    # Normalize the NDC codes in the mapping table so that they can be merged
    ndc_map.loc[:, "new_ndc"] = ndc_map["productndc"].apply(format_ndc_table)
    ndc_map = ndc_map.drop_duplicates(subset=["new_ndc", "nonproprietaryname"])
    med["new_ndc"] = med.ndc.apply(ndc_to_str)

    # Left join the med dataset to the mapping information
    med = med.merge(ndc_map, how="inner", left_on="new_ndc", right_on="new_ndc")

    # Function generates a list of EPCs, as a drug can have multiple EPCs
    med["EPC"] = med.pharm_classes.apply(get_EPC)

    return med
