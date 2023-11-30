import pandas as pd
from tqdm import tqdm
from my_preprocessing.raw_file_info import (
    HOSP_PREDICTIONS_PATH,
    MAP_NDC_PATH,
    load_hosp_procedures_icd,
    load_hosp_lab_events,
    load_hosp_admissions,
    HospAdmissions,
)
from my_preprocessing.admission_imputer import impute_hadm_ids
from my_preprocessing.ndc import (
    format_ndc_table,
    read_ndc_mapping2,
    ndc_to_str,
    get_EPC,
)


def make_labs_events_features(cohort_path: str) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["admittime"])
    admissions = load_hosp_admissions()[
        [
            HospAdmissions.PATIENT_ID.value,
            HospAdmissions.ID.value,
            HospAdmissions.ADMITTIME.value,
            HospAdmissions.DISCHTIME.value,
        ]
    ]
    chunksize = 10000000
    usecols = ["itemid", "subject_id", "hadm_id", "charttime", "valuenum", "valueuom"]
    processed_chunks = []
    for chunk in tqdm(load_hosp_lab_events(chunksize=chunksize, use_cols=usecols)):
        chunk = chunk.dropna(subset=["valuenum"])
        chunk.loc[:, "valueuom"] = chunk["valueuom"].fillna(0)

        # to remove?
        chunk = chunk[chunk["subject_id"].isin(cohort["subject_id"].unique())]
        # Split and impute hadm_ids
        chunk_with_hadm = chunk[chunk["hadm_id"].notna()]
        chunk_no_hadm = chunk[chunk["hadm_id"].isna()]

        chunk_without_hadm = impute_hadm_ids(
            chunk_no_hadm[
                ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
            ].copy(),
            admissions,
        )
        chunk_without_hadm["hadm_id"] = chunk_without_hadm["hadm_id_new"]
        chunk_without_hadm = chunk_without_hadm[
            ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
        ]
        chunk = pd.concat([chunk_with_hadm, chunk_without_hadm], ignore_index=True)
        # Merge with cohort data
        chunk = chunk.merge(cohort[["hadm_id", "admittime", "dischtime"]], on="hadm_id")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["lab_time_from_admit"] = chunk["charttime"] - chunk["admittime"]

        chunk.dropna(inplace=True)
        processed_chunks.append(chunk)

    df_cohort = pd.concat(processed_chunks, ignore_index=True)
    print("# Itemid: ", df_cohort.itemid.nunique())
    print("# Admissions: ", df_cohort.hadm_id.nunique())
    print("Total number of rows: ", df_cohort.shape[0])
    return df_cohort


def make_hosp_procedures_icd(cohort_path: str) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["admittime"])
    module = load_hosp_procedures_icd()
    df_cohort = module.merge(
        cohort[["hadm_id", "admittime", "dischtime"]],
        how="inner",
        left_on="hadm_id",
        right_on="hadm_id",
    )
    df_cohort["proc_time_from_admit"] = df_cohort["chartdate"] - df_cohort["admittime"]
    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    print(
        "# Unique ICD9 Procedures:  ",
        df_cohort.loc[df_cohort.icd_version == 9].icd_code.dropna().nunique(),
    )
    print(
        "# Unique ICD10 Procedures: ",
        df_cohort.loc[df_cohort.icd_version == 10].icd_code.dropna().nunique(),
    )

    print("\nValue counts of each ICD version:\n", df_cohort.icd_version.value_counts())
    print("# Admissions:  ", df_cohort.hadm_id.nunique())
    print("Total number of rows: ", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort


def make_hosp_prescriptions(cohort_path: str) -> pd.DataFrame:
    adm = pd.read_csv(
        cohort_path, usecols=["hadm_id", "admittime"], parse_dates=["admittime"]
    )
    med = pd.read_csv(
        HOSP_PREDICTIONS_PATH,
        compression="gzip",
        usecols=[
            "subject_id",
            "hadm_id",
            "drug",
            "starttime",
            "stoptime",
            "ndc",
            "dose_val_rx",
        ],
        parse_dates=["starttime", "stoptime"],
    )
    med = med.merge(adm, left_on="hadm_id", right_on="hadm_id", how="inner")
    med["start_hours_from_admit"] = med["starttime"] - med["admittime"]
    med["stop_hours_from_admit"] = med["stoptime"] - med["admittime"]

    # Normalize drug strings and remove potential duplicates

    med.drug = med.drug.fillna("").astype(str)
    med.drug = med.drug.apply(
        lambda x: x.lower().strip().replace(" ", "_") if not "" else ""
    )
    med.drug = med.drug.dropna().apply(lambda x: x.lower().strip())
    med = ndc_meds(med, MAP_NDC_PATH)

    print("Number of unique type of drug: ", med.drug.nunique())
    print(
        "Number of unique type of drug (after grouping to use Non propietary names): ",
        med.nonproprietaryname.nunique(),
    )
    print("Total number of rows: ", med.shape[0])
    print("# Admissions:  ", med.hadm_id.nunique())

    return med


def ndc_meds(med, mapping: str) -> pd.DataFrame:
    # Convert any nan values to a dummy value
    med.ndc = med.ndc.fillna(-1)

    # Ensures the decimal is removed from the ndc col
    med.ndc = med.ndc.astype("Int64")

    # Read in NDC mapping table
    ndc_map = read_ndc_mapping2(mapping)[
        ["productndc", "nonproprietaryname", "pharm_classes"]
    ]

    # Normalize the NDC codes in the mapping table so that they can be merged
    ndc_map["new_ndc"] = ndc_map.productndc.apply(format_ndc_table)
    ndc_map.drop_duplicates(subset=["new_ndc", "nonproprietaryname"], inplace=True)
    med["new_ndc"] = med.ndc.apply(ndc_to_str)

    # Left join the med dataset to the mapping information
    med = med.merge(ndc_map, how="inner", left_on="new_ndc", right_on="new_ndc")

    # Function generates a list of EPCs, as a drug can have multiple EPCs
    med["EPC"] = med.pharm_classes.apply(get_EPC)

    return med
