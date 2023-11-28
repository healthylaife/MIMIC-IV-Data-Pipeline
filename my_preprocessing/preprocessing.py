import pandas as pd
from tqdm import tqdm
from uuid import uuid1
from collections import defaultdict
from typing import Union, List, Tuple
import numpy as np
from functools import partial
from multiprocessing import Pool
from glob import glob
import os
from my_preprocessing.raw_files import (
    ICU_CHART_EVENTS_PATH,
    ChartEvents,
    load_icu_outputevents,
    ICU_PROCEDURE_EVENTS_PATH,
    HOSP_PREDICTIONS_PATH,
    MAP_NDC_PATH,
    ICU_INPUT_EVENT_PATH,
    InputEvents,
    load_hosp_procedures_icd,
    load_hosp_lab_events,
    load_hosp_admissions,
    HospAdmissions,
)


def preproc_chartevents(cohort_path: str, chunksize=10000000) -> pd.DataFrame:
    """Function for processing hospital observations from a pickled cohort. optimized for memory efficiency"""

    # Only consider values in our cohort TODO: filter?
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["intime"])
    processed_chunks = []
    for chunk in tqdm(
        pd.read_csv(
            ICU_CHART_EVENTS_PATH,
            compression="gzip",
            usecols=[
                ChartEvents.STAY_ID.value,
                ChartEvents.CHARTTIME.value,
                ChartEvents.ITEMID.value,
                ChartEvents.VALUENUM.value,
                ChartEvents.VALUEOM.value,
            ],
            parse_dates=[ChartEvents.CHARTTIME.value],
            chunksize=chunksize,
        )
    ):
        chunk = chunk.dropna(subset=["valuenum"])
        chunk_merged = chunk.merge(
            cohort[["stay_id", "intime"]],
            how="inner",
            left_on="stay_id",
            right_on="stay_id",
        )
        chunk_merged["event_time_from_admit"] = (
            chunk_merged["charttime"] - chunk_merged["intime"]
        )
        chunk_merged.drop(["charttime", "intime"], axis=1, inplace=True)
        chunk_merged.dropna(inplace=True)
        chunk_merged.drop_duplicates(inplace=True)
        processed_chunks.append(chunk_merged)
    df_cohort = pd.concat(processed_chunks, ignore_index=True)
    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    return df_cohort


def preproc_output_events(cohort_path: str) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort.
    Function is structured to save memory when reading and transforming data."""
    outputevents = load_icu_outputevents()
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["intime"])
    df_cohort = outputevents.merge(
        cohort[["stay_id", "intime", "outtime"]],
        how="inner",
        left_on="stay_id",
        right_on="stay_id",
    )
    df_cohort["event_time_from_admit"] = df_cohort["charttime"] - df_cohort["intime"]
    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort


def preproc_icu_procedure_events(cohort_path: str) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    module = pd.read_csv(
        ICU_PROCEDURE_EVENTS_PATH,
        compression="gzip",
        usecols=["stay_id", "starttime", "itemid"],
        parse_dates=["starttime"],
    ).drop_duplicates()
    # Only consider values in our cohort
    cohort = pd.read_csv(cohort_path, compression="gzip", parse_dates=["intime"])
    df_cohort = module.merge(
        cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
        how="inner",
        left_on="stay_id",
        right_on="stay_id",
    )
    df_cohort["event_time_from_admit"] = df_cohort["starttime"] - df_cohort["intime"]

    df_cohort = df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.dropna().nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])
    return df_cohort


def ndc_meds(med, mapping: str) -> pd.DataFrame:
    # Convert any nan values to a dummy value
    med.ndc = med.ndc.fillna(-1)

    # Ensures the decimal is removed from the ndc col
    med.ndc = med.ndc.astype("Int64")

    # The NDC codes in the prescription dataset is the 11-digit NDC code, although codes are missing
    # their leading 0's because the column was interpreted as a float then integer; this function restores
    # the leading 0's, then obtains only the PRODUCT and MANUFACTUERER parts of the NDC code (first 9 digits)
    def to_str(ndc):
        if ndc < 0:  # dummy values are < 0
            return np.nan
        ndc = str(ndc)
        return (("0" * (11 - len(ndc))) + ndc)[0:-2]

    # The mapping table is ALSO incorrectly formatted for 11 digit NDC codes. An 11 digit NDC is in the
    # form of xxxxx-xxxx-xx for manufactuerer-product-dosage. The hyphens are in the correct spots, but
    # the number of digits within each section may not be 5-4-2, in which case we add leading 0's to each
    # to restore the 11 digit format. However, we only take the 5-4 sections, just like the to_str function
    def format_ndc_table(ndc):
        parts = ndc.split("-")
        return ("0" * (5 - len(parts[0])) + parts[0]) + (
            "0" * (4 - len(parts[1])) + parts[1]
        )

    def read_ndc_mapping2(map_path):
        ndc_map = pd.read_csv(map_path, header=0, delimiter="\t", encoding="latin1")
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
        ndc_map.columns = list(map(str.lower, ndc_map.columns))
        return ndc_map

    # Read in NDC mapping table
    ndc_map = read_ndc_mapping2(mapping)[
        ["productndc", "nonproprietaryname", "pharm_classes"]
    ]

    # Normalize the NDC codes in the mapping table so that they can be merged
    ndc_map["new_ndc"] = ndc_map.productndc.apply(format_ndc_table)
    ndc_map.drop_duplicates(subset=["new_ndc", "nonproprietaryname"], inplace=True)
    med["new_ndc"] = med.ndc.apply(to_str)

    # Left join the med dataset to the mapping information
    med = med.merge(ndc_map, how="inner", left_on="new_ndc", right_on="new_ndc")

    # In NDC mapping table, the pharm_class col is structured as a text string, separating different pharm classes from eachother
    # This can be [PE], [EPC], and others, but we're interested in EPC. Luckily, between each commas, it states if a phrase is [EPC]
    # So, we just string split by commas and keep phrases containing "[EPC]"
    def get_EPC(s):
        """Gets the Established Pharmacologic Class (EPC) from the mapping table"""
        if type(s) != str:
            return np.nan
        words = s.split(",")
        return [x for x in words if "[EPC]" in x]

    # Function generates a list of EPCs, as a drug can have multiple EPCs
    med["EPC"] = med.pharm_classes.apply(get_EPC)

    return med


def preprocess_hosp_prescriptions(cohort_path: str) -> pd.DataFrame:
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


def preprocess_icu_input_events(cohort_path: str) -> pd.DataFrame:
    adm = pd.read_csv(
        cohort_path,
        usecols=["hadm_id", "stay_id", "intime"],
        parse_dates=["intime"],
    )
    med = pd.read_csv(
        ICU_INPUT_EVENT_PATH,
        compression="gzip",
        usecols=[
            InputEvents.SUBJECT_ID.value,
            InputEvents.STAY_ID.value,
            InputEvents.ITEMID.value,
            InputEvents.STARTTIME.value,
            InputEvents.ENDTIME.value,
            InputEvents.RATE.value,
            InputEvents.AMOUNT.value,
            InputEvents.ORDERID.value,
        ],
        parse_dates=[InputEvents.STARTTIME.value, InputEvents.ENDTIME.value],
    )
    med = med.merge(
        adm, left_on=InputEvents.STAY_ID.value, right_on="stay_id", how="inner"
    )
    med["start_hours_from_admit"] = med[InputEvents.STARTTIME.value] - med["intime"]
    med["stop_hours_from_admit"] = med[InputEvents.ENDTIME.value] - med["intime"]
    med = med.dropna()
    print("# of unique type of drug: ", med[InputEvents.ITEMID.value].nunique())
    print("# Admissions:  ", med[InputEvents.STAY_ID.value].nunique())
    print("# Total rows", med.shape[0])

    return med


def hadm_imputer(
    charttime: pd._libs.tslibs.timestamps.Timestamp,
    hadm_old: Union[str, float],
    hadm_ids_w_timestamps: List[
        Tuple[
            str,
            pd._libs.tslibs.timestamps.Timestamp,
            pd._libs.tslibs.timestamps.Timestamp,
        ]
    ],
) -> Tuple[str, pd._libs.tslibs.timestamps.Timestamp]:
    # if old hadm exists use that
    if not np.isnan(hadm_old):
        hadm_old = int(hadm_old)
        admtime, dischtime = [
            [adm_time, disch_time]
            for h_id, adm_time, disch_time in hadm_ids_w_timestamps
            if h_id == hadm_old
        ][0]
        return (
            hadm_old,
            admtime.strftime("%Y-%m-%d %H:%M:%S"),
            dischtime.strftime("%Y-%m-%d %H:%M:%S"),
        )
    # get the difference between this lab event charttime and all admit times for this subject_id
    hadm_ids_w_timestamps = [
        [
            hadm_id,
            admittime.strftime("%Y-%m-%d %H:%M:%S"),
            dischtime.strftime("%Y-%m-%d %H:%M:%S"),
            charttime.normalize() - admittime.normalize(),
            charttime.normalize() - dischtime.normalize(),
        ]
        for hadm_id, admittime, dischtime in hadm_ids_w_timestamps
    ]
    # the lab charttime must be in between admit time and discharge time
    hadm_ids_w_timestamps = [
        x for x in hadm_ids_w_timestamps if x[3].days >= 0 and x[4].days <= 0
    ]
    # there should be exactly one hadm_id that satisfies this criteria
    # if multiple, select the hadm id with admittime closest to the lab event charttime
    hadm_ids_w_timestamps = sorted(hadm_ids_w_timestamps, key=lambda x: x[3])
    if not hadm_ids_w_timestamps:
        return None, None, None
    return_data = hadm_ids_w_timestamps[0][:3]
    return return_data


def impute_missing_hadm_ids(
    lab_table: pd.DataFrame, subject_hadm_admittime_tracker: defaultdict
) -> pd.DataFrame:
    list_rows_lab = []
    all_lab_cols = lab_table.columns
    for row in lab_table.itertuples():
        existing_data = {
            col_name: row.__getattribute__(col_name) for col_name in all_lab_cols
        }
        new_hadm_id, new_admittime, new_dischtime = hadm_imputer(
            row.charttime,
            row.hadm_id,
            subject_hadm_admittime_tracker.get(
                row.subject_id, []
            ),  # using get as defaultdict will create key if does not exist
        )
        existing_data["hadm_id_new"] = new_hadm_id
        existing_data["admittime"] = new_admittime
        existing_data["dischtime"] = new_dischtime
        list_rows_lab.append(existing_data)
    tab_name = str(uuid1())
    pd.DataFrame(list_rows_lab).to_csv(f"{tab_name}.csv")


def impute_hadm_ids(
    lab_table: Union[str, pd.DataFrame], admission_table: Union[str, pd.DataFrame]
) -> pd.DataFrame:
    if isinstance(lab_table, str):
        lab_table = pd.read_csv(lab_table)
    if isinstance(admission_table, str):
        admission_table = pd.read_csv(admission_table)
    lab_table["charttime"] = pd.to_datetime(lab_table.charttime)
    admission_table["admittime"] = pd.to_datetime(admission_table.admittime)
    admission_table["dischtime"] = pd.to_datetime(admission_table.dischtime)
    # get a dictionary like this->
    """ {
        "sub_id_1": [["hadm_1", "admittime1", "dischtime1"], ["hadm_2", "admittime2", "dischtime2"]],
        "sub_id_2": [["hadm_1", "admittime1", "dischtime1"], ["hadm_2", "admittime2", "dischtime2"]],
        ...
    """
    subject_hadm_admittime_tracker = defaultdict(list)
    for row in admission_table.itertuples():
        subject_hadm_admittime_tracker[row.subject_id].append(
            [row.hadm_id, row.admittime, row.dischtime]
        )
    lab_size = lab_table.shape[0]
    chunks = 100
    tab_size = lab_size // chunks
    lab_table_chunks = []
    for i in range(chunks):
        st, en = i * tab_size, (i + 1) * tab_size
        lab_table_chunks.append(lab_table[st:en])
    if lab_size - chunks * tab_size > 0:
        lab_table_chunks.append(lab_table[chunks * tab_size :])
    # we dont need the original lab table as it is chunkified, hope is to save memory as lab table is huge
    del lab_table
    impute_missing_hadm_ids_w_lookup = partial(
        impute_missing_hadm_ids,
        subject_hadm_admittime_tracker=subject_hadm_admittime_tracker,
    )
    # print(impute_missing_hadm_ids_w_lookup)
    # print(len(lab_table_chunks))
    with Pool(8) as p:
        p.map(impute_missing_hadm_ids_w_lookup, lab_table_chunks)
    all_csvs = glob("*.csv")
    lab_tab = pd.DataFrame()
    for csv in all_csvs:
        lab_tab = pd.concat([lab_tab, pd.read_csv(csv)])
        os.remove(csv)
    return lab_tab


def preproc_labs_events_features(cohort_path: str) -> pd.DataFrame:
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


def preproc_hosp_procedures_icd(cohort_path: str) -> pd.DataFrame:
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
