import pandas as pd
from tqdm import tqdm
from my_preprocessing.icd_conversion import standardize_icd
from my_preprocessing.uom_conversion import drop_wrong_uom
from my_preprocessing.outlier_removal import outlier_imputation
from my_preprocessing.raw_files import (
    load_hosp_diagnosis_icd,
)
from my_preprocessing.preproc_files import (
    COHORT_PATH,
    PREPROC_DIAG_PATH,
    PREPROC_PROC_PATH,
    PREPROC_PROC_ICU_PATH,
    PREPROC_MED_ICU_PATH,
    PREPROC_LABS_PATH,
    PREPROC_OUT_ICU_PATH,
    PREPROC_DIAG_ICU_PATH,
    PREPROC_CHART_ICU_PATH,
    PREPROC_MED_PATH,
)
from my_preprocessing.icu_features import (
    make_output_events,
    make_chart_events,
    make_icu_procedure_events,
    make_icu_input_events,
)
from my_preprocessing.hosp_features import (
    make_labs_events_features,
    make_hosp_prescriptions,
    make_hosp_procedures_icd,
)


DIAGNOSIS_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "icd_code",
    "root_icd10_convert",
    "root",
]

DIAGNOSIS_NON_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "root_icd10_convert",
    "root",
]

OUTPUT_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "charttime",
    "intime",
    "event_time_from_admit",
]

PROCEDURES_ICD_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "starttime",
    "intime",
    "event_time_from_admit",
]

PROCEDURES_ICD_NON_ICU_HEADERS = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "icd_version",
    "chartdate",
    "admittime",
    "proc_time_from_admit",
]

LAB_EVENTS_HEADERS = [
    "subject_id",
    "hadm_id",
    "charttime",
    "itemid",
    "admittime",
    "lab_time_from_admit",
    "valuenum",
]

PRESCRIPTIONS_HEADERS = [
    "subject_id",
    "hadm_id",
    "starttime",
    "stoptime",
    "drug",
    "nonproprietaryname",
    "start_hours_from_admit",
    "stop_hours_from_admit",
    "dose_val_rx",
]

INPUT_EVENTS_HEADERS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "itemid",
    "starttime",
    "endtime",
    "start_hours_from_admit",
    "stop_hours_from_admit",
    "rate",
    "amount",
    "orderid",
]

CHART_EVENT_HEADERS = ["stay_id", "itemid", "event_time_from_admit", "valuenum"]


def save_diag_features(cohort_output: str, use_icu: bool) -> pd.DataFrame:
    print("[EXTRACTING DIAGNOSIS DATA]")
    hosp_diagnose = load_hosp_diagnosis_icd()
    admission_cohort = pd.read_csv(
        COHORT_PATH / (cohort_output + ".csv.gz"), compression="gzip"
    )
    admissions_cohort_cols = (
        ["hadm_id", "stay_id", "label"] if use_icu else ["hadm_id", "label"]
    )
    diag = hosp_diagnose.merge(
        admission_cohort[admissions_cohort_cols],
        how="inner",
        left_on="hadm_id",
        right_on="hadm_id",
    )
    cols = DIAGNOSIS_ICU_HEADERS if use_icu else DIAGNOSIS_NON_ICU_HEADERS
    diag = standardize_icd(diag)[cols]
    diag.to_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    return diag


def save_output_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING OUPTPUT EVENTS DATA]")
    out = make_output_events(COHORT_PATH / (cohort_output + ".csv.gz"))
    out = out[OUTPUT_ICU_HEADERS]
    out.to_csv(PREPROC_OUT_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    return out


def save_chart_events_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING CHART EVENTS DATA]")
    chart = make_chart_events(COHORT_PATH / (cohort_output + ".csv.gz"))
    chart = drop_wrong_uom(chart, 0.95)
    chart = chart[CHART_EVENT_HEADERS]
    chart.to_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
    return chart


def save_icu_procedures_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING PROCEDURES DATA]")
    proc = make_icu_procedure_events(COHORT_PATH / (cohort_output + ".csv.gz"))
    proc = proc[PROCEDURES_ICD_ICU_HEADERS]
    proc.to_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    return proc


def save_hosp_procedures_icd_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING PROCEDURES DATA]")
    proc = make_hosp_procedures_icd(COHORT_PATH / (cohort_output + ".csv.gz"))
    proc = proc[PROCEDURES_ICD_NON_ICU_HEADERS]
    proc.to_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    return proc


def save_icu_input_events_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING MEDICATIONS DATA]")
    med = make_icu_input_events(COHORT_PATH / (cohort_output + ".csv.gz"))
    med = med[INPUT_EVENTS_HEADERS]
    med.to_csv(PREPROC_MED_ICU_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    return med


def save_lab_events_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING LABS DATA]")
    labevents = make_labs_events_features(COHORT_PATH / (cohort_output + ".csv.gz"))
    labevents = drop_wrong_uom(labevents, 0.95)
    labevents = labevents[LAB_EVENTS_HEADERS]
    labevents.to_csv(PREPROC_LABS_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED LABS DATA]")
    return labevents


def save_hosp_prescriptions_features(cohort_output: str) -> pd.DataFrame:
    print("[EXTRACTING MEDICATIONS DATA]")
    prescriptions = make_hosp_prescriptions(COHORT_PATH / (cohort_output + ".csv.gz"))
    prescriptions = prescriptions[PRESCRIPTIONS_HEADERS]
    prescriptions.to_csv(PREPROC_MED_PATH, compression="gzip")
    print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    return prescriptions


def feature_icu(
    cohort_output,
    diag_flag=True,
    out_flag=True,
    chart_flag=True,
    proc_flag=True,
    med_flag=True,
):
    diag, out, chart, proc, med = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    if diag_flag:
        diag = save_diag_features(cohort_output, use_icu=True)

    if out_flag:
        out = save_output_features(cohort_output)

    if chart_flag:
        chart = save_chart_events_features(cohort_output)

    if proc_flag:
        proc = save_icu_procedures_features(cohort_output)

    if med_flag:
        med = save_icu_input_events_features(cohort_output)
    return diag, out, chart, proc, med


def feature_non_icu(
    cohort_output, diag_flag=True, lab_flag=True, proc_flag=True, med_flag=True
):
    diag, lab, proc, med = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    if diag_flag:
        diag = save_diag_features(cohort_output, use_icu=False)

    if lab_flag:
        lab = save_lab_events_features(cohort_output)

    if proc_flag:
        proc = save_hosp_procedures_icd_features(cohort_output)

    if med_flag:
        med = save_hosp_prescriptions_features(cohort_output)
    return diag, lab, proc, med


## NEW CODE


def generate_summary_icu(diag_flag, proc_flag, med_flag, out_flag, chart_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
        freq = (
            diag.groupby(["stay_id", "new_icd_code"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["new_icd_code"])["mean_frequency"].mean().reset_index()
        total = diag.groupby("new_icd_code").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="new_icd_code", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/diag_summary.csv", index=False)
        summary["new_icd_code"].to_csv("./data/summary/diag_features.csv", index=False)

    if med_flag:
        med = pd.read_csv(PREPROC_MED_ICU_PATH, compression="gzip")
        freq = (
            med.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

        missing = (
            med[med["amount"] == 0]
            .groupby("itemid")
            .size()
            .reset_index(name="missing_count")
        )
        total = med.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="itemid", how="right")
        summary = pd.merge(freq, summary, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/med_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/med_features.csv", index=False)

    if proc_flag:
        proc = pd.read_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
        freq = (
            proc.groupby(["stay_id", "itemid"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = proc.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/proc_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/proc_features.csv", index=False)

    if out_flag:
        out = pd.read_csv(PREPROC_OUT_ICU_PATH, compression="gzip")
        freq = (
            out.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = out.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/out_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/out_features.csv", index=False)

    if chart_flag:
        chart = pd.read_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
        freq = (
            chart.groupby(["stay_id", "itemid"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

        missing = (
            chart[chart["valuenum"] == 0]
            .groupby("itemid")
            .size()
            .reset_index(name="missing_count")
        )
        total = chart.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="itemid", how="right")
        summary = pd.merge(freq, summary, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/chart_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/chart_features.csv", index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")


def features_selection_icu(
    cohort_output,
    diag_flag,
    proc_flag,
    med_flag,
    out_flag,
    chart_flag,
    group_diag,
    group_med,
    group_proc,
    group_out,
    group_chart,
):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
            features = pd.read_csv("./data/summary/diag_features.csv")
            diag = diag[diag["new_icd_code"].isin(features["new_icd_code"].unique())]

            print("Total number of rows", diag.shape[0])
            diag.to_csv(
                PREPROC_DIAG_ICU_PATH,
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if med_flag:
        if group_med:
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv(PREPROC_MED_ICU_PATH, compression="gzip")
            features = pd.read_csv("./data/summary/med_features.csv")
            med = med[med["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", med.shape[0])
            med.to_csv(
                PREPROC_MED_ICU_PATH,
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv(PREPROC_PROC_ICU_PATH, compression="gzip")
            features = pd.read_csv("./data/summary/proc_features.csv")
            proc = proc[proc["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", proc.shape[0])
            proc.to_csv(
                PREPROC_PROC_ICU_PATH,
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if out_flag:
        if group_out:
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv(PREPROC_OUT_ICU_PATH, compression="gzip")
            features = pd.read_csv("./data/summary/out_features.csv", header=0)
            out = out[out["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", out.shape[0])
            out.to_csv(
                PREPROC_OUT_ICU_PATH,
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")

    if chart_flag:
        if group_chart:
            print("[FEATURE SELECTION CHART EVENTS DATA]")

            chart = pd.read_csv(
                PREPROC_CHART_ICU_PATH,
                compression="gzip",
                index_col=None,
            )

            features = pd.read_csv("./data/summary/chart_features.csv", header=0)
            chart = chart[chart["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", chart.shape[0])
            chart.to_csv(
                PREPROC_CHART_ICU_PATH,
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")


def preprocess_features_icu(
    cohort_output,
    diag_flag,
    group_diag,
    chart_flag,
    clean_chart,
    impute_outlier_chart,
    thresh,
    left_thresh,
):
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(PREPROC_DIAG_ICU_PATH, compression="gzip")
        if group_diag == "Keep both ICD-9 and ICD-10 codes":
            diag["new_icd_code"] = diag["icd_code"]
        if group_diag == "Convert ICD-9 to ICD-10 codes":
            diag["new_icd_code"] = diag["root_icd10_convert"]
        if group_diag == "Convert ICD-9 to ICD-10 and group ICD-10 codes":
            diag["new_icd_code"] = diag["root"]

        diag = diag[["subject_id", "hadm_id", "stay_id", "new_icd_code"]].dropna()
        print("Total number of rows", diag.shape[0])
        diag.to_csv(PREPROC_DIAG_ICU_PATH, compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if chart_flag:
        if clean_chart:
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv(PREPROC_CHART_ICU_PATH, compression="gzip")
            chart = outlier_imputation(
                chart, "itemid", "valuenum", thresh, left_thresh, impute_outlier_chart
            )

            print("Total number of rows", chart.shape[0])
            chart.to_csv(
                PREPROC_CHART_ICU_PATH,
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")


def preprocess_features_hosp(
    cohort_output,
    diag_flag,
    proc_flag,
    med_flag,
    lab_flag,
    group_diag,
    group_med,
    group_proc,
    clean_labs,
    impute_labs,
    thresh,
    left_thresh,
):
    # print(thresh)
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(PREPROC_DIAG_PATH, compression="gzip")
        if group_diag == "Keep both ICD-9 and ICD-10 codes":
            diag["new_icd_code"] = diag["icd_code"]
        if group_diag == "Convert ICD-9 to ICD-10 codes":
            diag["new_icd_code"] = diag["root_icd10_convert"]
        if group_diag == "Convert ICD-9 to ICD-10 and group ICD-10 codes":
            diag["new_icd_code"] = diag["root"]

        diag = diag[["subject_id", "hadm_id", "new_icd_code"]].dropna()
        print("Total number of rows", diag.shape[0])
        diag.to_csv(PREPROC_DIAG_PATH, compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if med_flag:
        print("[PROCESSING MEDICATIONS DATA]")
        if group_med:
            med = pd.read_csv(PREPROC_MED_PATH, compression="gzip")
            if group_med:
                med["drug_name"] = med["nonproprietaryname"]
            else:
                med["drug_name"] = med["drug"]
            med = med[
                [
                    "subject_id",
                    "hadm_id",
                    "starttime",
                    "stoptime",
                    "drug_name",
                    "start_hours_from_admit",
                    "stop_hours_from_admit",
                    "dose_val_rx",
                ]
            ].dropna()
            print("Total number of rows", med.shape[0])
            med.to_csv(PREPROC_MED_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    if proc_flag:
        print("[PROCESSING PROCEDURES DATA]")
        proc = pd.read_csv(
            PREPROC_PROC_PATH,
            compression="gzip",
        )
        if group_proc == "ICD-9 and ICD-10":
            proc = proc[
                [
                    "subject_id",
                    "hadm_id",
                    "icd_code",
                    "chartdate",
                    "admittime",
                    "proc_time_from_admit",
                ]
            ]
            print("Total number of rows", proc.shape[0])
            proc.dropna().to_csv(PREPROC_PROC_PATH, compression="gzip", index=False)
        elif group_proc == "ICD-10":
            proc = proc.loc[proc.icd_version == 10][
                [
                    "subject_id",
                    "hadm_id",
                    "icd_code",
                    "chartdate",
                    "admittime",
                    "proc_time_from_admit",
                ]
            ].dropna()
            print("Total number of rows", proc.shape[0])
            proc.to_csv(PREPROC_PROC_PATH, compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if lab_flag:
        if clean_labs:
            print("[PROCESSING LABS DATA]")
            labs = pd.read_csv(PREPROC_LABS_PATH, compression="gzip")
            labs = outlier_imputation(
                labs, "itemid", "valuenum", thresh, left_thresh, impute_labs
            )

            print("Total number of rows", labs.shape[0])
            labs.to_csv(PREPROC_LABS_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")


def generate_summary_hosp(diag_flag, proc_flag, med_flag, lab_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv(PREPROC_DIAG_PATH, compression="gzip", header=0)
        freq = (
            diag.groupby(["hadm_id", "new_icd_code"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["new_icd_code"])["mean_frequency"].mean().reset_index()
        total = diag.groupby("new_icd_code").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="new_icd_code", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/diag_summary.csv", index=False)
        summary["new_icd_code"].to_csv("./data/summary/diag_features.csv", index=False)

    if med_flag:
        med = pd.read_csv(PREPROC_MED_PATH, compression="gzip")
        freq = (
            med.groupby(["hadm_id", "drug_name"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["drug_name"])["mean_frequency"].mean().reset_index()

        missing = (
            med[med["dose_val_rx"] == 0]
            .groupby("drug_name")
            .size()
            .reset_index(name="missing_count")
        )
        total = med.groupby("drug_name").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="drug_name", how="right")
        summary = pd.merge(freq, summary, on="drug_name", how="right")
        summary["missing%"] = 100 * (summary["missing_count"] / summary["total_count"])
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/med_summary.csv", index=False)
        summary["drug_name"].to_csv("./data/summary/med_features.csv", index=False)

    if proc_flag:
        proc = pd.read_csv(PREPROC_PROC_PATH, compression="gzip")
        freq = (
            proc.groupby(["hadm_id", "icd_code"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["icd_code"])["mean_frequency"].mean().reset_index()
        total = proc.groupby("icd_code").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="icd_code", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/proc_summary.csv", index=False)
        summary["icd_code"].to_csv("./data/summary/proc_features.csv", index=False)

    if lab_flag:
        chunksize = 10000000
        labs = pd.DataFrame()
        for chunk in tqdm(
            pd.read_csv(
                PREPROC_LABS_PATH,
                compression="gzip",
                index_col=None,
                chunksize=chunksize,
            )
        ):
            if labs.empty:
                labs = chunk
            else:
                labs = labs.append(chunk, ignore_index=True)
        freq = (
            labs.groupby(["hadm_id", "itemid"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

        missing = (
            labs[labs["valuenum"] == 0]
            .groupby("itemid")
            .size()
            .reset_index(name="missing_count")
        )
        total = labs.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="itemid", how="right")
        summary = pd.merge(freq, summary, on="itemid", how="right")
        summary["missing%"] = 100 * (summary["missing_count"] / summary["total_count"])
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/labs_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/labs_features.csv", index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")


def features_selection_hosp(
    cohort_output,
    diag_flag,
    proc_flag,
    med_flag,
    lab_flag,
    group_diag,
    group_med,
    group_proc,
    clean_labs,
):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv(PREPROC_DIAG_PATH, compression="gzip")
            features = pd.read_csv("./data/summary/diag_features.csv")
            diag = diag[diag["new_icd_code"].isin(features["new_icd_code"].unique())]

            print("Total number of rows", diag.shape[0])
            diag.to_csv(PREPROC_DIAG_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if med_flag:
        if group_med:
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv(PREPROC_MED_PATH, compression="gzip")
            features = pd.read_csv("./data/summary/med_features.csv")
            med = med[med["drug_name"].isin(features["drug_name"].unique())]
            print("Total number of rows", med.shape[0])
            med.to_csv(PREPROC_MED_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv(PREPROC_PROC_PATH, compression="gzip")
            features = pd.read_csv("./data/summary/proc_features.csv")
            proc = proc[proc["icd_code"].isin(features["icd_code"].unique())]
            print("Total number of rows", proc.shape[0])
            proc.to_csv(PREPROC_PROC_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if lab_flag:
        if clean_labs:
            print("[FEATURE SELECTION LABS DATA]")
            chunksize = 10000000
            labs = pd.DataFrame()
            for chunk in tqdm(
                pd.read_csv(
                    PREPROC_LABS_PATH,
                    compression="gzip",
                    index_col=None,
                    chunksize=chunksize,
                )
            ):
                if labs.empty:
                    labs = chunk
                else:
                    labs = labs.append(chunk, ignore_index=True)
            features = pd.read_csv("./data/summary/labs_features.csv", header=0)
            labs = labs[labs["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", labs.shape[0])
            labs.to_csv(PREPROC_LABS_PATH, compression="gzip", index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")
