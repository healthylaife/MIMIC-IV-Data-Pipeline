import pandas as pd
from tqdm import tqdm
from pipeline.file_info.preproc.cohort import COHORT_PATH, CohortHeader


def generate_admission_cohort(cohort_output: str) -> pd.DataFrame:
    data = pd.read_csv(
        COHORT_PATH / f"{cohort_output}.csv.gz",
        compression="gzip",
    )
    for col in [CohortHeader.ADMIT_TIME, CohortHeader.DISCH_TIME]:
        data[col] = pd.to_datetime(data[col])
    data[CohortHeader.LOS] = (
        int(
            data[CohortHeader.DISCH_TIME] - data[CohortHeader.ADMIT_TIME]
        ).dt.total_seconds()
        / 3600
    )
    data = data[data[CohortHeader.LOS] > 0]
    data[CohortHeader.AGE] = data[CohortHeader.AGE].astype(int)
    return data


def generate_diag(cohort, use_icu):
    diag = pd.read_csv(
        PREPROC_DIAG_ICU_PATH if use_icu else PREPROC_DIAG_PATH,
        compression="gzip",
    )
    diag = diag[
        diag[DiagnosesHeader.HOSPITAL_ADMISSION_ID].isin(
            cohort[CohortHeader.HOSPITAL_ADMISSION_ID]
        )
    ]
    diag_per_adm = diag.groupby(DiagnosesHeader.HOSPITAL_ADMISSION_ID).size().max()
    return diag, diag_per_adm


def generate_proc(self, use_icu):
    proc = pd.read_csv(
        PREPROC_PROC_ICU_PATH if use_icu else PREPROC_PROC_PATH,
        compression="gzip",
    )
    proc = proc[
        proc[ProceduresHeader.HOSPITAL_ADMISSION_ID].isin(self.cohort["hadm_id"])
    ]
    proc[["start_days", "dummy", "start_hours"]] = proc[
        "proc_time_from_admit"
    ].str.split(" ", expand=True)
    proc[["start_hours", "min", "sec"]] = proc["start_hours"].str.split(
        ":", expand=True
    )
    proc["start_time"] = pd.to_numeric(proc["start_days"]) * 24 + pd.to_numeric(
        proc["start_hours"]
    )
    proc = proc.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
    proc = proc[proc["start_time"] >= 0]

    ###Remove where event time is after discharge time
    proc = pd.merge(proc, self.cohort[["hadm_id", "los"]], on="hadm_id", how="left")
    proc["sanity"] = proc["los"] - proc["start_time"]
    proc = proc[proc["sanity"] > 0]
    del proc["sanity"]

    return proc


def generate_labs(self):
    chunksize = 10000000
    final = pd.DataFrame()
    for labs in tqdm(
        pd.read_csv(
            "./data/features/preproc_labs.csv.gz",
            compression="gzip",
            header=0,
            index_col=None,
            chunksize=chunksize,
        )
    ):
        labs = labs[labs["hadm_id"].isin(self.data["hadm_id"])]
        labs[["start_days", "dummy", "start_hours"]] = labs[
            "lab_time_from_admit"
        ].str.split(" ", expand=True)
        labs[["start_hours", "min", "sec"]] = labs["start_hours"].str.split(
            ":", expand=True
        )
        labs["start_time"] = pd.to_numeric(labs["start_days"]) * 24 + pd.to_numeric(
            labs["start_hours"]
        )
        labs = labs.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
        labs = labs[labs["start_time"] >= 0]

        ###Remove where event time is after discharge time
        labs = pd.merge(labs, self.data[["hadm_id", "los"]], on="hadm_id", how="left")
        labs["sanity"] = labs["los"] - labs["start_time"]
        labs = labs[labs["sanity"] > 0]
        del labs["sanity"]

        if final.empty:
            final = labs
        else:
            final = pd.concat([final, labs], ignore_index=True)

    return final


def generate_meds(self):
    meds = pd.read_csv(
        "./data/features/preproc_med.csv.gz",
        compression="gzip",
        header=0,
        index_col=None,
    )
    meds[["start_days", "dummy", "start_hours"]] = meds[
        "start_hours_from_admit"
    ].str.split(" ", expand=True)
    meds[["start_hours", "min", "sec"]] = meds["start_hours"].str.split(
        ":", -1, expand=True
    )
    meds["start_time"] = pd.to_numeric(meds["start_days"]) * 24 + pd.to_numeric(
        meds["start_hours"]
    )
    meds[["start_days", "dummy", "start_hours"]] = meds[
        "stop_hours_from_admit"
    ].str.split(" ", expand=True)
    meds[["start_hours", "min", "sec"]] = meds["start_hours"].str.split(
        ":", expand=True
    )
    meds["stop_time"] = pd.to_numeric(meds["start_days"]) * 24 + pd.to_numeric(
        meds["start_hours"]
    )
    meds = meds.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
    #####Sanity check
    meds["sanity"] = meds["stop_time"] - meds["start_time"]
    meds = meds[meds["sanity"] > 0]
    del meds["sanity"]
    #####Select hadm_id as in main file
    meds = meds[meds["hadm_id"].isin(self.data["hadm_id"])]
    meds = pd.merge(meds, self.data[["hadm_id", "los"]], on="hadm_id", how="left")

    #####Remove where start time is after end of visit
    meds["sanity"] = meds["los"] - meds["start_time"]
    meds = meds[meds["sanity"] > 0]
    del meds["sanity"]
    ####Any stop_time after end of visit is set at end of visit
    meds.loc[meds["stop_time"] > meds["los"], "stop_time"] = meds.loc[
        meds["stop_time"] > meds["los"], "los"
    ]
    del meds["los"]

    meds["dose_val_rx"] = meds["dose_val_rx"].apply(pd.to_numeric, errors="coerce")

    return meds
