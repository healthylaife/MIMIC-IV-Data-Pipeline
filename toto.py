from pathlib import Path
import pandas as pd

RAW_DIR = "raw_data"
MIMIC_DIR = "mimicciv"
PREPROC_DIR = "preproc_data"
COHORT_DIR = "cohort"
# mimic4_path=root_dir + "/mimiciv/2.0/",


raw_path = Path(RAW_DIR)
mimiciv_path = raw_path / Path("mimiciv_2_0")
preproc_path = Path(PREPROC_DIR)
cohort_path = preproc_path / Path(COHORT_DIR)


USE_ICU_DATA = 1
USE_ADMISSION_DATA = 0
DISEASE_LABEL = ""

# USE_ICU_DATA


admit_col = "intime"
disch_col = "outtime"


group_col = "subject_id"
adm_visit_col = "hadm_id"
visit_col = "stay_id"

visits = pd.read_csv(
    mimiciv_path / "icu" / "icustays.csv.gz",
    compression="gzip",
    header=0,
    index_col=None,
    parse_dates=[admit_col, disch_col],
)
breakpoint()

patients = pd.read_csv(
    mimiciv_path / "hosp" / "patients.csv.gz",
    compression="gzip",
    header=0,
    index_col=None,
    # usecols=[
    #     group_col,
    #     "anchor_year",
    #     "anchor_age",
    #     "anchor_year_group",
    #     "dod",
    #     "gender",
    # ],
)
patients["yob"] = (
    patients["anchor_year"] - patients["anchor_age"]
)  # get yob to ensure a given visit is from an adult
patients["min_valid_year"] = patients["anchor_year"] + (
    2019 - patients["anchor_year_group"].str.slice(start=-4).astype(int)
)

# Define anchor_year corresponding to the anchor_year_group 2017-2019. This is later used to prevent consideration
# of visits with prediction windows outside the dataset's time range (2008-2019)
# [[group_col, visit_col, admit_col, disch_col]]
visit_pts = visits[
    [group_col, visit_col, adm_visit_col, admit_col, disch_col, "los"]
].merge(
    patients[
        [
            group_col,
            "anchor_year",
            "anchor_age",
            "yob",
            "min_valid_year",
            "dod",
            "gender",
        ]
    ],
    how="inner",
    left_on=group_col,
    right_on=group_col,
)


# only take adult patients
#     visit_pts['Age']=visit_pts[admit_col].dt.year - visit_pts['yob']
#     visit_pts = visit_pts.loc[visit_pts['Age'] >= 18]
visit_pts["Age"] = visit_pts["anchor_age"]
visit_pts = visit_pts.loc[visit_pts["Age"] >= 18]

##Add Demo data
eth = pd.read_csv(
    mimiciv_path / "hosp" / "admissions.csv.gz",
    compression="gzip",
    header=0,
    usecols=["hadm_id", "insurance", "race"],
    index_col=None,
)
visit_pts = visit_pts.merge(eth, how="inner", left_on="hadm_id", right_on="hadm_id")

visit_pts[
    [
        group_col,
        visit_col,
        adm_visit_col,
        admit_col,
        disch_col,
        "los",
        "min_valid_year",
        "dod",
        "Age",
        "gender",
        "race",
        "insurance",
    ]
]
breakpoint()
