# import ipywidgets as widgets
from pathlib import Path

# import pandas as pd

# import preprocessing.day_intervals_preproc.day_intervals_cohort_v2 as day_intervals_cohort_v2
import my_preprocessing.day_intervals_cohort as day_intervals_cohort

# OPTION (for target...)
# USE_ICU 'ICU',
# LABEL 'Mortality'
# time: int: 0
# icd_code: str: 'No Disease Filter'
RAW_DIR = "raw_data"
PREPROC_DIR = "preproc_data"
COHORT_DIR = "cohort"
# disease_label ''


# mimic4_path=root_dir + "/mimiciv/2.0/",


raw_path = Path(RAW_DIR)
preproc_path = Path(PREPROC_DIR)
cohort_path = preproc_path / Path(COHORT_DIR)

raw_path.mkdir(parents=True, exist_ok=True)
cohort_path.mkdir(parents=True, exist_ok=True)

cohort_output = day_intervals_cohort.extract_data(
    use_ICU="ICU",
    label="Mortality",
    time=0,
    icd_code="No Disease Filter",
    root_dir=RAW_DIR,
    preproc_dir=PREPROC_DIR,
    disease_label="",
)
