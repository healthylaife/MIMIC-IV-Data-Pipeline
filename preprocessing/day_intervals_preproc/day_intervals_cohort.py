import datetime
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './..')

TESTING = True # Variable for testing functions
OUTPUT_DIR = './data/day_intervals/cohort'

def get_visit_pts(mimic4_path:str, group_col:str, visit_col:str, admit_col:str, disch_col:str, use_mort:bool, use_ICU=False):
    """Combines the MIMIC-IV core/patients table information with either the icu/icustays or core/admissions data.

    Parameters:
    mimic4_path: path to mimic-iv folder containing MIMIC-IV data
    group_col: patient identifier to group patients (normally subject_id)
    visit_col: visit identifier for individual patient visits (normally hadm_id or stay_id)
    admit_col: column for visit start date information (normally admittime or intime)
    disch_col: column for visit end date information (normally dischtime or outtime)
    use_ICU: describes whether to speficially look at ICU visits in icu/icustays OR look at general admissions from core/admissions
    """

    visit = None
    if use_ICU:
        visit = pd.read_csv(mimic4_path + "icu/icustays.csv.gz", compression='gzip', header=0, index_col=None, parse_dates=[admit_col, disch_col])
    else:
        visit = pd.read_csv(mimic4_path + "core/admissions.csv.gz", compression='gzip', header=0, index_col=None, parse_dates=[admit_col, disch_col])
    
    if not use_mort:
        visit = visit.loc[visit.hospital_expire_flag == 0]    # remove hospitalizations with a death; impossible for readmission for such visits

    pts = pd.read_csv(
            mimic4_path + "core/patients.csv.gz", compression='gzip', header=0, index_col = None, usecols=[group_col, 'anchor_year', 'anchor_age', 'anchor_year_group', 'dod']
        )
    pts['yob']= pts['anchor_year'] - pts['anchor_age']  # get yob to ensure a given visit is from an adult
    pts['min_valid_year'] = pts['anchor_year'] + (2019 - pts['anchor_year_group'].str.slice(start=-4).astype(int))

    # Define anchor_year corresponding to the anchor_year_group 2017-2019. This is later used to prevent consideration
    # of visits with prediction windows outside the dataset's time range (2008-2019)
    visit_pts = visit[[group_col, visit_col, admit_col, disch_col]].merge(
            pts[[group_col, 'anchor_year', 'anchor_age', 'yob', 'min_valid_year', 'dod']], how='inner', left_on=group_col, right_on=group_col
        )
    visit_pts = visit_pts.loc[adm_pts[admit_col].dt.year - visit_pts['yob'] >= 18]

    return visit_pts.dropna(subset=['min_valid_year'])[[group_col, visit_col, admit_col, disch_col, 'min_valid_year', 'dod']]


def validate_row(row, ctrl, invalid, max_year, disch_col, valid_col, gap):
    """Checks if visit's prediction window potentially extends beyond the dataset range (2008-2019).
    An 'invalid row' is NOT guaranteed to be outside the range, only potentially outside due to
    de-identification of MIMIC-IV being done through 3-year time ranges.
    
    To be invalid, the end of the prediction window's year must both extend beyond the maximum seen year
    for a patient AND beyond the year that corresponds to the 2017-2019 anchor year range for a patient"""

    pred_year = (row[disch_col] + gap).year
    if max_year < pred_year and pred_year > row[valid_col]:
        invalid = invalid.append(row)
    else:
        ctrl = ctrl.append(row)
    return ctrl, invalid


def partition_by_readmit(df:pd.DataFrame, gap:datetime.timedelta, group_col:str, visit_col:str, admit_col:str, disch_col:str, valid_col:str):
    """Applies labels to individual visits according to whether or not a readmission has occurred within the specified `gap` days.
    For a given visit, another visit must occur within the gap window for a positive readmission label.
    The gap window starts from the disch_col time and the admit_col of subsequent visits are considered."""
    
    case = pd.DataFrame()   # hadm_ids with readmission within the gap period
    ctrl = pd.DataFrame()   # hadm_ids without readmission within the gap period
    invalid = pd.DataFrame()    # hadm_ids that are not considered in the cohort

    # Iterate through groupbys based on group_col (subject_id). Data is sorted by subject_id and admit_col (admittime)
    # to ensure that the most current hadm_id is last in a group.
    for subject, group in df[[group_col, visit_col, admit_col, disch_col, valid_col]].sort_values(by=[group_col, admit_col]).groupby(group_col):
        max_year = group.max()[disch_col].year

        if group.shape[0] <= 1:
            ctrl, invalid = validate_row(group.iloc[0], ctrl, invalid, max_year, disch_col, valid_col, gap)   # A group with 1 row has no readmission; goes to ctrl
        else:
            for idx in range(group.shape[0]-1):
                visit_time = group.iloc[idx][disch_col]  # For each index (a unique hadm_id), get its timestamp
                if group.loc[
                    (group[admit_col] > visit_time) &    # Readmissions must come AFTER the current timestamp
                    (group[admit_col] - visit_time <= gap)   # Distance between a timestamp and readmission must be within gap
                    ].shape[0] >= 1:                # If ANY rows meet above requirements, a readmission has occurred after that visit

                    case = case.append(group.iloc[idx])
                else:
                    # If no readmission is found, only add to ctrl if prediction window is guaranteed to be within the
                    # time range of the dataset (2008-2019). Visits with prediction windows existing in potentially out-of-range
                    # dates (like 2018-2020) are excluded UNLESS the prediction window takes place the same year as the visit,
                    # in which case it is guaranteed to be within 2008-2019

                    ctrl = ctrl.append(group.iloc[idx])

            ctrl, invalid = validate_row(group.iloc[-1], ctrl, invalid, max_year, disch_col, valid_col, gap)  # The last hadm_id datewise is guaranteed to have no readmission logically
            print(f"[ {gap.days} DAYS ] {case.shape[0] + ctrl.shape[0]}/{df.shape[0]} hadm_ids processed")

    return case, ctrl, invalid


def partition_by_mort(df:pd.DataFrame, group_col:str, visit_col:str, admit_col:str, disch_col:str, death_col:str, valid_col:str):
    """Applies labels to individual visits according to whether or not a death has occurred within
    the times of the specified admit_col and disch_col"""

    invalid = df.loc[(df[admit_col].isna()) | (df[disch_col].isna())]

    cohort = df.loc[(~df[admit_col].isna()) & (~df[disch_col].isna())]
    cohort['label'] = (~cohort[death_col].isna()) & (cohort[death_col] >= cohort[admit_col]) & (cohort[death_col] <= cohort[disch_col])
    cohort['label'] = cohort['label'].astype("Int32")

    return cohort, invalid


def get_case_ctrls(df:pd.DataFrame, gap:int, group_col:str, visit_col:str, admit_col:str, disch_col:str, valid_col:str, death_col:str, use_mort=False) -> pd.DataFrame:
    """Handles logic for creating the labelled cohort based on arguments passed to extract().

    Parameters:
    df: dataframe with patient data
    gap: specified time interval gap for readmissions
    group_col: patient identifier to group patients (normally subject_id)
    visit_col: visit identifier for individual patient visits (normally hadm_id or stay_id)
    admit_col: column for visit start date information (normally admittime or intime)
    disch_col: column for visit end date information (normally dischtime or outtime)
    valid_col: generated column containing a patient's year that corresponds to the 2017-2019 anchor time range
    dod_col: Date of death column
    """

    case = None  # hadm_ids with readmission within the gap period
    ctrl = None   # hadm_ids without readmission within the gap period
    invalid = None    # hadm_ids that are not considered in the cohort
    gap = datetime.timedelta(days=gap)  # transform gap into a timedelta to compare with datetime columns

    if use_mort:
        return partition_by_mort(df, gap, group_col, visit_col, admit_col, disch_col, death_col, valid_col)
    else:
        case, ctrl, invalid = partition_by_readmit(df, gap, group_col, visit_col, admit_col, disch_col, valid_col)

        case['label'] = np.ones(case.shape[0]).astype(int)
        ctrl['label'] = np.zeros(ctrl.shape[0]).astype(int)

        return pd.concat([case, ctrl], axis=0), invalid

    print(f"[ {gap.days} DAYS ] {invalid.shape[0]} hadm_ids are invalid")
    # case hadm_ids are labelled 1 for readmission, ctrls have a 0 label
    case['label'] = np.ones(case.shape[0]).astype(int)
    ctrl['label'] = np.zeros(ctrl.shape[0]).astype(int)

    return pd.concat([case, ctrl], axis=0), invalid


def extract(cohort_output:str, summary_output:str, use_ICU:str, label:str):
    """Extracts cohort data and summary from MIMIC-IV data based on provided parameters.

    Parameters:
    cohort_output: name of labelled cohort output file
    summary_output: name of summary output file
    use_ICU: state whether to use ICU patient data or not
    label: Can either be '{day} day Readmission' or 'Mortality', decides what binary data label signifies"""

    cohort, invalid, pts = None, None, None
    group_col, visit_col, admit_col, disch_col, death_col = "", "", "", "", ""
    use_mort = label == "Mortality"

    if use_ICU == 'ICU':
        group_col='subject_id'
        visit_col='stay_id'
        admit_col='intime'
        disch_col='outtime'
        death_col='dod'

        pts = get_visit_pts(
            mimic4_path="./mimic-iv-1.0/",
            group_col=group_col,
            visit_col=visit_col,
            admit_col=admit_col,
            disch_col=disch_col
        )
    else:
        group_col='subject_id'
        visit_col='hadm_id'
        admit_col='admittime'
        disch_col='dischtime'
        death_col='dod'

        pts = get_visit_pts(
            mimic4_path="./mimic-iv-1.0/",
            group_col=group_col,
            visit_col=visit_col,
            admit_col=admit_col,
            disch_col=disch_col
        )

    if use_mort:
        cohort, invalid = get_case_ctrls(pts, None, group_col, visit_col, admit_col, disch_col,'min_valid_year', death_col, use_mort=True)
    else:
        interval = int(label[:3].strip())
        cohort, invalid = get_case_ctrls(pts, interval, group_col, visit_col, admit_col, disch_col,'min_valid_year', death_col)

    # test_case_ctrls(test_case=False, df=pts, df_new=cohort, invalid=invalid)
    cohort.to_csv(f"{cohort_output}.csv.gz", index=False, compression='gzip')

    print(f"{label} FOR {use_ICU} DATA")
    print("Rowsize of dataset: ", cohort.shape[0])
    print("Number of invalid visits: ", invalid.shape[0])
    print(cohort.label.value_counts())
    print("====================")


def test_case_ctrls(test_case=True, df=None, df_new=None, invalid=None):
    """Function for testing get_case_ctrls() with a small dataset and for testing attributes on the entire dataset"""
    if test_case:
        df = pd.DataFrame({
            'subject_id':[1] + [2]*2 + [3]*7 + [4]*3 + [5]*2,
            'hadm_id': range(15),
            'admittime': [
                datetime.datetime(year=2000, month=1 ,day=1) + datetime.timedelta(days=i) for i in [
                    1,      # Test that group size of 1 automatically sends to ctrl
                    1, 5,   # Ensures that the last visit is always ctrl (given that its within the min valid year)
                    1, 3, 5, 60, 20, 100, 1000, # Tests reordering and that gaps are appropriately sent to ctrl
                    1, 350, 357,    # Ensures that readmissions extending beyond the minimum valid year are removed
                    1, 43     # Test to make sure readmissions are based on the end of disch_time, NOT admit_time
                ]
            ],
            'dischtime': [
                datetime.datetime(year=2000, month=1 ,day=1) + datetime.timedelta(days=i) for i in [
                    2,      # Test that group size of 1 automatically sends to ctrl
                    2, 7,   # Ensures that the last visit is always ctrl (given that its within the min valid year)
                    2, 4, 6, 62, 22, 103, 1003, # Tests reordering
                    2, 353, 360,    # Ensures that readmissions extending beyond the minimum valid year are removed
                    40, 46     # Test to make sure readmissions are based on the end of disch_time, NOT admit_time
                ]
            ],
            'min_valid_year':[2020]*10 + [2000]*5
        })
        df_new, invalid = get_case_ctrls(df, 30, 'subject_id', 'hadm_id', 'admittime', 'dischtime', 'min_valid_year') #, 'yob'

        assert df_new.loc[df_new.label == 1].shape[0] == 6, f"New df has {df_new.loc[df_new.label == 1].shape[0]} readmits"
        assert df_new.loc[df_new.label == 0].shape[0] == 8, f"New df has {df_new.loc[df_new.label == 0].shape[0]} ctrls"

    assert df.shape[0] == df_new.shape[0] + invalid.shape[0]    # Ensure sizes are did not change
    assert set(df.subject_id.unique()) == set(np.concatenate((df_new.subject_id.unique(), invalid.subject_id.unique())))    # Check that subject_ids are identical
    assert df.hadm_id.nunique() == df_new.hadm_id.nunique() + invalid.hadm_id.nunique() # Check that all hadm_ids remained

    print("All tests passed!")


if __name__ == '__main__':
    if TESTING:
        test_case_ctrls()
    else:
        output_info = []
        adm_pts = get_adm_pts("./mimic-iv-1.0/")
        for i in [30, 60, 90]:
            cohort, invalid = get_case_ctrls(
                df=adm_pts,
                gap=i,
                group_col='subject_id',
                visit_col='hadm_id',
                admit_col='admittime',
                disch_col='dischtime',
                valid_col='min_valid_year'
                )
            test_case_ctrls(test_case=False, df=adm_pts, df_new=cohort, invalid=invalid)
            cohort.to_csv(f"{OUTPUT_DIR}/cohort{i}day.csv.gz", index=False, compression='gzip')

            output_info.append((i, cohort.shape[0], invalid.shape[0], cohort.label.value_counts()))

        for info in output_info:
            print(f"{info[0]} DAY GAP")
            print("Rowsize of dataset: ", info[1])
            print("Number of invalid hadm_ids: ", info[2])
            print(info[3])
            print("====================")