import datetime
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './..')

TESTING = False # Variable for testing functions
OUTPUT_DIR = './data/day_intervals/cohort'

def get_adm_pts(mimic4_path:str, use_ICD:False):
    adm = pd.read_csv(mimic4_path + "core/admissions.csv.gz", compression='gzip', header=0, index_col=None, parse_dates=['admittime'])
    adm = adm.loc[adm.hospital_expire_flag == 0]    # remove hospitalizations with a death; impossible for readmission for such visits

    pts = pd.read_csv(
            mimic4_path + "core/patients.csv.gz", compression='gzip', header=0, index_col = None, usecols=['subject_id', 'anchor_year', 'anchor_age', 'anchor_year_group']
        )
    pts['yob']= pts['anchor_year'] - pts['anchor_age']  # get yob to ensure a given visit is from an adult
    pts['min_valid_year'] = pts['anchor_year'] + (2019 - pts['anchor_year_group'].str.slice(start=-4).astype(int))

    # Define anchor_year corresponding to the anchor_year_group 2017-2019. This is later used to prevent consideration
    # of visits with prediction windows outside the dataset's time range (2008-2019)

    adm_pts = adm[['subject_id', 'hadm_id', 'admittime']].merge(
            pts[['subject_id', 'anchor_year', 'anchor_age', 'yob', 'min_valid_year']], how='inner', left_on='subject_id', right_on='subject_id'
        )
    adm_pts = adm_pts.loc[adm_pts['admittime'].dt.year - adm_pts['yob'] >= 18]
    # adm_pts['max_valid_year'] = get_range(adm_pts, 'admittime', 'anchor_year', measure='years', keep_lower=True)

    return adm_pts.dropna(subset=['min_valid_year'])[['subject_id', 'hadm_id', 'admittime', 'min_valid_year']]


def is_row_valid(row, gap, time_col, valid_col, max_year):
    pred_year = (row[time_col] + gap).year
    if pred_year > row[valid_col] and max_year < pred_year:
        return False
    return True


def get_case_ctrls(df:pd.DataFrame, gap:int, group_col:str, visit_col:str, time_col:str, valid_col:str) -> pd.DataFrame:
    case = pd.DataFrame()   # hadm_ids with readmission within the gap period
    ctrl = pd.DataFrame()   # hadm_ids without readmission within the gap period
    invalid = pd.DataFrame()    # hadm_ids that are not considered in the cohort
    gap = datetime.timedelta(days=gap)  # transform gap into a timedelta to compare with datetime columns

    def validate_row(row, ctrl, invalid, max_year):
        if is_row_valid(row, gap, time_col, valid_col, max_year):
            ctrl = ctrl.append(row)
        else:
            invalid = invalid.append(row)
        return ctrl, invalid

    # Iterate through groupbys based on group_col (subject_id). Data is sorted by subject_id and time_col (admittime)
    # to ensure that the most current hadm_id is last in a group.
    for subject, group in df[[group_col, visit_col, time_col, valid_col]].sort_values(by=[group_col, time_col]).groupby(group_col):
        max_year = group.max()[time_col].year

        if group.shape[0] <= 1:
            ctrl, invalid = validate_row(group.iloc[0], ctrl, invalid, max_year)   # A group with 1 row has no readmission; goes to ctrl
        else:
            for idx in range(group.shape[0]-1):
                visit_time = group.iloc[idx][time_col]  # For each index (a unique hadm_id), get its timestamp
                if group.loc[
                    (group[time_col] > visit_time) &    # Readmissions must come AFTER the current timestamp
                    (group[time_col] - visit_time <= gap)   # Distance between a timestamp and readmission must be within gap
                    ].shape[0] >= 1:                # If ANY rows meet above requirements, a readmission has occurred after that visit

                    case = case.append(group.iloc[idx])
                else:
                    # If no readmission is found, only add to ctrl if prediction window is guaranteed to be within the
                    # time range of the dataset (2008-2019). Visits with prediction windows existing in potentially out-of-range
                    # dates (like 2018-2020) are excluded UNLESS the prediction window takes place the same year as the visit,
                    # in which case it is guaranteed to be within 2008-2019

                    ctrl = ctrl.append(group.iloc[idx])

            ctrl, invalid = validate_row(group.iloc[-1], ctrl, invalid, max_year)  # The last hadm_id datewise is guaranteed to have no readmission logically
            print(f"{case.shape[0] + ctrl.shape[0]}/{df.shape[0]} hadm_ids processed")

    print(f"{invalid.shape[0]} hadm_ids are invalid")
    # case hadm_ids are labelled 1 for readmission, ctrls have a 0 label
    case['label'] = np.ones(case.shape[0]).astype(int)
    ctrl['label'] = np.zeros(ctrl.shape[0]).astype(int)

    return pd.concat([case, ctrl], axis=0), invalid


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
                    1, 3, 5, 60, 20, 100, 1000, # Tests reordering
                    1, 350, 370,    # Ensures that readmissions extending beyond the minimum valid year
                    1, 360     #
                ]
            ],
            'min_valid_year':[2020]*10 + [2000]*5
            # 'yob': [1980]*10
        })
        df_new, invalid = get_case_ctrls(df, 30, 'subject_id', 'hadm_id', 'admittime', 'min_valid_year') #, 'yob'

        assert df_new.loc[df_new.label == 1].shape[0] == 5, f"New df has {df_new.loc[df_new.label == 1].shape[0]} readmits"
        assert df_new.loc[df_new.label == 0].shape[0] == 9, f"New df has {df_new.loc[df_new.label == 0].shape[0]} ctrls"

    assert df.shape[0] == df_new.shape[0] + invalid.shape[0]    # Ensure sizes are did not change
    assert set(df.subject_id.unique()) == set(np.concatenate((df_new.subject_id.unique(), invalid.subject_id.unique())))    # Check that subject_ids are identical
    assert df.hadm_id.nunique() == df_new.hadm_id.nunique() + invalid.hadm_id.nunique() # Check that all hadm_ids remained

    print("All tests passed!")

def extract(cohort_output:str, summary_output:str, use_ICU:bool, interval:int, label:str):
    """
    Extracts cohort data and summary from MIMIC-IV data based on provided parameters.

    Parameters:
    cohort_output: name of labelled cohort output file
    summary_output: name of summary output file
    use_ICU: state whether to use ICU patient data or not
    interval: define what interval of days to check for readmission. Ignore if use_mortality=True
    label: Can either be 'readmission' or 'mortality', decides what binary data label signifies
    """
    cohort, invalid = None, None
    pts = get_adm_pts("./mimic-iv-1.0/")

    if use_ICU:
        pass
    else:
        cohort, invalid = get_case_ctrls(pts, interval, 'subject_id', 'hadm_id', 'admittime', 'min_valid_year')
    # test_case_ctrls(test_case=False, df=pts, df_new=cohort, invalid=invalid)
    cohort.to_csv(f"{cohort_output}.csv.gz", index=False, compression='gzip')

if __name__ == '__main__':
    if TESTING:
        test_case_ctrls()
    else:
        adm_pts = get_adm_pts("./mimic-iv-1.0/")
        for i in [30, 60, 90]:
            cohort, invalid = get_case_ctrls(adm_pts, i, 'subject_id', 'hadm_id', 'admittime', 'min_valid_year')
            test_case_ctrls(test_case=False, df=adm_pts, df_new=cohort, invalid=invalid)
            cohort.to_csv(f"{OUTPUT_DIR}/cohort{i}day.csv.gz", index=False, compression='gzip')