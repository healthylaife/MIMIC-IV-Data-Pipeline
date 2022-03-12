import unittest
from day_intervals_cohort import *

class TestDayIntervals(unittest.TestCase):

    def test_case_ctrls_readmission(self):
        """Function for testing get_case_ctrls() with a small dataset and for testing attributes on the entire dataset"""
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
        df_new, invalid = get_case_ctrls(
                df=df,
                gap=30,
                group_col='subject_id',
                visit_col='hadm_id',
                admit_col='admittime',
                disch_col='dischtime',
                valid_col='min_valid_year',
                death_col='dod',
                use_mort=False
            )

        self.assertEqual(df_new.loc[df_new.label == 1].shape[0], 6, f"New df has {df_new.loc[df_new.label == 1].shape[0]} readmits")
        self.assertEqual(df_new.loc[df_new.label == 0].shape[0], 8, f"New df has {df_new.loc[df_new.label == 0].shape[0]} ctrls")

        self.assertEqual(df.shape[0], df_new.shape[0] + invalid.shape[0])   # Ensure sizes are did not change
        self.assertEqual( set(df.subject_id.unique()), set(np.concatenate((df_new.subject_id.unique(), invalid.subject_id.unique())))  )  # Check that subject_ids are identical
        self.assertEqual(df.hadm_id.nunique(), df_new.hadm_id.nunique() + invalid.hadm_id.nunique()) # Check that all hadm_ids remained

        print("Readmission Tests Passed!")


    def test_case_ctrls_mortality(self):
        """Function for testing get_case_ctrls() mortality with a small dataset"""
        df = pd.DataFrame({
            'subject_id':[1]*2 + [2]*2 + [3],
            'hadm_id': range(5),
            'admittime': [
                    datetime.datetime(year=2000, month=1 ,day=1), # Should give a negative label; death outside of visit range
                    datetime.datetime(year=2000, month=1 ,day=10), # Should give a positive label; death inside of visit range
                    np.NaN,                                       # Tests that rows with a NaN admittime or dischtime are invalid
                    datetime.datetime(year=2000, month=1 ,day=5), # Tests that rows with a NaN admittime or dischtime are invalid
                    datetime.datetime(year=2000, month=1 ,day=30) # Tests that visits with a NaN deathtime are put in ctrl
            ],
            'dischtime': [
                    datetime.datetime(year=2000, month=1 ,day=5), # Should give a negative label; death outside of visit range
                    datetime.datetime(year=2000, month=1 ,day=20), # Should give a positive label; death inside of visit range
                    datetime.datetime(year=2000, month=1 ,day=5), # Tests that rows with a NaN admittime or dischtime are invalid
                    np.NaN,                                       # Tests that rows with a NaN admittime or dischtime are invalid
                    datetime.datetime(year=2000, month=2 ,day=10) # Tests that visits with a NaN deathtime are put in ctrl
            ],
            'dod':[
                    datetime.datetime(year=2000, month=1 ,day=15), # Should give a negative label; death outside of visit range
                    datetime.datetime(year=2000, month=1 ,day=15), # Should give a positive label; death inside of visit range
                    datetime.datetime(year=2000, month=1 ,day=3), # Tests that rows with a NaN admittime or dischtime are invalid
                    datetime.datetime(year=2000, month=1 ,day=3), # Tests that rows with a NaN admittime or dischtime are invalid
                    np.NaN                                         # Tests that visits with a NaN deathtime are put in ctrl
            ]
        })

        df_new, invalid = get_case_ctrls(
                df=df,
                gap=30,
                group_col='subject_id',
                visit_col='hadm_id',
                admit_col='admittime',
                disch_col='dischtime',
                valid_col='min_valid_year',
                death_col='dod',
                use_mort=True
            )

        self.assertEqual(df_new.loc[df_new.label == 1].shape[0], 1, f"New df has {df_new.loc[df_new.label == 1].shape[0]} cases")
        self.assertEqual(df_new.loc[df_new.label == 0].shape[0], 2, f"New df has {df_new.loc[df_new.label == 0].shape[0]} ctrls")

        self.assertEqual(df.shape[0], df_new.shape[0] + invalid.shape[0])   # Ensure sizes are did not change
        self.assertEqual( set(df.subject_id.unique()), set(np.concatenate((df_new.subject_id.unique(), invalid.subject_id.unique())))  )  # Check that subject_ids are identical
        self.assertEqual(df.hadm_id.nunique(), df_new.hadm_id.nunique() + invalid.hadm_id.nunique()) # Check that all hadm_ids remained

        print("Mortality Tests Passed!")

if __name__ == '__main__':
    unittest.main()