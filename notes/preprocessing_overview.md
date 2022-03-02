# Preprocessing Overview
*Last updated: 3/1/22*

## Time Series Readmission

### `preproc_cohort.ipynb`

- Filter out individuals less than 18 y/o and extract anchor_year_groups from a designated anchor year

- Anchor year situation:
	- Anchor year groups tell you what range of years a particular fake year corresponds to
		- To standardize the data, I specified a base year that corresponds to a patient's 2008-2010 time range
		    - This acts as "timedelta 0" for a patient
		- I then let the observation window go up to the 2014-2016 time range
			- A patient is labelled as having a readmisson when they have a visit in the observation range AND the prediction range (after 2014-2016)

- Because of how the time ranges are in 3-year intervals, you can potentially get time ranges like 2007-2009 and 2018-2020
	- I removed records such as these so that an observation window and prediction window would be consistent
        - Performed using `get_range()` of `hosp_preprocess_util.py`
		- Observation window (7 yr) = 2008-2010 to 2014-2016
        - Prediction window (3 yr) = 2015-2017 to 2017-2019

### `preproc_diag.ipynb`

- Preprocesses the `diagnoses_icd` table

- Uses the ICD9 -> ICD10 mapping table (`utils/mappings/ICD9_to_ICD10_mapping.txt`) to convert ICD9 codes to ICD10
	- To allow for more conversions, we only look at the root category (first 3 characters of the ICD9 code) when converting
	- Conversions occur in `preproc_icd_module()` and `standardizard_icd()` in the `hosp_preprocess_util.py` module
- After converting, we reduce the number of ICD10 codes present by only taking their root categories (again, first 3 characters of the code)

### `preproc_drugs.ipynb`

- Preprocesses the `prescriptions` table

- Uses the `NDC_product_table.csv` to map `ndc` codes to nonproprietary names.
	- The NDC codes in the mapping table and `prescriptions` table needed to be preprocessed to be in the same format.
	- Use of nonproprietary names reduces the redundancy of using brand names that the `prescriptions` table currently has 

### `preproc_proc.ipynb`
- Preprocesses the `procedures` table

- Unable to find mapping table for the procedure ICD9 -> 10 codes, so ICD9 codes have been filtered out

### `preproc_labs.ipynb`
- Preprocesses the `labevents` table

- Uses `valuenum` column to get numerical test values

- Manually looked through tests that had multiple unit of measures (`valueuom`)
	- If a test had unit of measures that were different from eachother, I kept the most frequent measurement
		- Could benefit from automating this process and/or finding a mapping table for different units of measures

### `normalize_preprocs.ipynb`
- Converts the preprocessed datasets (`preproc_{diag/labs/meds/proc}`) into "long format" datasets
- Modifies the timedelta so that every patient has a timedelta of 0 (the earliest date that patient has across long format datasets)
- Output files are suffixed with `norm`

## Day Interval Readmission

### `days_interval_cohort.py`

- Take every visit as a sample and check for readmission in the next 30/60/90 days following that visit's discharge date
    - Visits whose prediction range potentially extends beyond the 2019 limit is excluded