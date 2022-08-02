

### Files in the folder

- **./day_intervals_preproc**
  - **day_intervals_cohort.py** file is used to extract samples, labels and demographic data for cohorts.
  - **disease_cohort.py** is used to filter samples based on diagnoses codes at time of admission
  
- **./hosp_module_preproc**
  - **feature_selection_hosp.py** is used to extract, clean and summarize selected features for non-ICU data.
  - **feature_selection_icu.py** is used to extract, clean and summarize selected features for ICU data.
  Both above files internally use files in /./utils folder for feature extraction, cleaning, outlier removal, unit conversion.



