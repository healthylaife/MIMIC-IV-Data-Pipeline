# import numpy as np
# import pandas as pd


# ########################## MAPPING ##########################
# def read_icd_mapping(map_path):
#     mapping = pd.read_csv(map_path, header=0, delimiter="\t")
#     mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
#     return mapping


# def preproc_icd_module_old(
#     module_path: str,
#     adm_cohort_path: str,
#     icd_map_path=None,
#     map_code_colname=None,
#     only_icd10=True,
# ) -> pd.DataFrame:
#     """Takes an module dataset with ICD codes and puts it in long_format, optionally mapping ICD-codes by a mapping table path"""

#     def get_module_cohort(module_path: str):
#         module = pd.read_csv(module_path, compression="gzip", header=0)
#         adm_cohort = pd.read_csv(adm_cohort_path, compression="gzip", header=0)
#         return module.merge(
#             adm_cohort[["hadm_id", "stay_id", "label"]],
#             how="inner",
#             left_on="hadm_id",
#             right_on="hadm_id",
#         )

#     def standardize_icd(mapping, df, root=False):
#         """Takes an ICD9 -> ICD10 mapping table and a modulenosis dataframe; adds column with converted ICD10 column"""

#         def icd_9to10(icd):
#             # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
#             if root:
#                 icd = icd[:3]
#             try:
#                 # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
#                 return mapping.loc[mapping[map_code_colname] == icd].icd10cm.iloc[0]
#             except:
#                 # print("Error on code", icd)
#                 return np.nan

#         # Create new column with original codes as default
#         col_name = "icd10_convert"
#         if root:
#             col_name = "root_" + col_name
#         df[col_name] = df["icd_code"].values

#         # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
#         for code, group in df.loc[df.icd_version == 9].groupby(by="icd_code"):
#             new_code = icd_9to10(code)
#             for idx in group.index.values:
#                 # Modify values of original df at the indexes in the groups
#                 df.at[idx, col_name] = new_code

#         if only_icd10:
#             # Column for just the roots of the converted ICD10 column
#             df["root"] = df[col_name].apply(
#                 lambda x: x[:3] if type(x) is str else np.nan
#             )

#     module = get_module_cohort(module_path)

#     # Optional ICD mapping if argument passed
#     if icd_map_path:
#         icd_map = read_icd_mapping(icd_map_path)
#         # print(icd_map)
#         standardize_icd(icd_map, module, root=True)
#         print(
#             "# unique ICD-9 codes",
#             module[module["icd_version"] == 9]["icd_code"].nunique(),
#         )
#         print(
#             "# unique ICD-10 codes",
#             module[module["icd_version"] == 10]["icd_code"].nunique(),
#         )
#         print(
#             "# unique ICD-10 codes (After converting ICD-9 to ICD-10)",
#             module["root_icd10_convert"].nunique(),
#         )
#         print(
#             "# unique ICD-10 codes (After clinical gruping ICD-10 codes)",
#             module["root"].nunique(),
#         )
#         print("# Admissions:  ", module.stay_id.nunique())
#         print("Total rows", module.shape[0])
#     return module


# def preproc_icd_module(
#     module_path: str,
#     adm_cohort_path: str,
#     icd_map_path=None,
#     map_code_colname=None,
#     only_icd10=True,
# ) -> pd.DataFrame:
#     """Takes an module dataset with ICD codes and puts it in long_format, optionally mapping ICD-codes by a mapping table path"""

#     def get_module_cohort(module_path: str):
#         module = pd.read_csv(module_path, compression="gzip", header=0)
#         adm_cohort = pd.read_csv(adm_cohort_path, compression="gzip", header=0)
#         return module.merge(
#             adm_cohort[["hadm_id", "stay_id", "label"]],
#             how="inner",
#             left_on="hadm_id",
#             right_on="hadm_id",
#         )

#     def standardize_icd_old(mapping, df, root=False):
#         """Takes an ICD9 -> ICD10 mapping table and a modulenosis dataframe; adds column with converted ICD10 column"""

#         def icd_9to10(icd):
#             # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
#             if root:
#                 icd = icd[:3]
#             try:
#                 # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
#                 return mapping.loc[mapping[map_code_colname] == icd].icd10cm.iloc[0]
#             except:
#                 # print("Error on code", icd)
#                 return np.nan

#         # Create new column with original codes as default
#         col_name = "icd10_convert"
#         if root:
#             col_name = "root_" + col_name
#         df[col_name] = df["icd_code"].values

#         # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
#         for code, group in df.loc[df.icd_version == 9].groupby(by="icd_code"):
#             new_code = icd_9to10(code)
#             for idx in group.index.values:
#                 # Modify values of original df at the indexes in the groups
#                 df.at[idx, col_name] = new_code

#         if only_icd10:
#             # Column for just the roots of the converted ICD10 column
#             df["root"] = df[col_name].apply(
#                 lambda x: x[:3] if type(x) is str else np.nan
#             )

#     def standardize_icd(mapping, df, root=False, only_icd10=False):
#         """Converts ICD9 codes to ICD10 codes using a provided mapping table.
#         Adds a column to the DataFrame with the converted ICD10 codes."""

#         map_code_colname = (
#             "icd9"  # or the appropriate column name in your mapping table
#         )

#         def icd_9to10(icd):
#             """Converts an ICD9 code to ICD10. Uses only the root of the ICD9 code if specified."""
#             if root:
#                 icd = icd[:3]
#             return (
#                 mapping.loc[mapping[map_code_colname] == icd, "icd10cm"].iloc[0]
#                 if icd in mapping[map_code_colname].values
#                 else np.nan
#             )

#         # Apply conversion function to each ICD9 code
#         col_name = "root_icd10_convert" if root else "icd10_convert"
#         df[col_name] = df.loc[df.icd_version == 9, "icd_code"].apply(icd_9to10)

#         if only_icd10:
#             # Extract the root of the converted ICD10 codes
#             df["root"] = df[col_name].apply(
#                 lambda x: x[:3] if isinstance(x, str) else np.nan
#             )

#         return df

#     module = get_module_cohort(module_path)

#     # Optional ICD mapping if argument passed
#     if icd_map_path:
#         icd_map = read_icd_mapping(icd_map_path)
#         # print(icd_map)
#         standardize_icd(icd_map, module, root=True)
#         print(
#             "# unique ICD-9 codes",
#             module[module["icd_version"] == 9]["icd_code"].nunique(),
#         )
#         print(
#             "# unique ICD-10 codes",
#             module[module["icd_version"] == 10]["icd_code"].nunique(),
#         )
#         print(
#             "# unique ICD-10 codes (After converting ICD-9 to ICD-10)",
#             module["root_icd10_convert"].nunique(),
#         )
#         print(
#             "# unique ICD-10 codes (After clinical gruping ICD-10 codes)",
#             module["root"].nunique(),
#         )
#         print("# Admissions:  ", module.stay_id.nunique())
#         print("Total rows", module.shape[0])
#     return module
