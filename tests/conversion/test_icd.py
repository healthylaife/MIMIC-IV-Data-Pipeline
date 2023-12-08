import pandas as pd
from pipeline.conversion.icd import IcdConverter
from pipeline.file_info.raw.hosp import HospDiagnosesIcd


def test_converter():
    """
    Tests the IcdConverter class for standardizing ICD codes and extracting hospital
    admission IDs based on specific ICD-10 codes.

    This test validates:
    - The conversion of ICD codes from version 9 to version 10.
    - The extraction of 'root' ICD codes.
    - The retrieval of hospital admission IDs for a given ICD-10 code.
    """

    # Given: Sample ICD codes, versions, and hospital admission IDs
    icd_codes = [
        "4139",
        "V707",
        "41401",
        "D696",
        "S030XXA",
        "S25512A",
        "I5022",
        "42821",
        "4280",
    ]
    icd_versions = [9, 9, 9, 10, 10, 10, 10, 9, 9]

    admissions = [
        1,
        1,
        1,
        2,
        3,
        3,
        3,
        4,
        4,
    ]
    df = pd.DataFrame(
        {
            "icd_code": icd_codes,
            "icd_version": icd_versions,
            "hadm_id": admissions,
        }
    )

    icd_converter = IcdConverter()
    st_dia = icd_converter.standardize_icd(df)
    hids = icd_converter.get_pos_ids(st_dia, "I50")

    #  Expected results for root ICD-10 conversion and hospital admission IDs
    expected_root_icd10 = [
        "I208",
        "Z0000",
        "I2510",
        "D696",
        "S030XXA",
        "S25512A",
        "I5022",
        "I50814",
        "I50814",
    ]
    expected_root = [
        "I20",
        "Z00",
        "I25",
        "D69",
        "S03",
        "S25",
        "I50",
        "I50",
        "I50",
    ]
    assert st_dia["root_icd10_convert"].values.tolist() == expected_root_icd10
    assert st_dia["root"].values.tolist() == expected_root
    assert hids.tolist() == [3, 4]
