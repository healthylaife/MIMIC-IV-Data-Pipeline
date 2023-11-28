import pandas as pd
import numpy as np


# The NDC codes in the prescription dataset is the 11-digit NDC code, although codes are missing
# their leading 0's because the column was interpreted as a float then integer; this function restores
# the leading 0's, then obtains only the PRODUCT and MANUFACTUERER parts of the NDC code (first 9 digits)
def ndc_to_str(ndc):
    if ndc < 0:  # dummy values are < 0
        return np.nan
    ndc = str(ndc)
    return (("0" * (11 - len(ndc))) + ndc)[0:-2]


# The mapping table is ALSO incorrectly formatted for 11 digit NDC codes. An 11 digit NDC is in the
# form of xxxxx-xxxx-xx for manufactuerer-product-dosage. The hyphens are in the correct spots, but
# the number of digits within each section may not be 5-4-2, in which case we add leading 0's to each
# to restore the 11 digit format. However, we only take the 5-4 sections, just like the to_str function
def format_ndc_table(ndc):
    parts = ndc.split("-")
    return ("0" * (5 - len(parts[0])) + parts[0]) + (
        "0" * (4 - len(parts[1])) + parts[1]
    )


def read_ndc_mapping2(map_path):
    ndc_map = pd.read_csv(map_path, header=0, delimiter="\t", encoding="latin1")
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
    ndc_map.columns = list(map(str.lower, ndc_map.columns))
    return ndc_map


# In NDC mapping table, the pharm_class col is structured as a text string, separating different pharm classes from eachother
# This can be [PE], [EPC], and others, but we're interested in EPC. Luckily, between each commas, it states if a phrase is [EPC]
# So, we just string split by commas and keep phrases containing "[EPC]"
def get_EPC(s):
    """Gets the Established Pharmacologic Class (EPC) from the mapping table"""
    if type(s) != str:
        return np.nan
    words = s.split(",")
    return [x for x in words if "[EPC]" in x]
