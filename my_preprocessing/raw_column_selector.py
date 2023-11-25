class RawSelector:
    def __init__(self, use_icu: bool, label: str, icd_code: str, disease_label: str):
        self.use_icu = use_icu
        self.label = label
        self.icd_code = icd_code
        self.disease_label = disease_label
        self.visit_col = "stay_id" if use_icu else "hadm_id"
        self.admit_col = "intime" if use_icu else "admittime"
        self.dish_col = "outtime" if use_icu else "hadm_id"
        self.admit_col = "intime" if use_icu else "dischtime"
        self.adm_visit_col = "hadm_id" if use_icu else ""
