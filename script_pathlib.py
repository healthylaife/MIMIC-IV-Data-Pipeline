from pathlib import Path
import pandas as pd


DATA_DIR = Path("preproc_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = "demo_subject_id.csv"
toto = pd.read_csv(DATA_DIR / DATA_FILE)
print(len(toto))
