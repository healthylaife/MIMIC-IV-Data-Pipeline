import pandas as pd
import logging
from pipeline.file_info.preproc.cohort import COHORT_PATH, CohortHeader

logger = logging.getLogger()


def generate_admission_cohort(cohort_output: str) -> pd.DataFrame:
    data = pd.read_csv(
        COHORT_PATH / f"{cohort_output}.csv.gz",
        compression="gzip",
    )
    for col in [CohortHeader.ADMIT_TIME, CohortHeader.DISCH_TIME]:
        data[col] = pd.to_datetime(data[col])

    data[CohortHeader.LOS] = (
        (
            data[CohortHeader.DISCH_TIME] - data[CohortHeader.ADMIT_TIME]
        ).dt.total_seconds()
        / 3600
    ).astype(int)
    data = data[data[CohortHeader.LOS] > 0]
    data[CohortHeader.AGE] = data[CohortHeader.AGE].astype(int)

    logger.info("[ READ COHORT ]")
    return data
