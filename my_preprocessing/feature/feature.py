from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path


class Feature(ABC):
    @abstractmethod
    def summary_path(self) -> Path:
        pass

    def feature_path(self) -> Path:
        pass

    @abstractmethod
    def make(self, cohort: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def preproc(self):
        pass
