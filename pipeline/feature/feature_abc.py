from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path


class Feature(ABC):
    """
    Abstract base class for a feature in the dataset.
    Defines the structure and required methods for a feature.
    """

    @abstractmethod
    def extract_from(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the feature data and return it as a DataFrame.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Save the feature data to a file.
        """
        pass

    @abstractmethod
    def preproc(self) -> None:
        """
        Preprocess the feature data.
        """
        pass

    @abstractmethod
    def summary(self) -> None:
        """
        Generate a summary of the feature.
        """
        pass
