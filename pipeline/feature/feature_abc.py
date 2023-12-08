from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path


class Feature(ABC):
    """
    Abstract base class for a feature in the dataset.
    Defines the structure and required methods for a feature.
    """

    @abstractmethod
    def summary_path(self) -> Path:
        """
        Path where the summary of the feature is stored.
        """
        pass

    @abstractmethod
    def feature_path(self) -> Path:
        """
        Path where the feature data is stored.
        """
        pass

    @abstractmethod
    def make(self) -> pd.DataFrame:
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
