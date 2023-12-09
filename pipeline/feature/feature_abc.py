from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
from pathlib import Path

# dictionaire d info avec les path, le name, les options der group et de clean?
# feature name pour les log
# extract_path, extract_summary_path, preproc_path, summary_path, (cleaned_path?)


class Feature(ABC):
    """
    Abstract base class for a feature in the dataset.
    Defines the structure and required methods for a feature.
    """

    @abstractproperty
    def df(self):
        return self.df

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
