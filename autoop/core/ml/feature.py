from pydantic import BaseModel
from typing import Literal, Optional
import numpy as np
import pandas as pd
from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """
    Handles feature properties and statistics.
    """
    name: str  # Feature name
    dtype: str  # Data type, e.g., 'int', 'float', etc.
    feature_type: Literal['numerical', 'categorical']  # Feature type
    dataset: Optional[Dataset] = None  # Dataset containing the feature values

    def __str__(self) -> str:
        """
        Returns a string representation of the feature.

        Returns:
            str: Feature details as a string.
        """
        return (
            f"Feature(name={self.name}, dtype={self.dtype}, "
            f"type={self.feature_type})"
        )

    def calculate_statistics(self) -> dict:
        """
        Calculates statistics for numerical features.

        Returns:
            dict: Statistics like mean, median, and standard deviation.

        Raises:
            ValueError: If dataset is not set.
            TypeError: If feature is not numerical.
        """
        if self.dataset is None:
            raise ValueError("Dataset not set.")

        data = self.dataset.read()
        if self.feature_type != 'numerical':
            raise TypeError("Only numerical features have statistics.")

        return {
            'mean': np.mean(data[self.name]),
            'median': np.median(data[self.name]),
            'std_dev': np.std(data[self.name]),
            'min': np.min(data[self.name]),
            'max': np.max(data[self.name])
        }

    @staticmethod
    def detect_feature_type(values: pd.Series) -> Literal['numerical',
                                                          'categorical']:
        """
        Detects feature type based on series data.

        Args:
            values (pd.Series): Series to detect type.

        Returns:
            Literal['numerical', 'categorical']: Detected feature type.
        """
        return (
            'numerical'
            if np.issubdtype(values.dtype, np.number)
            else 'categorical'
        )

    def set_values(self, dataset: Dataset):
        """
        Sets the dataset and detects feature type.

        Args:
            dataset (Dataset): Dataset to set for the feature.
        """
        self.dataset = dataset
        data = dataset.read()
        self.feature_type = self.detect_feature_type(data[self.name])
