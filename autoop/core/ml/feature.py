from pydantic import BaseModel
from typing import Literal, Optional
import numpy as np
import pandas as pd
from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    name: str  # Feature name
    dtype: str  # Data type, e.g., 'int', 'float', etc.
    feature_type: Literal['numerical', 'categorical']  # Feature type
    dataset: Optional[Dataset] = None  # Dataset containing the feature values

    def __str__(self) -> str:
        return (
            f"Feature(name={self.name}, dtype={self.dtype}, "
            f"type={self.feature_type})"
        )

    def calculate_statistics(self) -> dict:
        # Check if dataset is provided and extract DataFrame
        if self.dataset is None:
            raise ValueError("Dataset not set.")

        data = self.dataset.read()  # Read the dataset to get the DataFrame
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
        # Detect feature type based on the data type of the Series
        return (
            'numerical'
            if np.issubdtype(values.dtype, np.number)
            else 'categorical'
        )

    def set_values(self, dataset: Dataset):
        # Set dataset and detect feature type from it
        self.dataset = dataset
        data = dataset.read()
        self.feature_type = self.detect_feature_type(data[self.name])
