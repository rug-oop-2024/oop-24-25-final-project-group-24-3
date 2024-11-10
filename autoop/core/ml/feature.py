from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """Represents a feature in a dataset with its type,
    name, and other attributes."""
    name: str = Field(..., description="The name of the feature")
    type: Literal["numerical", "categorical"] = Field(
        ..., description="The type of the feature"
    )
    dataset: Dataset = Field(
        ..., description="The dataset to which this feature belongs")
    values: np.ndarray = Field(..., description="The values for this feature")

    def __str__(self) -> str:
        """
        Returns a string representation of the feature.

        Returns:
            str: Feature details as a string.
        """
        return (
            f"Feature(name={self.name}, type={self.type}, "
            f"dataset={self.dataset}, values={self.values})"
        )
