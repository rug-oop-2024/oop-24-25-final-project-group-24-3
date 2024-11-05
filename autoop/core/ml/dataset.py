from autoop.core.ml.artifact import Artifact
# from abc import ABC, abstractmethod
import pandas as pd
import io
# test


class Dataset(Artifact):
    """
    Represents a dataset artifact.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a Dataset with type set to "dataset".
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0"):
        """
        Creates a Dataset from a DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to save.
            name (str): Name of the dataset.
            asset_path (str): Storage path for the dataset.
            version (str): Version of the dataset.

        Returns:
            Dataset: A new Dataset instance.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads and returns the dataset as a DataFrame.

        Returns:
            pd.DataFrame: The dataset.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Converts a DataFrame to bytes and saves it.

        Args:
            data (pd.DataFrame): The DataFrame to save.

        Returns:
            bytes: The saved data in bytes format.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
