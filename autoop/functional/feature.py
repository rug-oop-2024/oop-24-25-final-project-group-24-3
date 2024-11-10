from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects and assigns types to features in the dataset, classifying each as
    either numerical or categorical.

    Args:
        dataset (Dataset): The dataset containing features to be classified.

    Returns:
        List[Feature]: A list of Feature objects with their types set to
        either "numerical" or "categorical".
    """
    features = []
    for column_name in dataset.data.columns:
        if dataset.data[column_name].dtype.kind in 'biufc':
            # Numeric types: boolean, integer, unsigned, float, complex
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column_name, type=feature_type)
        features.append(feature)

    return features
