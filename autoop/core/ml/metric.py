from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import sklearn.metrics as skl


METRICS = [
    "mean_squared_error",
    "accuracy",
    "auc",
    "mae",
    "recall",
    "rsquared",
]


def get_metric(name: str, predictions: np.ndarray,
               ground_truth: np.ndarray) -> 'Metric':
    """
    Factory function to get a metric instance by name.

    Args:
        name (str): Name of the metric.
        predictions (np.ndarray): Model predictions.
        ground_truth (np.ndarray): Actual values.

    Returns:
        Metric: Instance of the requested metric class.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    if name == "accuracy":
        return Accuracy(predictions, ground_truth)
    elif name == "mse":
        return MeanSquaredError(predictions, ground_truth)
    elif name == "auc":
        return AUC(predictions, ground_truth)
    elif name == "mae":
        return MeanAbsoluteError(predictions, ground_truth)
    elif name == "recall":
        return Recall(predictions, ground_truth)
    elif name == "rsquared":
        return RSquared(predictions, ground_truth)
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")


class Metric(ABC):
    """
    Base class for all metrics.
    """
    def __init__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> None:
        """
        Initializes the Metric with predictions and ground truth.

        Args:
            predictions (np.ndarray): Model predictions.
            ground_truth (np.ndarray): Actual values.
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.size = len(predictions)
        self.result: Union[float, None] = None

    @abstractmethod
    def __call__(self) -> float:
        """
        Calculate the metric.

        Returns:
            float: The computed metric value.
        """
        pass


class Accuracy(Metric):
    """
    Calculates the accuracy of the model's predictions.
    """
    def __call__(self) -> float:
        """Compute the accuracy score.

        Returns:
            float: The accuracy value.
        """
        self.result = np.mean(self.predictions == self.ground_truth)
        return self.result


class AUC(Metric):
    """
    Calculates the Area Under the Curve (AUC) for classification.
    """
    def __call__(self) -> float:
        """Compute the AUC score.

        Returns:
            float: The AUC value.
        """
        self.result = skl.roc_auc_score(self.ground_truth, self.predictions)
        return self.result


class Recall(Metric):
    """
    Calculates the recall score for classification.
    """
    def __call__(self) -> float:
        """Compute the recall score.

        Returns:
            float: The recall value.
        """
        true_positives = np.sum(
            (self.ground_truth == 1) & (self.predictions == 1)
        )
        false_negatives = np.sum(
            (self.ground_truth == 1) & (self.predictions == 0)
        )
        self.result = (true_positives / (true_positives + false_negatives)
                       if (true_positives + false_negatives) > 0 else 0.0)
        return self.result


class MeanSquaredError(Metric):
    """
    Calculates the Mean Squared Error (MSE) for regression.
    """
    def __call__(self) -> float:
        """Compute the Mean Squared Error.

        Returns:
            float: The MSE value.
        """
        self.result = np.mean((self.predictions - self.ground_truth) ** 2)
        return self.result


class MeanAbsoluteError(Metric):
    """
    Calculates the Mean Absolute Error (MAE) for regression.
    """
    def __call__(self) -> float:
        """Compute the Mean Absolute Error.

        Returns:
            float: The MAE value.
        """
        self.result = np.mean(np.abs(self.predictions - self.ground_truth))
        return self.result


class RSquared(Metric):
    """
    Calculates the R-squared (coefficient of determination) for regression.
    """
    def __call__(self) -> float:
        """Compute the R-squared score.

        Returns:
            float: The R-squared value.
        """
        total_variance = np.sum((self.ground_truth - np.mean(
            self.ground_truth)) ** 2)
        explained_variance = np.sum(
            (self.predictions - self.ground_truth) ** 2)
        self.result = 1 - (explained_variance / total_variance)
        return self.result
