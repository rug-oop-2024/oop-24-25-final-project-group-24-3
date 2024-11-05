from abc import ABC, abstractmethod
# from typing import Any
import numpy as np
import sklearn.metrics as skl


METRICS = [
    "mean_squared_error",
    "accuracy",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str, predictions: np.ndarray, ground_truth: np.ndarray):
    """
    Factory function to get a metric by name.
    Return a metric instance given its str name.
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

    def _init_(self, predictions: np.ndarray, ground_truth: np.ndarray)\
            -> float:
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.size = len(predictions)
        self.result = None

    @abstractmethod
    def _call_(self):
        pass


class Accuracy(Metric):
    """
    Returns the accuracy of the model's predictions
    """
    def _init_(self, predictions: np.ndarray, ground_truth: np.ndarray)\
            -> None:
        super()._init_(predictions, ground_truth)
        self.size = len(predictions)

    def _call_(self) -> float:
        self.result = np.sum(self.predictions == self.ground_truth) / self.size

        return self.result


class AUC(Metric):
    """
    Returns the model's classification performance
    """
    def _init_(self, predictions: np.ndarray, ground_truth: np.ndarray)\
            -> None:
        super()._init_(predictions, ground_truth)
        self.size = len(predictions)

    def _call_(self) -> float:
        data = np.column_stack((self.predictions, self.ground_truth))

        # Calculate TPR and FPR
        true_positives = 0
        false_positives = 0
        true_pos_rate = []
        false_pos_rate = []
        positives = np.sum(self.predictions)
        negatives = len(self.ground_truth) - positives

        for prediction, true in data:
            if true == 1 and prediction == 1:  # True Positive
                true_positives += 1
            elif true == 0 and prediction == 1:  # False Positive
                false_positives += 1

            true_pos_rate.append(true_positives / positives)
            false_pos_rate.append(false_positives / negatives)

        # Calculate AUC using the trapezoidal rule
        self.result = np.trapz(true_pos_rate, false_pos_rate)

        return self.result


class Recall(Metric):
    """
    Returns the the proportion of actual positive cases correctly
    identified by the model
    """
    def _init_(self, predictions: np.ndarray,
               ground_truth: np.ndarray) -> None:
        super()._init_(predictions, ground_truth)
        self.size = len(predictions)

    def _call_(self) -> float:
        data = np.column_stack((self.predictions, self.ground_truth))
        true_positives = 0
        false_negatives = 0

        for prediction, true in data:
            if true == 1 and prediction == 1:  # True Positive
                true_positives += 1
            elif true == 1 and prediction == 0:  # False Negative
                false_negatives += 1

        self.result = true_positives / (true_positives + false_negatives)

        return self.result


class MeanSquaredError(Metric):
    """
    Returns the mean squared error of the model's predictions
    """
    def _init_(self, predictions: np.ndarray, ground_truth: np.ndarray)\
            -> None:
        super()._init_(predictions, ground_truth)
        self.size = len(predictions)

    def _call_(self) -> float:
        self.result = np.sum(
            (self.predictions - self.ground_truth)**2
        ) / self.size

        return self.result


class MeanAbsoluteError(Metric):
    """
    Returns the mean absolute error of the model's predictions
    """
    def _init_(self, predictions: np.ndarray, ground_truth: np.ndarray)\
            -> None:
        super()._init_(predictions, ground_truth)
        self.size = len(predictions)

    def _call_(self) -> float:
        self.result = np.sum(
            np.abs(self.predictions - self.ground_truth)
        ) / self.size

        return self.result


class RSquared(Metric):
    """
    Measures how well the model's predictions approximate to actual data points
    """
    def _init_(self, predictions: np.ndarray,
               ground_truth: np.ndarray) -> None:
        super()._init_(predictions, ground_truth)
        self.size = len(predictions)

    def _call_(self):
        diff = self.ground_truth - self.predictions
        mean_diff = self.ground_truth - np.mean(self.ground_truth)
        self.result = 1 - (np.sum(diff * 2) / np.sum(mean_diff * 2))
        return self.result


if __name__ == "_main_":
    predictions_c = np.array([1, 0, 0, 1, 1])
    labels_c = np.array([1, 1, 0, 1, 0])

    predictions_r = np.array([1, 4, 5, 6, 3, 4])
    labels_r = np.array([1, 4, 2, 5, 3, 6])

    accuracy = get_metric("accuracy", predictions_c, labels_c)
    auc = get_metric("auc", predictions_c, labels_c)
    recall = get_metric("recall", predictions_c, labels_c)

    mse = get_metric("mse", predictions_r, labels_r)
    mae = get_metric("mae", predictions_r, labels_r)
    rsquared = get_metric("rsquared", predictions_r, labels_r)

    print("Accuracy:", accuracy(), skl.accuracy_score(labels_c, predictions_c))
    print("MSE:", mse(), skl.mean_squared_error(labels_r, predictions_r))
    print("AUC:", auc(), skl.roc_auc_score(labels_c, predictions_c))
    print("MAE:", mae(), skl.mean_absolute_error(labels_r, predictions_r))
    print("Recall:", recall(), skl.recall_score(labels_c, predictions_c))
    print("R Squared:", rsquared(), skl.r2_score(labels_r, predictions_r))
