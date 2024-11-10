from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from typing import Literal


class Model(ABC):
    """
    Base class for machine learning models.
    """

    def __init__(self, model_type: Literal["classification",
                                           "regression"]) -> None:
        """
        Initializes the model with a specified type.

        Args:
            model_type (Literal["classification",
            "regression"]): Type of model.
        """
        self.model_type = model_type
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on given data.

        Args:
            X (np.ndarray): Features for training.
            y (np.ndarray): Target values for training.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predictions.
        """
        pass

    def save(self) -> Artifact:
        """
        Save a deepcopy of the model as an artifact.

        Returns:
            Artifact: The saved model artifact.
        """
        model_copy = deepcopy(self)
        return Artifact(type="model", name="BaseModel",
                        asset_path="path/to/model", data=model_copy)


# Classification Models

class LogisticRegressionModel(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = LogisticRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class DecisionTreeClassifierModel(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = DecisionTreeClassifier()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class KNeighborsClassifierModel(Model):
    def __init__(self, n_neighbors: int = 3):
        super().__init__(model_type="classification")
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# Regression Models

class LinearRegressionModel(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class DecisionTreeRegressorModel(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = DecisionTreeRegressor()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class KNeighborsRegressorModel(Model):
    def __init__(self, n_neighbors: int = 3):
        super().__init__(model_type="regression")
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
