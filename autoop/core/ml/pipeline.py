from typing import List, Dict, Any, Union
import pickle
import numpy as np
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features


class Pipeline:
    """
    A class for building and executing a machine learning pipeline
    with preprocessing, training, and evaluation.
    """
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8) -> None:
        """
        Initializes the Pipeline class.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._split = split

        is_categorical = target_feature.type == "categorical"
        is_classification = model.type != "classification"

        if is_categorical and is_classification:
            raise ValueError(
                "Model type must be classification for"
                " categorical target feature"
            )

        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """
        Provides a formatted string representation of the pipeline,
        detailing the model type, input features, target feature,
        data split ratio, and evaluation metrics.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """Returns a saved copy of the model."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Returns a list of artifacts generated during pipeline execution."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder", "StandardScaler"]:
                if artifact_type == "OneHotEncoder":
                    data = artifact["encoder"]
                else:
                    data = artifact["scaler"]
                artifacts.append(Artifact(name=name, data=pickle.dumps(data)))

        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact
                         (name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Dict[str, Any]) -> None:
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        target_feature_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)

        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)

        self._output_vector = target_data
        self._input_vectors = [data for _, data, _ in input_results]

    def _split_data(self) -> None:
        split_index = int(self._split * len(self._output_vector))
        self._train_X = [vector[:split_index] for
                         vector in self._input_vectors]
        self._test_X = [vector[split_index:] for vector in self._input_vectors]
        self._train_y = self._output_vector[:split_index]
        self._test_y = self._output_vector[split_index:]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)

        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))

        self._predictions = predictions

    def _evaluate_train(self):
        """Evaluate the model on the training set."""
        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        self._train_metrics_results = []
        train_predictions = self._model.predict(X_train)
        for metric in self._metrics:
            result = metric.evaluate(train_predictions, Y_train)
            self._train_metrics_results.append((metric, result))

    def execute(self) -> Dict[str, Union[List[Any], np.array]]:
        """
        Executes the entire pipeline process including preprocessing,
        training, and evaluation, and returns metrics for both train
        and test sets.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate_train()
        self._evaluate()

        return {
            "train_metrics": self._train_metrics_results,
            "test_metrics": self._metrics_results,
            "predictions": self._predictions,
        }
