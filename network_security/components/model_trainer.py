import os
import sys

from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from network_security.entity.config_entity import ModelTrainerConfig

from network_security.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)

from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.metrics import r2_score

import mlflow


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            # Assigning scores to variables
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            # Logging metrics
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            # Logging best model
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        # Creating a dictionary of models to train
        models = {
            "Logistic Regression": LogisticRegression(verbose=1),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
        }

        # Parameters for the model training
        params = {
            "Logistic Regression": {},
            "KNN": {"n_neighbors": [3, 5, 7]},
            "Decision Tree": {"criterion": ["gini", "entropy", "log_loss"]},
            "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

        # Evaluating each of the machine learning models
        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            params=params,
        )

        # Get the best model score and name from the model_report dictionary
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        # Getting the best model
        best_model = models[best_model_name]

        # Getting predictions and scores of the best model for the train set
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(
            y_true=y_train, y_pred=y_train_pred
        )

        # Tracking experiments with MLFlow for train set
        self.track_mlflow(
            best_model=best_model, classification_metric=classification_train_metric
        )

        # Getting prediction and scores of the best model for the test set
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(
            y_true=y_test, y_pred=y_test_pred
        )

        # Tracking experimetns with MLFlow for test set
        self.track_mlflow(
            best_model=best_model, classification_metric=classification_test_metric
        )

        # Importing the pickle file and saving it into variable
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        # Creating a directory for the model to be saved in
        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        # Creating an instnace of NetworkModel with the preprocessor and the best model, then saving it into the best model folder
        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(
            self.model_trainer_config.trained_model_file_path, object=network_model
        )

        # Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            # Saving train and test data file paths into variables
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            # Loading train and test arrays
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            # Splitting train and test sets into independent and dependent variables
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            model_trainer_artifact = self.train_model(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
