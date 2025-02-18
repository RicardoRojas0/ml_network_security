import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from network_security.constants.training_pipeline import TARGET_COLUMN
from network_security.constants.training_pipeline import (
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)

from network_security.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)

from network_security.entity.config_entity import DataTransformationConfig

from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact: DataValidationArtifact = (
                data_validation_artifact
            )
            self.data_transformation_config: DataTransformationConfig = (
                data_transformation_config
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation Class"
        )
        try:
            # Initializing a KNNImputer object with the key-value pair params (**)
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"KNNImputer initialized with this parameters: {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            # Intializing a Pipeline object with the imputer
            processor: Pipeline = Pipeline([("imputer", imputer)])

            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(
            "Entered the initiate_data_transformation method of DataTransformation Class"
        )
        try:
            logging.info("Initiating data transformation")

            # Getting the artifacts file paths from the validation stage for training and test sets
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Creating dependent (target) and independent (features) variables for training set
            input_features_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            # Replacing the -1 values for 0 values in the target feature
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # Creating dependent (target) and independent (features) variables for test set
            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Replacing the -1 values for 0 values in the target feature
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # Imputing the NaN values of the independent variables for both training and test sets
            preprocessor = self.get_data_transformer_object()
            transformed_input_features_train = preprocessor.fit_transform(
                input_features_train_df
            )
            transformed_input_features_test = preprocessor.transform(
                input_features_test_df
            )

            # Combining the transformed features with the target into an array for both training and test sets
            train_array = np.c_[
                transformed_input_features_train, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                transformed_input_features_test, np.array(target_feature_test_df)
            ]

            # Saving numpy arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_array,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_array,
            )

            # Saving preprocessor object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                object=preprocessor,
            )

            # Hardcoded preprocessor pusher
            save_object("final_model/preprocessor.pkl", preprocessor)

            # Data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            )

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
