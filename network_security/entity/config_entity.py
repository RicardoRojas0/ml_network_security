import os
from datetime import datetime
from network_security.constants import training_pipeline


class TrainingPipelineConfig:
    """
    Configuration class for the training pipeline.

    This class manages the settings and paths required for the training pipeline,
    including the pipeline name, artifact directory, and timestamp.

    Attributes:
        pipeline_name (str): The name of the training pipeline.
        artifact_name (str): The directory name where artifacts are stored.
        artifact_dir (str): The full path to the artifact directory, including a timestamp.
        timestamp (str): The timestamp when the configuration was created, formatted as "%Y_%m_%d_%H_%M_%S".
    """

    def __init__(self, timestamp=datetime.now()):
        """
        Initializes the TrainingPipelineConfig instance.

        Args:
            timestamp (datetime, optional): The timestamp for the configuration, defaults to the current datetime.
        """
        timestamp = timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp: str = timestamp


class DataIngestionConfig:
    """
    Configuration for data ingestion process.

    Attributes:
        data_ingestion_dir (str): Directory path for data ingestion artifacts.
        feature_store_file_path (str): File path for the feature store.
        training_file_path (str): File path for the training data.
        testing_file_path (str): File path for the testing data.
        train_test_split_ration (float): Ratio for splitting data into training and testing sets.
        collection_name (str): Name of the collection in the database.
        database_name (str): Name of the database.

    Args:
        training_pipeline_config (TrainingPipelineConfig): Configuration for the training pipeline.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME,
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME,
        )
        self.train_test_split_ratio: float = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    """
    DataValidationConfig is a configuration class for setting up paths related to data validation
    in the pipeline.

    Attributes:
        data_validation_dir (str): Directory for storing data validation artifacts.
        valid_data_dir (str): Directory for storing valid data.
        invalid_data_dir (str): Directory for storing invalid data.
        valid_train_file_path (str): Path to the valid training data file.
        valid_test_file_path (str): Path to the valid test data file.
        invalid_training_file_path (str): Path to the invalid training data file.
        invalid_test_file_path (str): Path to the invalid test data file.
        drift_report_file_path (str): Path to the data drift report file.

    Args:
        training_pipeline_config (TrainingPipelineConfig): Configuration object for the training pipeline.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR,
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR,
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME,
        )
        self.invalid_training_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME,
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME,
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME,
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = (
            training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
        )
