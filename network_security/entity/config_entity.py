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
