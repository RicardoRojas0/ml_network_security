import os
import numpy as np

# Training pipeline constants
TARGET_COLUMN: str = "Result"  # This comes from the dataset
PIPELINE_NAME: str = "network_security"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "phisingData.csv"
TRAIN_FILE_NAME: str = "train_data.csv"
TEST_FILE_NAME: str = "test_data.csv"
SCHEMA_FILE_PATH = os.path.join("schema", "schema.yaml")

# Data ingestion constants
DATA_INGESTION_COLLECTION_NAME: str = "network_security_data"
DATA_INGESTION_DATABASE_NAME: str = "machine_learning_db"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data validation constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Data transformation constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessor.pkl"

# KNN Imputer - To replace NaN values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

# Model Trainer constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05
SAVED_MODEL_DIR = os.path.join("saved_models")