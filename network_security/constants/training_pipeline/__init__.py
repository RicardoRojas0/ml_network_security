# Training pipeline constants
TARGET_COLUMN: str = "Result"  # This comes from the dataset
PIPELINE_NAME: str = "network_security"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "phisingData.csv"
TRAIN_FILE_NAME: str = "train_data.csv"
TEST_FILE_NAME: str = "test_data.csv"

# Data ingestion constants
DATA_INGESTION_COLLECTION_NAME: str = "network_security_data"
DATA_INGESTION_DATABASE_NAME: str = "machine_learning_db"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
