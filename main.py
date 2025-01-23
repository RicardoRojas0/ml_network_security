import sys
from network_security.components.data_ingestion import DataIngestion
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
)

# Main function to initiate data ingestion process
if __name__ == "__main__":
    try:
        # Initialize data ingestion configuration
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Initiating data ingestion process...")

        # Initiating data ingestion process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion process completed successfully.")
        print(data_ingestion_artifact)

    except NetworkSecurityException as e:
        raise NetworkSecurityException(e, sys)
