import sys
from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

# Main function to initiate data ingestion process
if __name__ == "__main__":
    try:
        # Initiating training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # Initiating data ingestion configuration
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Initiating data ingestion process...")

        # Initiating data ingestion process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion process completed successfully.")
        print(data_ingestion_artifact)

        # Initiating data validation configuration
        data_validation_config = DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config,
        )
        logging.info("Initiating data validation process...")

        # Inititating data validation process
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation process completed successfully")
        print(data_validation_artifact)

        # Initiating data transformation configuration
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config,
        )
        logging.info("Initiating data transformation proces...")

        # Initiating data transformation process
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        logging.info("Data transformation process completed successfully")
        print(data_transformation_artifact)

        # Initiating model trainer configuration
        model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=training_pipeline_config
        )
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )
        logging.info("Initiating model trainer process...")

        # Initiating model trainer process
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model trainer process completed successfully")

    except NetworkSecurityException as e:
        raise NetworkSecurityException(e, sys)
