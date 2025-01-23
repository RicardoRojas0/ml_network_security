import os
import sys
import pymongo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import DataIngestionArtifact
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the MongoDB URL from the environment variables
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    """
    DataIngestion is responsible for handling the data ingestion process for network security analysis.
    This includes exporting data from a MongoDB collection, storing it into a feature store, and splitting
    the data into training and testing sets.

    Attributes:
        data_ingestion_config (DataIngestionConfig):
            Configuration for data ingestion, including database details, file paths, and split ratios.

    Methods:
        __init__(data_ingestion_config: DataIngestionConfig):
            Initializes the DataIngestion instance with the provided configuration.

        export_collection_as_dataframe():
            Exports a MongoDB collection as a pandas DataFrame, handling '_id' column and "na" values.

        export_data_into_feature_store(dataframe: pd.DataFrame):
            Exports the given DataFrame into a feature store file in CSV format.

        split_data_as_train_test(dataframe: pd.DataFrame):
            Splits the given dataframe into training and testing sets, saves them as CSV files, and logs the process.

        initiate_data_ingestion():
            Initiates the data ingestion process which includes exporting data from a collection,
            storing it into a feature store, and splitting the data into training and testing sets.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion instance with the provided configuration.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        """
        Export a MongoDB collection as a pandas DataFrame.

        This method connects to a MongoDB database using the configuration provided
        in the data_ingestion_config attribute, retrieves the specified collection,
        and converts it into a pandas DataFrame. The method also handles the following:
        - Drops the '_id' column created by MongoDB.
        - Replaces "na" values with np.nan.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the MongoDB collection.

        Raises:
            NetworkSecurityException: If there is any exception during the process.
        """
        try:
            # Connect to MongoDB and retrieve the collection
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            # Drop the _id column created by MongoDB
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            # Replace "na" values with np.nan
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Exports the given DataFrame into a feature store file in CSV format.
        This method ensures that the directory for the feature store file exists,
        and then writes the DataFrame to the specified file path in CSV format.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be exported to the feature store.

        Raises:
            NetworkSecurityException: If there is any exception during the export process.
        """

        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Create the directory if it does not exist for the feature store file
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Export the DataFrame to a CSV file
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the given dataframe into training and testing sets, saves them as CSV files, and logs the process.

        Args:
            dataframe (pd.DataFrame): The dataframe to be split into training and testing sets.

        Raises:
            NetworkSecurityException: If an error occurs during the data splitting or file operations.

        Returns:
            None
        """
        try:
            # Split the data into training and testing sets
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
            )
            logging.info("Data split into training and testing sets.")

            # Create the directory if it does not exist for the training and testing data files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Training data directory created.")

            # Export the training and testing data to CSV files
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info("Training and testing data exported successfully.")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process which includes exporting data from a collection,
        storing it into a feature store, and splitting the data into training and testing sets.

        Returns:
            DataIngestionArtifact: An object containing the file paths for the training and testing datasets.

        Raises:
            NetworkSecurityException: If any error occurs during the data ingestion process.
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
