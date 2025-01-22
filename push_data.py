import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from network_security.exceptions.exception import NetworkSecurityException

# Load the environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)


verify = certifi.where()


class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json(self, file_path):
        """
        Convert the csv file to json format
        """
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_to_mongodb(self, records, database, collection):
        """
        Insert the records to the MongoDB database
        """
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=verify)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)

            return len(self.records)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


# Ensure that the code is not executed when the module is imported
if __name__ == "__main__":
    FILE_PATH = "network_security/data/phisingData.csv"
    DATABASE = "machine_learning_db"
    collection = "network_security_data"
    network_obj = NetworkDataExtract()
    records = network_obj.csv_to_json(file_path=FILE_PATH)
    number_of_records = network_obj.insert_data_to_mongodb(
        records=records, database=DATABASE, collection=collection
    )
    print(f"Number of records inserted: {number_of_records}")
