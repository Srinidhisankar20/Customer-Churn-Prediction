from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.entity.config_entity import DataIngestionConfig
from CustomerChurn.entity.artifact_entity import DataIngestionArtifact
import os, sys
import pandas as pd
import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

"""
step 1: get the data from mongodb
step 2: save the data in feature store
step 3: split the data into train and test
step 4: save the data into ingested folder
"""

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    # step 1 : Read data from mongodb collection as a pandas dataframe
    def import_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    #step 2: Save the extracted dataframe into feature store
    def export_dataframe_to_featurestore(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_filepath = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_filepath)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_filepath,index=False,header=True)
            return dataframe
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    # step 3: split the data into train and test set
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def initiate_data_ingestion(self):
        try:
            dataframe = self.import_collection_as_dataframe()
            dataframe = self.export_dataframe_to_featurestore(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                         test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact
                                        
        except Exception as e:
            raise CustomerChurnException(e, sys)

