import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

##Read the data from source

class ChurnDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomerChurnException(e,sys)
    def cv_to_json_convertor(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values()) 
            ## converting into list of json arrays
            return records
        except Exception as e:
            raise CustomerChurnException(e,sys)
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise CustomerChurnException(e,sys)
        

if __name__=='__main__':
    FILE_PATH = "ChurnData\Telco_customer_churn.csv"
    DATABASE = "SRINIDHI"
    Collection = "NetworkData"
    churnobj = ChurnDataExtract()
    records = churnobj.cv_to_json_convertor(FILE_PATH)
    no_of_records = churnobj.insert_data_mongodb(records,DATABASE,Collection)
    print(f"Total number of records inserted in MongoDB are : {no_of_records}")
