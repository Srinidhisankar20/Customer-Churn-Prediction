# Here we initiate all the stages in the pipeline
import os,sys
from CustomerChurn.logging.logger import logging
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.components.data_ingestion import DataIngestion
from CustomerChurn.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from CustomerChurn.entity.artifact_entity import DataIngestionArtifact
#from CustomerChurn.cloud.s3_syncer import S3Sync

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        #self.s3_sync = S3Sync()

    # Start Data Ingestion 
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys)
    

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise CustomerChurnException(e, sys)