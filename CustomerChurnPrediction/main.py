from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from CustomerChurn.entity.artifact_entity import DataIngestionArtifact
from CustomerChurn.components.data_ingestion import DataIngestion
import os,sys

if __name__ == "__main__":
    try:
        logging.info("Data Ingestion Started")
        training_pipeline_config = TrainingPipelineConfig()

        #Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config) 
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed and artifact:")
        print(data_ingestion_artifact)
    except Exception as e:
        raise CustomerChurnException(e, sys)