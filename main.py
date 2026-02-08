from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from CustomerChurn.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact
from CustomerChurn.components.data_ingestion import DataIngestion
from CustomerChurn.components.data_validation import DataValidation
from CustomerChurn.components.data_transformation import DataTransformation
from CustomerChurn.components.model_trainer import ModelTrainer
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
    
        #Data Validation
        logging.info("Data Validation Started")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()   
        logging.info("Data Validation completed and artifact:")
        print(data_validation_artifact)

        #Data Transformation
        logging.info("Data Transformation Started")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed and artifact:")
        print(data_transformation_artifact)

        #Model Trainer
        logging.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Training completed")

    except Exception as e:
        raise CustomerChurnException(e, sys)
    