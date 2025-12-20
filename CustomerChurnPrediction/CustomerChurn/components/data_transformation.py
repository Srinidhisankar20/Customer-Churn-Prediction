import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from CustomerChurn.constants.training_pipeline import *
from CustomerChurn.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from CustomerChurn.entity.config_entity import DataTransformationConfig
from CustomerChurn.utils.main_utils.utils import save_numpy_array_data,save_object
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise CustomerChurnException(e,sys)
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
        except Exception as e:
            raise CustomerChurnException(e,sys)