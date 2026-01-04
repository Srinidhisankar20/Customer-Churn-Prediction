from dataclasses import dataclass
import numpy as np

#Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

#Data Validation Artifact
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str   
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

#Data Transformation Artifact
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy_score: float
    y_true: np.ndarray
    y_pred: np.ndarray

#Model Trainer Artifact
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    test_metric_artifact: ClassificationMetricArtifact
    best_model_name: str
    best_model_score: float
    

