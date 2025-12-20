from dataclasses import dataclass

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
