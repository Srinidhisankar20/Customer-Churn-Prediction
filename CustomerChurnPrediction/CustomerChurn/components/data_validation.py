from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.constants import training_pipeline
from CustomerChurn.constants.training_pipeline import SCHEMA_FILE_PATH
from CustomerChurn.entity.config_entity import TrainingPipelineConfig,DataValidationConfig
from CustomerChurn.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from CustomerChurn.utils.main_utils.utils import read_yaml_file, write_yaml_file
import os,sys
import pandas as pd
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    #step 1 : Read data from file location
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    # step 2: Validate the no of columns
    def validate_number_of_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                print("Number of columns are equal:",len(dataframe.columns))
                return True
            else:
                return False
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    #step 3 : Validate the column names
    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = set(self._schema_config["columns"].keys())
            dataframe_columns = set(dataframe.columns)

            missing_columns = expected_columns - dataframe_columns
            extra_columns = dataframe_columns - expected_columns

            if missing_columns or extra_columns:
                logging.error(f"Missing columns: {missing_columns}")
                logging.error(f"Extra columns: {extra_columns}")
                return False

            return True
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    #step 4: Validate the data types
    def validate_data_types(self, dataframe: pd.DataFrame) -> bool:
        try:
            for column, expected_dtype in self._schema_config["columns"].items():
                if column in dataframe.columns:
                    if dataframe[column].dtype.name != expected_dtype:
                        logging.error(
                            f"Column {column} expected {expected_dtype} "
                            f"but found {dataframe[column].dtype}"
                        )
                        return False
            return True
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    #step 5: Validate the target column
    def validate_target_column(self, dataframe: pd.DataFrame) -> bool:
        try:
            target_column = self._schema_config["target_column"][0]

            if target_column not in dataframe.columns:
                logging.error("Target column missing")
                return False

            if dataframe[target_column].isnull().sum() > 0:
                logging.error("Target column contains null values")
                return False

            if not set(dataframe[target_column].unique()).issubset({0, 1}):
                logging.error("Invalid values found in target column")
                return False

            return True
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    #step 6: Check for missing values    
    def check_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            missing_report = dataframe.isnull().mean()
            missing_columns = missing_report[missing_report > 0].to_dict()
            if missing_columns:
                logging.warning(f"Missing value report: {missing_columns}")
            return True
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    #step 7: Check for duplicate rows
    def check_duplicates(self, dataframe: pd.DataFrame) -> bool:
        try:
            duplicate_count = dataframe.duplicated().sum()
            if duplicate_count > 0:
                logging.warning(f"Duplicate rows found: {duplicate_count}")
            return True
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def detect_dataset_drift(self,base_df: pd.DataFrame,current_df: pd.DataFrame,threshold: float = 0.05) -> bool:
        try:
            status = True
            report = {}
            numerical_columns = self._schema_config["numerical_columns"]
            for column in numerical_columns:
                base_data = base_df[column].dropna()
                current_data = current_df[column].dropna()
                ks_result = ks_2samp(base_data, current_data)
                drift_detected = ks_result.pvalue < threshold
                if drift_detected:
                    status = False
                report[column] = {
                    "p_value": float(ks_result.pvalue),
                    "drift_detected": drift_detected
                }

            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path),exist_ok=True)
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path,content=report)

            return status
        except Exception as e:
            raise CustomerChurnException(e, sys)



    # Initiate the data validation
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read data
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)

            # =========================
            # SCHEMA & DATA VALIDATION
            # =========================
            for df, name in [(train_dataframe, "Train"), (test_dataframe, "Test")]:

                if not self.validate_number_of_columns(df):
                    raise Exception(f"{name} dataframe column count mismatch")

                if not self.validate_column_names(df):
                    raise Exception(f"{name} dataframe column name mismatch")
                else:
                    print(f"{name} Dataframe columns are as expected")

                if not self.validate_data_types(df):
                    raise Exception(f"{name} dataframe data type mismatch")
                else:
                    print(f"{name} Dataframe data types are as expected")

                if not self.validate_target_column(df):
                    raise Exception(f"{name} dataframe target validation failed")
                else:
                    print(f"{name} Dataframe target column is as expected")

                # Detection only (no fixing here)
                self.check_missing_values(df)
                self.check_duplicates(df)

            # =========================
            # DATA DRIFT CHECK
            # =========================
            drift_status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )

            # =========================
            # SAVE VALIDATED DATA
            # =========================
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True
            )

            # =========================
            # CREATE ARTIFACT
            # =========================
            data_validation_artifact = DataValidationArtifact(
                validation_status=True,   # schema validation passed
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomerChurnException(e, sys)
