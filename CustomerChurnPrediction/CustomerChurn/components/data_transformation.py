import sys
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from CustomerChurn.constants.training_pipeline import USE_FEATURE_SCALING,ORDINAL_FEATURES,NOMINAL_FEATURES,SERVICE_FEATURES,DROP_COLUMNS,BINARY_FEATURES,NUMERICAL_FEATURES
from CustomerChurn.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from CustomerChurn.entity.config_entity import DataTransformationConfig
from CustomerChurn.utils.main_utils.utils import save_numpy_array_data,save_object
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.constants.training_pipeline import TARGET_COLUMN

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    #step 1: read data
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    #step 2: preprocessing pipeline - Feature Encoding & Missing Value Imputation
    def get_preprocessor_object_without_scaling(self) -> ColumnTransformer:
        try:
            logging.info("Creating preprocessing object without scaling")

            
            # Ordinal categorical pipeline (Contract)
            ordinal_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[['Month-to-month', 'One year', 'Two year']]
            ))
        ])
            # Nominal categorical pipeline
            nominal_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

            preprocessor = ColumnTransformer(
            transformers=[
                
                ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
                ("nominal", nominal_pipeline, NOMINAL_FEATURES),
                ("service", nominal_pipeline, SERVICE_FEATURES),
                ("num", num_pipeline, NUMERICAL_FEATURES)
            ]
        )
            

            return preprocessor

        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    def get_preprocessor_object_with_scaling(self) -> ColumnTransformer:
        try:
            logging.info("Creating preprocessing object with scaling")
            
            ordinal_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    categories=[['Month-to-month', 'One year', 'Two year']]
                ))
            ])

            nominal_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ])

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                
                    ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
                    ("nominal", nominal_pipeline, NOMINAL_FEATURES),
                    ("service", nominal_pipeline, SERVICE_FEATURES),
                    ("num", num_pipeline, NUMERICAL_FEATURES)
                ]
            )
                

            return preprocessor
        except Exception as e:
            raise CustomerChurnException(e, sys)



    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            
            #Reading the validated train & test data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            #Drop the columns
            train_df.drop(columns=DROP_COLUMNS,axis=1,inplace=True)
            test_df.drop(columns=DROP_COLUMNS,axis=1,inplace=True)

            train_df["Total Charges"] = train_df["Total Charges"].replace(" ",np.nan)
            test_df["Total Charges"] = test_df["Total Charges"].replace(" ",np.nan)

            #Change the TotalCharges column type from object to float
            train_df['Total Charges'] = pd.to_numeric(train_df['Total Charges'],errors='coerce')
            test_df['Total Charges'] = pd.to_numeric(test_df['Total Charges'],errors='coerce')

            #fill the missing values in TotalCharges column with value 0
            train_df['Total Charges'] = train_df['Total Charges'].fillna(0)
            test_df['Total Charges'] = test_df['Total Charges'].fillna(0)

            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            # Map Yes/No columns to 0/1
            for col in ['Senior Citizen','Partner','Dependents','Phone Service','Paperless Billing']:
                train_df[col] = train_df[col].map({'No':0, 'Yes':1})
                test_df[col] = test_df[col].map({'No':0, 'Yes':1})

            # Map Gender separately
            train_df['Gender'] = train_df['Gender'].map({'Female':0, 'Male':1})
            test_df['Gender'] = test_df['Gender'].map({'Female':0, 'Male':1})

            print("\nClass distribution BEFORE SMOTE:")
            print(target_feature_train_df.value_counts())
            logging.info(f"Class distribution BEFORE SMOTE:\n{target_feature_train_df.value_counts()}")

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)   
            target_feature_test_df = test_df[TARGET_COLUMN]

            #applying preprocessing object on training and testing dataframe
            if USE_FEATURE_SCALING:
                preprocessor = self.get_preprocessor_object_with_scaling()
            else:
                preprocessor = self.get_preprocessor_object_without_scaling()
            print("ORDINAL_FEATURES:", ORDINAL_FEATURES)
            print("Columns in input_feature_train_df:", list(input_feature_train_df.columns))

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

            X_train_resampled, y_train_resampled = smote.fit_resample(transformed_input_train_feature,target_feature_train_df)

            print("\nClass distribution AFTER SMOTE:")
            print(pd.Series(y_train_resampled).value_counts())
            logging.info(f"Class distribution AFTER SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, ) #saves the train_df as train.npy
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object)

            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise CustomerChurnException(e,sys)