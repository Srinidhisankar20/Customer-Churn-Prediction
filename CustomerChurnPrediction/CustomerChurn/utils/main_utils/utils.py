import yaml
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from CustomerChurn.utils.ml_utils.metric.classification_metric import get_classification_score
import os,sys
import numpy as np
import pickle


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomerChurnException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        # Remove existing file if replace=True
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        # Create directories if not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Always write YAML file
        with open(file_path, "w") as file:
            yaml.dump(content, file)

        logging.info(f"YAML file successfully written at: {file_path}")

    except Exception as e:
        raise CustomerChurnException(e, sys)
    
#saving the data to file as numpy array
def save_numpy_array(file_path: str,array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise CustomerChurnException(e,sys) from e

def load_numpy_array(file_path:str,) -> object:
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomerChurnException(e,sys) from e
    
def save_object(file_path: str, obj: object)->None:
    try:
        logging.info("Entered the save_object method to save the preprocessed object")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("saved the preprocessed object as pickle file")
    except Exception as e:
        raise CustomerChurnException(e,sys) from e
    
def load_object(file_path:str,) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomerChurnException(e,sys) from e

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for model_name,model in models.items():
            para = params[model_name]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            classification_test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            #choosing the metric for comparison
            report[model_name] = {
                                "f1_score": classification_test_metrics.f1_score,
                                "recall_score": classification_test_metrics.recall_score,
                                "precision_score": classification_test_metrics.precision_score,
                                "accuracy_score": classification_test_metrics.accuracy_score
                            }

        return report
    

    except Exception as e:
        raise CustomerChurnException(e,sys) from e

'''def evaluate_models(X_train,y_train,X_test,y_test,models,param)->dict:
    try:
        report={}
        for model_name,model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            test_model_score = get_classification_score(y_true=y_test,y_pred=y_test_pred)
            report[model_name] = {"Model": model,
                                  "Train_Score": train_model_score,
                                    "Test_Score": test_model_score}
    except Exception as e:
        raise CustomerChurnException(e,sys) from e'''