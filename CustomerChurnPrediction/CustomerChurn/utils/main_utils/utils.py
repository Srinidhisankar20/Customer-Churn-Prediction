import yaml
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
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
def save_numpy_array_data(file_path: str,array: np.array):
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