import yaml
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
import os,sys

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