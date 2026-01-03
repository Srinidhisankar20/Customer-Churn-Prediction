from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os, sys

class ChurnModel:
    def __init__(Self,preprocessor,model):
        try:
            Self.preprocessor=preprocessor
            Self.model=model
        except Exception as e:
            raise CustomerChurnException(e,sys)
    def predict(self,X):
        try:
            X_preprocessed=self.preprocessor.transform(X)
            preds=self.model.predict(X_preprocessed)
            return preds
        except Exception as e:
            raise CustomerChurnException(e,sys)