from CustomerChurn.entity.artifact_entity import ClassificationMetricArtifact
from CustomerChurn.exception.exception import CustomerChurnException
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import os,sys

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        f1score=f1_score(y_true,y_pred)
        precisionscore=precision_score(y_true,y_pred)
        recallscore=recall_score(y_true,y_pred)
        accuracyscore=accuracy_score(y_true,y_pred)
        classification_metric_artifact=ClassificationMetricArtifact(
            f1_score=f1score,
            precision_score=precisionscore,
            recall_score=recallscore,
            accuracy_score=accuracyscore,
            y_true=y_true,
            y_pred=y_pred
        )
        return classification_metric_artifact
    except Exception as e:
        raise CustomerChurnException(e,sys)