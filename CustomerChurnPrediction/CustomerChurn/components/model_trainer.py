import os,sys
import tempfile
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ClassificationMetricArtifact
from CustomerChurn.entity.config_entity import ModelTrainerConfig
from CustomerChurn.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.metrics import classification_report
from CustomerChurn.utils.ml_utils.model.estimator import ChurnModel
from CustomerChurn.utils.main_utils.utils import save_object,load_object,load_numpy_array,evaluate_models

#import the models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
#model monitoring imports
from dotenv import load_dotenv
load_dotenv()
from urllib.parse import urlparse
import mlflow

#remote repository setup
import dagshub
dagshub.init(repo_owner='srinidhisankar21', repo_name='Customer-Churn-Prediction', mlflow=True)
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/srinidhisankar21/Customer-Churn-Prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "srinidhisankar21"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d2d4398107445e4d305af6da959ee08026f0897d"

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
        
    def track_mlflow(self,model,classification_metric):
        with mlflow.start_run():
            mlflow.set_tag("model",model.__class__.__name__)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("f1_score",classification_metric.f1_score)
            mlflow.log_metric("precision",classification_metric.precision_score)
            mlflow.log_metric("recall",classification_metric.recall_score)
            mlflow.log_metric("accuracy",classification_metric.accuracy_score)
            if hasattr(model, "oob_score_"):
                mlflow.log_metric("oob_score", model.oob_score_)
            # 🔹 Classification Report as Artifact
            report = classification_report(
                classification_metric.y_true,
                classification_metric.y_pred
            )

            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                f.write(report)
                report_path = f.name

            mlflow.log_artifact(report_path, artifact_path="classification_report")
            os.remove(report_path)
            mlflow.sklearn.log_model(sk_model = model,artifact_path="model")
            logging.info(f"Model tracked in MLflow")
    
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            #defining models
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(verbose=1,oob_score=True,random_state=42,n_jobs=-1),
                "Logistic Regression": LogisticRegression(random_state=42),

            }
            params = {

            "Decision Tree":{
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
            },
                "Random Forest":{
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"]
            },
            "Logistic Regression":{
                "penalty": ["l1", "l2"],
                "C": [0.01, 0.1, 1, 10],
                "solver": ["saga"],
                "class_weight": ["balanced"],
                "max_iter": [1000]
            }
            }

            #get model report
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                              X_test=X_test,y_test=y_test,
                                              models=models,params = params)

            #get best model name from report
            best_model_name = max(model_report, key=lambda x: model_report[x]["f1_score"])

            #get best model score from report
            best_model_score = model_report[best_model_name]["f1_score"]


            #get best model object
            ###----------here we are again training the best model.....but we already done in evaluate_models function?????-
            # it is because the evaluate model function does not return the model itself.. it just returns the score
        
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)

            classification_test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            
            ##-----------tracking using mlflow--------------------
            self.track_mlflow(best_model,classification_test_metrics)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)


            #create churn model object
            churn_model = ChurnModel(preprocessor=preprocessor,model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=churn_model)
            
            #model pusher
            save_object("final_model/model.pkl",best_model)


            #prepare the artifact
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                test_metric_artifact=classification_test_metrics,
                best_model_name=best_model_name,
                best_model_score=best_model_score
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomerChurnException(e,sys) from e
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading numpy arrays
            train_array = load_numpy_array(file_path=train_file_path)
            test_array = load_numpy_array(file_path=test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            logging.info(f"Initiating model training")
            model_trainer_artifact = self.train_model(X_train=X_train,y_train=y_train,
                                                      X_test=X_test,y_test=y_test)
            return model_trainer_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys) from e