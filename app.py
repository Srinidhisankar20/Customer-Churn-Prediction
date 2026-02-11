#this is app.py - fast api application for training and prediction of customer churn model

import sys
import os
import pymongo
from CustomerChurn.exception.exception import CustomerChurnException
from CustomerChurn.logging.logger import logging
from CustomerChurn.pipeline.training_pipeline import TrainingPipeline
from CustomerChurn.utils.ml_utils.model.estimator import ChurnModel
from CustomerChurn.utils.main_utils.utils import load_object
from CustomerChurn.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from fastapi.responses import Response
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import certifi

ca = certifi.where()
from dotenv import load_dotenv
load_dotenv()


mongo_db_url = os.getenv("MONGO_DB_URL")

client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory = "./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training is Successfull")
    except Exception as e:
        raise CustomerChurnException(e, sys)
    
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        churn_model = ChurnModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])
        y_pred = churn_model.predict(df)
        print(y_pred)
        df['predicted_column']= y_pred
        print(df['predicted_column'])
        df.to_csv('predicted_output/output.csv')
        table_html = df.to_html(classes = 'table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise CustomerChurnException(e, sys)
    
#change the host name.
if __name__ == "__main__":
    app_run(app, host="0.0.0.0",port=8000)
