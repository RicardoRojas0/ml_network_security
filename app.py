import os
import sys

import certifi
from dotenv import load_dotenv

import pymongo
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.pipeline.training_pipeline import TrainingPipeline
from network_security.utils.main_utils.utils import load_object
from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.constants.training_pipeline import (
    DATA_INGESTION_DATABASE_NAME,
    DATA_INGESTION_COLLECTION_NAME,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
import pandas as pd

# Obtain the certificate authority path
certificate_authority_path = certifi.where()

# Setting up MongoDB
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=certificate_authority_path)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Setting up FastAPI
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 templates for rendering HTML pages in FastAPI endpoints
templates = Jinja2Templates(directory="./templates")


# Home page
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


# Training page
@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training process was successfully completed.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# Getting predictions
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Load the data, model and preprocessor
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        # Create the model
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Make predictions
        y_pred = network_model.predict(df)
        print(f"Predictions: {y_pred}")

        # Add the predictions to the dataframe
        df["predicted_column"] = y_pred

        # Saving output into CSV format
        df.to_csv("prediction_output/output.csv")

        # Display information in the form of table
        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)
