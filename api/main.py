# references
# https://pydantic-docs.helpmanual.io/usage/validators/

# import pickle

import pandas as pd

# import numpy as np
from fastapi import FastAPI

# from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, validator

# internal dependencies
from .ml.model import CreditRisk_Preprocess_Predict
from .ml.model import CreditRisk_Classifier, get_input_column_names

# from enum import Enum
# from io import StringIO


# create fastapi app
app = FastAPI(
    title="Credit risk prediction API",
    description="A Smart Data Science Application running on FastAPI + uvicorn \
                  for credit risk classification",
    version="1.0.0",
)


class ClientProfile(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: float
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str
    model_version: str

    @validator("model_version")
    def check_model_version_value(self, v):
        if v not in ["v0", "v1", "v2"]:
            raise ValueError("model_version equal to 'v0', 'v1' or 'v2'")
        return v.lower()


@app.post("/predict")
async def predict_risk(client: ClientProfile):
    # load the api input 
    data = client.dict()

    # model version choosed by the user
    model_version = data["model_version"]

    # store the list of list
    data_in = [
        [
            data["checking_status"],
            data["duration"],
            data["credit_history"],
            data["purpose"],
            data["credit_amount"],
            data["savings_status"],
            data["employment"],
            data["installment_commitment"],
            data["personal_status"],
            data["other_parties"],
            data["residence_since"],
            data["property_magnitude"],
            data["age"],
            data["other_payment_plans"],
            data["housing"],
            data["existing_credits"],
            data["job"],
            data["num_dependents"],
            data["own_telephone"],
            data["foreign_worker"],
        ]
    ]

    # load input data in a dataframe
    df_input = pd.DataFrame(data_in, columns=get_input_column_names())

    # preprocess the entry
    preprocess_pipe = CreditRisk_Preprocess_Predict(df_input)

    # prepare features to be passed to the classifier based on the model version 
    if model_version.lower() == 'v0':
        # do a scaling on numerical inputs
        X_input = preprocess_pipe.prepare_input_features()
    else:
        # no scaling on the numerical inputs 
        X_input = preprocess_pipe.prepare_input_features()

    # init classifier object
    classifier = CreditRisk_Classifier()

    # load the model saved as a pickle. load model based on the version choosed
    loaded_model = classifier.load_model(model_version)

    # get the class and probability using the model
    prediction = loaded_model.predict(X_input)
    probability = loaded_model.predict_proba(data_in).max()

    return {"prediction": prediction[0], "probability": probability}


# # endpoint to process a file as the input
# @app.post("/files/")
# async def create_file(file: bytes = File(...), token: str = Form(...)):
#     s=str(file,'utf-8')
#     data = StringIO(s)
#     df=pd.read_csv(data)
#     print(df)

#     #return df
#     return {
#         "file": df,
#         "token": token,
#     }
