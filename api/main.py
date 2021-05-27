# references
# https://pydantic-docs.helpmanual.io/usage/validators/

from pickle import load
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, validator
from fastapi.responses import JSONResponse
from sklearn.metrics import accuracy_score, precision_score, recall_score

from api.ml.model import CreditRisk_Classifier
from api.ml.preprocess import CreditRisk_Preprocess_Predict, get_input_column_names
from api.config import Settings
import uvicorn
import os

# create fastapi app
app = FastAPI(
    title="Credit risk prediction API",
    description="A Smart Data Science Application running on FastAPI + uvicorn \
                  for credit risk classification",
    version="1.0.0"
)

# init app settings 
settings = Settings()

# user input templates 
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
    def check_model_version_value(cls, v):
        if v not in ["v0", "v1", "v2", "v3"]:
            raise ValueError("model_version equal to 'v0', 'v1', 'v2', 'v3'")
        return v.lower()

@app.get("/healthcheck")
def healthcheck():
    # check that the model and preprocess files exist
    try:
        # check files
        check_files = ["classifier_v0.pkl", "classifier_v1.pkl", "classifier_v2.pkl", "classifier_v3.pkl"
                    , "preprocess_pipe_scaled.pkl", "preprocess_pipe_unscaled.pkl"]
                    
        for m in check_files:
            # check preprocess files
            file_path = os.path.join(settings.model_folder, m)
            check_file = os.path.isfile(file_path) # check if the file exists 

            if not check_file:
                # error 500
                return JSONResponse(status_code=503, content={"status": "unhealthy", "message": f"File {m} can't be loaded"})

        return JSONResponse(status_code=200, content={"status": "healthy"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": e})

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

    # prepare features to be passed to the classifier based on the model version
    if model_version.lower() == "v0":
        preprocess_pipe = CreditRisk_Preprocess_Predict(use_scaler=True) # load preprocess pipe
        # do a scaling on numerical inputs
        X_input = preprocess_pipe.prepare_input_features(df_input)
    else:
        preprocess_pipe = CreditRisk_Preprocess_Predict(use_scaler=False) # load preprocess pipe
        # no scaling on the numerical inputs
        X_input = preprocess_pipe.prepare_input_features(df_input)

    # init classifier object
    classifier = CreditRisk_Classifier(model_version)

    # load the model saved as a pickle. load model based on the version choosed
    loaded_model = classifier.load_model()

    # get the class and probability using the model
    prediction = loaded_model.predict(X_input)
    probability = loaded_model.predict_proba(data_in).max()

    # prediction message
    if prediction[0] == 0:
        message_predict = "good"
    else:
        message_predict = "bad"

    return JSONResponse(status_code=200, content={"prediction": message_predict, "probability": probability})


# model performances
@app.get("/model_performances/{model_version}")
async def model_performances(model_version):
    
    # check the value of the version 
    if model_version.lower() not in ["v0", "v1", "v2"]:
        raise ValueError("model_version equal to 'v0', 'v1' or 'v2'")

    # prepare features to be passed to the classifier based on the model version
    if model_version.lower() == "v0":
        preprocess_pipe = CreditRisk_Preprocess_Predict(use_scaler=True) # load preprocess pipe
        # test data
        df_test = preprocess_pipe.load_test_data()
        # do a scaling on numerical inputs
        X_input = preprocess_pipe.prepare_input_features(df_test)
        y_test = df_test['class'].to_numpy()
    else:
        preprocess_pipe = CreditRisk_Preprocess_Predict(use_scaler=False) # load preprocess pipe
        # test data
        df_test = preprocess_pipe.load_test_data()
        # no scaling on the numerical inputs
        X_input = preprocess_pipe.prepare_input_features(df_test)
        y_test = df_test['class'].to_numpy()

    # load model 
    model = CreditRisk_Classifier(model_version).load_model()
    
    # get the predictions 
    y_predicts = model.predict(X_input)

    # perf on test data 
    accuracy = accuracy_score(y_test, y_predicts)
    precision = precision_score(y_test, y_predicts)
    recall = recall_score(y_test, y_predicts)

    return JSONResponse(status_code=200, content={'model_version': model_version,
                                                    'model_name': model.model_name[model_version],
                                                    'accuracy': accuracy,
                                                    'precision': precision,
                                                    'recall': recall
                                                    })

if __name__ == '__main__':
    uvicorn.run(app, workers=1, host=settings.host, port=settings.port)