# import pickle
# import numpy as np
# import pandas as pd
# from io import StringIO
# from pydantic import BaseModel
# from fastapi import FastAPI, File, Form, UploadFile

# app = FastAPI()
# # app = FastAPI(title='SMART Data Science Application',
# #               description='A Smart Data Science Application running on FastAPI + uvicorn',
# #               version='0.0.1')


# class ClientProfile(BaseModel):
#     checking_status: str
#     duration: int
#     credit_history: str
#     purpose: str
#     credit_amount: float
#     savings_status: str
#     employment: str
#     installment_commitment: int
#     personal_status: str
#     other_parties: str
#     residence_since: int
#     property_magnitude: str
#     age: int
#     other_payment_plans: str
#     housing: str
#     existing_credits: int
#     job: str
#     num_dependents: int
#     own_telephone: str
#     foreign_worker: str


# @app.post("/predict")
# async def predict_risk(client: ClientProfile):
#     data = client.dict()

#     # load the model saved as a pickle
#     loaded_model = pickle.load(open("classifier.pkl", "rb"))

#     # store the list of list
#     data_in = [
#         [
#             data["checking_status"],
#             data["duration"],
#             data["credit_history"],
#             data["purpose"],
#             data["credit_amount"],
#             data["savings_status"],
#             data["employment"],
#             data["installment_commitment"],
#             data["personal_status"],
#             data["other_parties"],
#             data["residence_since"],
#             data["property_magnitude"],
#             data["age"],
#             data["other_payment_plans"],
#             data["housing"],
#             data["existing_credits"],
#             data["job"],
#             data["num_dependents"],
#             data["own_telephone"],
#             data["foreign_worker"],
#         ]
#     ]

#     # preprocess the entry

#     # get the class and probability using the model
#     prediction = loaded_model.predict(data_in)
#     probability = loaded_model.predict_proba(data_in).max()

#     return {"prediction": prediction[0], "probability": probability}


# # # endpoint to process a file as the input
# # @app.post("/files/")
# # async def create_file(file: bytes = File(...), token: str = Form(...)):
# #     s=str(file,'utf-8')
# #     data = StringIO(s)
# #     df=pd.read_csv(data)
# #     print(df)

# #     #return df
# #     return {
# #         "file": df,
# #         "token": token,
# #     }
