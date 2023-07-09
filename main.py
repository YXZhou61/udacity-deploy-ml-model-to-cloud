# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import pickle
import json
import pandas as pd
import numpy as np

# Instantiate the app.
app = FastAPI()

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

def replace_hyphen(string: str) -> str:
    return string.replace('_', '-')

# define prediction base model
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    model_config = ConfigDict(
        alias_generator = replace_hyphen,
        json_schema_extra = {
            "examples": [
                {
                    "age": 39, 
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        },
        populate_by_name = True
    )
        






# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome!"}


@app.post("/inference/")
async def inference(data: InputData):
    with open('./model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    X = pd.DataFrame.from_dict(json.loads(data.model_dump_json()),orient='index').T
    X_categorical = X[cat_features].values
    X_continuous = X.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    preds = model.predict(X)[0]

    return {"preds":str(preds)}