from fastapi.testclient import TestClient
import json
# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_get():
    r = client.get("/")
    assert r.json() == {"greeting": "Welcome!"}
    assert r.status_code == 200

def test_api_post_prediction1():
    input_data = json.dumps({
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
                })
    r = client.post("/inference/", data = input_data)
    assert r.status_code == 200
    assert r.json() == {"preds": "0"}

def test_api_post_prediction2():
    input_data = json.dumps({
                    "age": 39, 
                    "workclass": " Private",
                    "fnlgt": 102953,
                    "education": " Bachelors",
                    "education-num": 13,
                    "marital-status":  " Married-civ-spouse",
                    "occupation": " Sales",
                    "relationship": " Husband",
                    "race": " White",
                    "sex": " Male",
                    "capital-gain": 7298,
                    "capital-loss": 0,
                    "hours-per-week": 55,
                    "native-country": " United-States"
                })
    r = client.post("/inference/", data = input_data)
    assert r.status_code == 200
    assert r.json() == {"preds": "1"}