from .model import train_model, compute_model_metrics, inference
import sklearn
import pickle
import numpy

def test_train_model():
    with open('./data/X_train_sample.pkl', 'rb') as f:
        X_train_sample = pickle.load(f)
    with open('./data/y_train_sample.pkl', 'rb') as f:
        y_train_sample = pickle.load(f)
    
    assert isinstance(train_model(X_train_sample, y_train_sample),sklearn.ensemble._forest.RandomForestClassifier)


def test_compute_model_metrics():
    y = [1,0,0,1,1]
    preds = [1,1,0,0,0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert (precision, recall, fbeta) == (0.5, 1/3, 0.4)

    y = [0,0,1,1,1]
    preds = [1,1,0,0,0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert (precision, recall, fbeta) == (0.0, 0.0, 1.0)

    y = [1,1,0,0,0]
    preds = [1,1,0,0,0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert (precision, recall, fbeta) == (1.0, 1.0, 1.0)


def test_inference():
    with open('./data/X_test_sample.pkl', 'rb') as f:
        X_test_sample = pickle.load(f)
    with open('./model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    preds = inference(model, X_test_sample)
    assert isinstance(preds, numpy.ndarray)
    assert set(preds) == {0,1}