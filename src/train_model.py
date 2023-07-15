# Script to train machine learning model.
# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

label = "salary"

def evaluate_performance_on_data_slice(model, df, feature_list, encoder, lb):
    """
    Computes the performance metrics when the value of a given feature is held fixed.

    Inputs
    ------
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    df : pd.DataFrame
        Data used for evaluation.
    feature_list : Given feature list whose value is fixed.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.

    Returns
    -------
    metrics_slice_df : pd.DataFrame
        Dataframe containing metrics for each slice of data.
    """

    metrics_dict_list = []
    for category in feature_list:
        for value in df[category].unique():
            df_temp = df[df[category] == value]
            X_temp, y_temp, encoder, lb = process_data(
        df_temp, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
    )
            preds_temp = inference(model, X_temp)
            precision, recall, fbeta = compute_model_metrics(y_temp, preds_temp)
            metrics_dict = {"slice": f"{category}_{value}",
                            "precision": precision,
                            "recall": recall,
                            "fbeta": fbeta}
            metrics_dict_list.append(metrics_dict)
    
    metrics_slice_df = pd.DataFrame(metrics_dict_list)
    return metrics_slice_df



# Add code to load in the data.
data = pd.read_csv('./data/census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
with open('./model/model.pkl','wb') as f:
    pickle.dump(model,f)
with open('./model/encoder.pkl','wb') as f:
    pickle.dump(encoder,f)

# Evaluation on test set
pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, pred)
print(f"precision = {precision}, recall = {recall}, fbeta = {fbeta}")

# Evaluation on data slices
metrics_slice_df = evaluate_performance_on_data_slice(model, test, cat_features, encoder, lb)
metrics_slice_df.to_csv('./data/slice_output.txt', sep='\t', index=False)
