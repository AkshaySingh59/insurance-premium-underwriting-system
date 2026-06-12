import shap
import pandas as pd

def get_shap_values(model, input_df):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_df)

    return shap_values


import joblib

rf_model = joblib.load(
    "models/randomforest.pkl"
)