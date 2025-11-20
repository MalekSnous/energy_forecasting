import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def feature_engineer(df):
    X = df.copy()
    X['sin_hour'] = np.sin(2*np.pi*X['hour']/24)
    X['cos_hour'] = np.cos(2*np.pi*X['hour']/24)
    X['sin_doy'] = np.sin(2*np.pi*X['dayofyear']/365)
    X['cos_doy'] = np.cos(2*np.pi*X['dayofyear']/365)
    return X[['temp_C','hour','dayofweek','sin_hour','cos_hour','sin_doy','cos_doy']]

def train_rf(X, y):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        'mae': mean_absolute_error(y_test, preds),
        'rmse': mean_squared_error(y_test, preds, squared=False)
    }

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
