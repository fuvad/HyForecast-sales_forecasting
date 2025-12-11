import xgboost as xgb
from joblib import dump
from .config import XGB_PARAMS, MODELS_DIR

def fit_xgb(X_train, y_train):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model

def save_xgb(model, store, dept):
    path = MODELS_DIR / f"xgb_{store}_{dept}.json"  #No need joblib as this is more efficient
    model.save_model(str(path))
