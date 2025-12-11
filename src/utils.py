from pathlib import Path
import pandas as pd
from .config import MODELS_DIR, METRICS_DIR, FORECASTS_DIR, OUT_DIR

def ensure_dirs():      #Makes sure output folders exist
    for p in [OUT_DIR, MODELS_DIR, METRICS_DIR, FORECASTS_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)

def mape(y_true, y_pred):       #Mean absolute percentage error
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mask = y_true != 0
    return (abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100

def date_index(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df
