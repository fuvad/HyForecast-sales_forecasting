import pandas as pd
from prophet import Prophet
from joblib import dump, load
from .config import PROPHET_PARAMS, MODELS_DIR, DATE_COL, TARGET_COL

def fit_prophet(group_df):
    dfp = group_df[[DATE_COL, TARGET_COL, "IsHoliday"]].rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
    m = Prophet(**PROPHET_PARAMS)
    m.add_regressor("IsHoliday")  # simple regressor
    m.fit(dfp)
    return m

def prophet_forecast(model, df_future):
    fut = df_future[[DATE_COL, "IsHoliday"]].rename(columns={DATE_COL: "ds"})
    fc = model.predict(fut)
    fc = fc[["ds", "yhat", "trend", "weekly", "yearly"]]
    fc = fc.rename(columns={"ds": DATE_COL})
    return fc

def save_prophet(model, store, dept):
    path = MODELS_DIR / f"prophet_{store}_{dept}.pkl"
    dump(model, path)

def load_prophet(store, dept):
    path = MODELS_DIR / f"prophet_{store}_{dept}.pkl"
    return load(path)
