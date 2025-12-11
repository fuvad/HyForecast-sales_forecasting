import pandas as pd
import numpy as np
from joblib import load
from .config import DATE_COL, KEY_COLS, FREQ
from .utils import ensure_dirs
from .data_prep import build_base_frame, add_time_features, make_lags
from .prophet_model import load_prophet, prophet_forecast
import xgboost as xgb
from pathlib import Path
from .config import MODELS_DIR

def load_xgb(store, dept):
    path = MODELS_DIR / f"xgb_{store}_{dept}.json"
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model

def forecast_future(store, dept, periods=12):
    df = build_base_frame()
    df_g = df[(df["Store"]==store)&(df["Dept"]==dept)].copy()

    # 1) Prophet future
    prop_model = load_prophet(store, dept)
    last_date = df_g[DATE_COL].max()
    future_dates = pd.date_range(last_date, periods=periods+1, freq=FREQ, inclusive="right")
    future = pd.DataFrame({DATE_COL: future_dates})
    # You may carry IsHoliday using features.csv for future; here assume 0  -> Not for production use
    future["IsHoliday"] = 0
    base_fc = prophet_forecast(prop_model, future)
    
    # 2) XGB features for future residuals (build lags from historical)
    df_feats = add_time_features(df_g)
    df_feats = make_lags(df_feats)
    last_rows = df_feats[df_feats[DATE_COL] == last_date]
    
    # roll forward simple lag features (naive approach for demo; in prod you’d simulate)    -> not for production use
    fut = future.copy()
    fut["weekofyear"] = fut[DATE_COL].dt.isocalendar().week.astype(int)
    fut["month"] = fut[DATE_COL].dt.month
    fut["dow"] = fut[DATE_COL].dt.weekday
    for col in ["Temperature","Fuel_Price","CPI","Unemployment","Size"]:
        fut[col] = last_rows[col].values[0] if col in last_rows.columns else 0
        
    # simplistic lag bootstrap: use last known lags/rolls   -> not for production use
    for col in ["lag_1","lag_2","lag_4","rollmean_4","rollmean_8","rollmean_12"]:
        if col in last_rows.columns:
            fut[col] = last_rows[col].values[0]
        else:
            fut[col] = np.nan
            
    feat_cols = [c for c in ["IsHoliday","Temperature","Fuel_Price","CPI","Unemployment","Size",
                             "weekofyear","month","dow",
                             "lag_1","lag_2","lag_4","rollmean_4","rollmean_8","rollmean_12"]
                 if c in fut.columns]

    xgb_model = load_xgb(store, dept)
    residual_pred = xgb_model.predict(fut[feat_cols])
    
    out = base_fc.copy()
    out["residual_pred"] = residual_pred
    out["yhat_hybrid"] = out["yhat"] + out["residual_pred"]
    return out

if __name__ == "__main__":
    ensure_dirs()
    fc = forecast_future(store=1, dept=1, periods=12)
    print(fc.head())