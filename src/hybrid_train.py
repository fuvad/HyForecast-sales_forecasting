import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from joblib import dump
from .config import DATE_COL, TARGET_COL, KEY_COLS, FORECASTS_DIR, METRICS_DIR
from .utils import ensure_dirs, mape
from .data_prep import build_base_frame, train_test_split, add_time_features, make_lags
from .prophet_model import fit_prophet, prophet_forecast, save_prophet
from .xgb_model import fit_xgb, save_xgb

def train_one_group(df_g):
    # split
    train_g, test_g = train_test_split(df_g)
    
        # ---------- Prophet base ----------
    prop_model = fit_prophet(train_g)
    base_fc = prophet_forecast(prop_model, test_g)  # forecast on TEST dates
    
    # merge base forecast back
    test_merged = test_g.merge(base_fc, on=DATE_COL, how="left")        #Compare Prophet’s predictions vs. the actual sales in test_g
    
    # residuals on TRAIN to train XGB (fit on train period with backcast)
    base_train = prophet_forecast(prop_model, train_g)
    train_with_base = train_g.merge(base_train, on=DATE_COL, how="left")
    train_with_base["residual"] = train_with_base[TARGET_COL] - train_with_base["yhat"]     #Residuals is the target variable for XGBoost
    
    # ---------- Features for XGB ----------
    # Build full frame to get lags (on all data)
    df_lag = make_lags(add_time_features(df_g))
    
    # TRAIN feature set
    train_feat = df_lag[df_lag[DATE_COL].isin(train_with_base[DATE_COL])]      #filters df_lag to keep only rows whose dates are in the training period
    train_feat = train_feat.merge(train_with_base[[DATE_COL, "residual"]], on=DATE_COL, how="left").dropna(subset=["residual"])     #Add the residuals
    
    # TEST feature set
    test_feat = df_lag[df_lag[DATE_COL].isin(test_g[DATE_COL])]     #filters df_lag to keep only rows whose dates are in the training period
    
    #All feature columns to use for XGBoost
    feat_cols = [
        "IsHoliday","Temperature","Fuel_Price","CPI","Unemployment","Size",
        "weekofyear","month","dow",
        "lag_1","lag_2","lag_4","rollmean_4","rollmean_8","rollmean_12"
    ]
    #Select only if the column is present
    feat_cols = [c for c in feat_cols if c in train_feat.columns]
    
    X_train = train_feat[feat_cols]
    y_train = train_feat["residual"]
    X_test  = test_feat[feat_cols]
    
    # fit XGB on residuals
    xgb_model = fit_xgb(X_train, y_train)
    res_pred  = xgb_model.predict(X_test)       #Predicts the residuals for future period
    
    # ---------- Hybrid output ----------
    test_merged["residual_pred"] = res_pred
    test_merged["yhat_hybrid"] = test_merged["yhat"] + test_merged["residual_pred"]     #Adds XGBoost’s correction to Prophet’s forecast
       
    # metrics
    mae_prophet = mean_absolute_error(test_merged[TARGET_COL], test_merged["yhat"])
    rmse_prophet = mean_squared_error(test_merged[TARGET_COL], test_merged["yhat"], squared=False)
    mape_prophet = mape(test_merged[TARGET_COL], test_merged["yhat"])

    mae_hybrid = mean_absolute_error(test_merged[TARGET_COL], test_merged["yhat_hybrid"])
    rmse_hybrid = mean_squared_error(test_merged[TARGET_COL], test_merged["yhat_hybrid"], squared=False)
    mape_hybrid = mape(test_merged[TARGET_COL], test_merged["yhat_hybrid"])
    
    return prop_model, xgb_model, test_merged, {
        "mae_prophet": mae_prophet, "rmse_prophet": rmse_prophet, "mape_prophet": mape_prophet,
        "mae_hybrid": mae_hybrid, "rmse_hybrid": rmse_hybrid, "mape_hybrid": mape_hybrid,
    }
    
    
def main(limit_groups=10):
    ensure_dirs()
    df = build_base_frame()

    metrics = []
    groups = df.groupby(KEY_COLS)
    
    for (store, dept), df_g in groups:
        # optional: limit number of groups while testing
        if limit_groups and len(metrics) >= limit_groups:
            break
        
        print(f"Training Store {store}, Dept {dept} ...")
        prop_model, xgb_model, test_out, metr = train_one_group(df_g)
        
        # save models & outputs
        save_prophet(prop_model, store, dept)
        save_xgb(xgb_model, store, dept)
        
        test_out.assign(Store=store, Dept=dept).to_csv(     #Just add cols
            FORECASTS_DIR / f"preds_{store}_{dept}.csv", index=False
        )
        metr_row = {"Store": store, "Dept": dept, **metr}
        metrics.append(metr_row)

    pd.DataFrame(metrics).to_csv(METRICS_DIR / "metrics.csv", index=False)
    print("Done. Metrics saved to outputs/metrics/metrics.csv")
    
if __name__ == "__main__":
    main(limit_groups=10)  # increase/remove to train all groups
