from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
MODELS_DIR = OUT_DIR / "models"
METRICS_DIR = OUT_DIR / "metrics"
FORECASTS_DIR = OUT_DIR / "forecasts"

# Columns
DATE_COL = "Date"
TARGET_COL = "Weekly_Sales"
KEY_COLS = ["Store", "Dept"]

# Time settings
FREQ = "W-FRI"  # Walmart dates are weekly Fridays
TEST_START = "2012-01-06"  # keep 2012 as test by default

# Prophet params
PROPHET_PARAMS = dict(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive"
)

# XGBoost params (good starting point)
XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)
