import pandas as pd
from .config import DATA_DIR, DATE_COL, TARGET_COL, KEY_COLS, TEST_START

def load_raw():
    train = pd.read_csv(DATA_DIR / "train.csv")        # Store, Dept, Date, Weekly_Sales, IsHoliday
    features = pd.read_csv(DATA_DIR / "features.csv")  # Store, Date, Temperature, Fuel_Price, CPI, Unemployment, IsHoliday
    stores = pd.read_csv(DATA_DIR / "stores.csv")      # Store, Type, Size
    return train, features, stores

def build_base_frame():     #Loads and Merges data
    train, features, stores = load_raw()

    # dates
    for df in (train, features):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # merge features & store info
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    # sort & basic cleanup
    df = df.sort_values([*KEY_COLS, DATE_COL]).reset_index(drop=True)

    # Forward fill exogenous where reasonable
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]:
        if col in df.columns:
            df[col] = df.groupby(KEY_COLS)[col].transform(lambda x: x.ffill().bfill())

    # cast helpful dtypes
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df

def train_test_split(df):
    train_df = df[df[DATE_COL] < TEST_START].copy()
    test_df  = df[df[DATE_COL] >= TEST_START].copy()
    return train_df, test_df

def add_time_features(df):
    df = df.copy()
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["month"] = df[DATE_COL].dt.month
    df["year"] = df[DATE_COL].dt.year
    df["dow"] = df[DATE_COL].dt.weekday
    return df

#lags - target value from previous time steps, 
def make_lags(df, target=TARGET_COL, lags=(1,2,4), rolls=(4,8,12)):
    df = df.copy().sort_values(DATE_COL)
    for l in lags:
        df[f"lag_{l}"] = df.groupby(KEY_COLS)[target].shift(l)
    for r in rolls:
        df[f"rollmean_{r}"] = df.groupby(KEY_COLS)[target].shift(1).rolling(r).mean()
    return df
