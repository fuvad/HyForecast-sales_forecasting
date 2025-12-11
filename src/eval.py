import pandas as pd
from .config import METRICS_DIR

def main():
    df = pd.read_csv(METRICS_DIR / "metrics.csv")
    agg = df.agg({
        "mae_prophet":"mean","rmse_prophet":"mean","mape_prophet":"mean",
        "mae_hybrid":"mean","rmse_hybrid":"mean","mape_hybrid":"mean"
    }).to_frame("mean").round(3)        #turns the Series into a one-column DataFrame and names that column "mean"
    print("Average metrics across trained groups:")
    print(agg)

if __name__ == "__main__":
    main()