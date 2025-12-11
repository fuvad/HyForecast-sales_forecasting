import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys, os
import warnings

warnings.filterwarnings("ignore")

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import FORECASTS_DIR, METRICS_DIR
from src.hybrid_infer import forecast_future

# --- Page Config ---
st.set_page_config(page_title="Hybrid Sales Forecasting", page_icon="📈", layout="wide")

# --- Header ---
st.title("📈 Hybrid Sales Forecasting (Prophet + XGBoost)")
st.markdown(
    "Analyze **historical performance** and **forecast future weeks** using a hybrid time series model combining Prophet and XGBoost."
)

# --- Load forecast files ---
pred_files = sorted(Path(FORECASTS_DIR).glob("preds_*.csv"))
if not pred_files:
    st.warning("No forecasts found. Run `python -m src.hybrid_train` first.")
    st.stop()

names = [p.stem.replace("preds_", "") for p in pred_files]
choice = st.selectbox("Select Store–Dept:", names)

# --- Tabs for Historical & Future Forecast ---
tab1, tab2 = st.tabs(["📊 Historical Performance", "🔮 Future Forecast"])

# =================================================================
# TAB 1: HISTORICAL PERFORMANCE
# =================================================================
with tab1:
    try:
        df = pd.read_csv(FORECASTS_DIR / f"preds_{choice}.csv")
        st.caption(f"Loaded predictions for **Store_Dept:** `{choice}`")
    except Exception as e:
        st.error(f"Failed to load forecast file: {e}")
        st.stop()

    # --- Toggle between Prophet & Hybrid ---
    show = st.radio("Show forecast type:", ["Hybrid vs Actual", "Prophet vs Actual"], horizontal=True)

    df["Date"] = pd.to_datetime(df["Date"])
    plot_cols = ["Weekly_Sales", "yhat_hybrid" if show == "Hybrid vs Actual" else "yhat"]

    # --- Plotly chart ---
    fig = px.line(
        df,
        x="Date",
        y=plot_cols,
        labels={"value": "Sales", "variable": ""},
        title=f"{show} for Store_Dept {choice}",
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title_text="",
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics Section ---
    st.markdown("### 📊 Performance Metrics")

    try:
        met = pd.read_csv(METRICS_DIR / "metrics.csv")

        # Ensure proper column types
        met["Store"] = met["Store"].astype(int)
        met["Dept"] = met["Dept"].astype(int)

        store, dept = map(int, choice.split("_"))
        met_row = met[(met["Store"] == store) & (met["Dept"] == dept)]

        if met_row.empty:
            st.info("Metrics not found for this group.")
        else:
            m = met_row.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE (Hybrid)", f"{m['mae_hybrid']:.2f}", f"{(m['mae_prophet'] - m['mae_hybrid']):+.2f} Δ vs Prophet")
            col2.metric("RMSE (Hybrid)", f"{m['rmse_hybrid']:.2f}", f"{(m['rmse_prophet'] - m['rmse_hybrid']):+.2f} Δ")
            col3.metric("MAPE (Hybrid %)", f"{m['mape_hybrid']:.2f}", f"{(m['mape_prophet'] - m['mape_hybrid']):+.2f} Δ")

            # --- Full Metrics Table (Direct Display) ---
            st.markdown("#### 📋 Full Metrics")

            met_display = met_row.reset_index(drop=True).copy()
            for c in met_display.columns:
                if met_display[c].dtype != "object":
                    met_display[c] = pd.to_numeric(met_display[c], errors="coerce").round(3)

            # st.dataframe(met_display, use_container_width=True, hide_index=True)

            # --- Download Button ---
            csv = met_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Metrics CSV",
                data=csv,
                file_name=f"metrics_{choice}.csv",
                mime="text/csv",
            )

    except FileNotFoundError:
        st.warning("Metrics file not found — train models first.")
    except Exception as e:
        st.error(f"Error reading metrics: {e}")

# =================================================================
# TAB 2: FUTURE FORECAST
# =================================================================
with tab2:
    st.subheader("🔮 Forecast Future Weeks")
    st.markdown("Generate forecasts for upcoming weeks using your trained hybrid model.")

    periods = st.slider("Select forecast horizon (weeks)", 4, 24, 12)
    store, dept = map(int, choice.split("_"))

    if st.button("Generate Forecast"):
        try:
            with st.spinner(f"Forecasting next {periods} weeks..."):
                fc = forecast_future(store, dept, periods=periods)

            # Validate forecast output
            if fc is None or fc.empty:
                st.warning("No forecast data returned. Check if model files exist and data schema matches.")
            else:
                fc["Date"] = pd.to_datetime(fc["Date"])

                # --- Forecast Chart ---
                fig2 = px.line(
                    fc,
                    x="Date",
                    y=["yhat", "yhat_hybrid"],
                    labels={"value": "Predicted Sales", "variable": ""},
                    title=f"Next {periods} Weeks Forecast (Prophet vs Hybrid)",
                )
                fig2.update_traces(line=dict(width=2))
                fig2.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)

                # --- Forecast Data (Direct Display) ---
                st.markdown("#### 📊 Forecast Data")

                fc_display = fc.reset_index(drop=True).copy()
                for c in fc_display.columns:
                    if fc_display[c].dtype != "object":
                        fc_display[c] = pd.to_numeric(fc_display[c], errors="coerce").round(3)

                # st.dataframe(fc_display, use_container_width=True, hide_index=True)

                # --- Download Forecast ---
                csv = fc_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Forecast CSV",
                    data=csv,
                    file_name=f"forecast_{choice}.csv",
                    mime="text/csv",
                )

        except FileNotFoundError:
            st.error("Model files not found. Please train models first using `python -m src.hybrid_train`.")
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
