import sys
import os

# фикс импорта
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from app.data.pipeline import DataPipeline
from app.ml.models.baseline import BaselineModel
from app.ml.models.xgboost import XGBoostModel


# --- стиль ---
def apply_dark_theme():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0a0f2c, #121a3a);
            color: #e0e6ff;
        }
        </style>
    """, unsafe_allow_html=True)


# --- график ---
def plot_timeseries(df: pd.DataFrame, forecast_df=None):
    fig = go.Figure()

    # реальные данные
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df["timestamp"], unit="ms"),
        y=df["close"],
        mode="lines",
        name="Real",
    ))

    # прогноз
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(forecast_df["timestamp"], unit="ms"),
            y=forecast_df["close"],
            mode="lines",
            name="ML Forecast",
            line=dict(dash="dash")  # пунктир
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0f2c",
        plot_bgcolor="#0a0f2c",
        font=dict(color="#e0e6ff"),
        title="Price + Forecast",
        xaxis_title="Time",
        yaxis_title="Price"
    )

    return fig


def main():
    st.set_page_config(page_title="Stock Predictor", layout="wide")
    apply_dark_theme()

    st.title("📈 Stock Predictor (MOEX)")

    # --- тикеры ---
    popular_stocks = ["SBER", "GAZP", "LKOH", "ROSN", "NVTK"]

    col1, col2 = st.columns([2, 3])

    with col1:
        selected = st.selectbox("Popular stocks", popular_stocks)

    with col2:
        custom = st.text_input("Or enter your ticker")

    ticker = custom.strip().upper() if custom else selected

    st.markdown(f"### Selected: `{ticker}`")

    # --- даты ---
    col3, col4 = st.columns(2)

    with col3:
        start_date = st.date_input(
            "Start date",
            value=datetime.now() - timedelta(days=30)
        )

    with col4:
        end_date = st.date_input(
            "End date",
            value=datetime.now()
        )

    # --- шаг ---
    timeframe_map = {
        "1 day": 24,
        "1 hour": 60,
        "10 min": 10
    }

    timeframe_label = st.selectbox("Timeframe", list(timeframe_map.keys()))
    interval = timeframe_map[timeframe_label]

    # --- горизонт прогноза ---
    steps = st.slider("Forecast steps", 5, 50, 10)

    # --- проверка дат ---
    if start_date > end_date:
        st.error("Start date must be before end date")
        return

    try:
        pipeline = DataPipeline()

        # --- загружаем ровно выбранные данные ---
        df_raw, _, _ = pipeline.load_from_moex(
            ticker=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval
        )

        if df_raw.empty:
            st.warning("No data returned from MOEX")
            return

        st.success(f"Loaded {len(df_raw)} points | {timeframe_label}")

        model_type = st.selectbox("Model", ["Baseline", "XGBoost"])

        # --- ML модель ---
        if model_type == "Baseline":
            model = BaselineModel()
            forecast_df = model.predict(df_raw, steps=steps)

        elif model_type == "XGBoost":
            model = XGBoostModel()
            forecast_df = model.predict(df_raw, steps=steps)

        # --- график ---
        fig = plot_timeseries(df_raw, forecast_df)
        st.plotly_chart(fig, use_container_width=True)

        # --- таблица ---
        with st.expander("Show data"):
            st.dataframe(df_raw)

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()