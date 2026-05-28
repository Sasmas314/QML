import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from app.data.pipeline import DataPipeline
from app.ml.models.lstm_model import LSTMModel
from app.qml.models.qml_model import QMLModel


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


# --- метрики ---
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return mae, rmse, mape


# --- подсветка таблицы ---
def style_table(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for i in df.index:
        if df.loc[i, "LSTM"] < df.loc[i, "QML"]:
            styles.loc[i, "LSTM"] = "background-color: #144d14; color: white;"
            styles.loc[i, "QML"] = "background-color: #5c1a1a; color: white;"
        else:
            styles.loc[i, "LSTM"] = "background-color: #5c1a1a; color: white;"
            styles.loc[i, "QML"] = "background-color: #144d14; color: white;"

    return df.style.apply(lambda _: styles, axis=None)


# --- график ---
def plot_timeseries(df, lstm_df=None, qml_df=None):
    fig = go.Figure()

    time = pd.to_datetime(df["timestamp"], unit="ms")
    price = df["close"]

    fig.add_trace(go.Scatter(
        x=time,
        y=price,
        mode="lines",
        name="Real",
        line=dict(color="#4facfe", width=2)
    ))

    last_time = time.iloc[-1]
    last_price = price.iloc[-1]

    def add_forecast(forecast_df, name, color, dash):
        x = pd.to_datetime(forecast_df["timestamp"], unit="ms")
        y = forecast_df["close"]

        x = pd.concat([pd.Series([last_time]), x])
        y = pd.concat([pd.Series([last_price]), y])

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, dash=dash)
        ))

    if lstm_df is not None:
        add_forecast(lstm_df, "LSTM", "orange", "dash")

    if qml_df is not None:
        add_forecast(qml_df, "QML", "cyan", "dot")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0f2c",
        plot_bgcolor="#0a0f2c",
        font=dict(color="#e0e6ff"),
        title="Forecast vs Real"
    )

    return fig


def main():
    st.set_page_config(page_title="Stock Predictor", layout="wide")
    apply_dark_theme()

    st.title("Stock Predictor (MOEX)")

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

    # --- split ---
    train_percent = st.slider("Train/Test split (%)", 30, 90, 80)

    # --- прогноз ---
    forecast_percent = st.slider("Forecast horizon (%)", 5, 50, 20)

    st.info(f"Train: {train_percent}% | Forecast: {forecast_percent}%")

    try:
        pipeline = DataPipeline()

        df_raw, _, _ = pipeline.load_from_moex(
            ticker=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval
        )

        if df_raw.empty:
            st.warning("No data")
            return

        st.success(f"Loaded {len(df_raw)} points")

        # --- split ---
        split = int(len(df_raw) * (train_percent / 100))
        train_df = df_raw.iloc[:split]
        test_df = df_raw.iloc[split:]

        steps_test = len(test_df)

        # --- модели ---
        lstm_model = LSTMModel(window_size=10, epochs=30)
        qml_model = QMLModel(window_size=10, epochs=20)

        # ✅ ГЛАВНОЕ: сначала обучаем
        lstm_model.fit(train_df)
        qml_model.fit(train_df)

        # потом предсказываем
        lstm_pred = lstm_model.predict(train_df, steps=steps_test)
        qml_pred = qml_model.predict(train_df, steps=steps_test)

        y_true = test_df["close"].values
        y_lstm = lstm_pred["close"].values[:len(y_true)]
        y_qml = qml_pred["close"].values[:len(y_true)]

        lstm_metrics = calculate_metrics(y_true, y_lstm)
        qml_metrics = calculate_metrics(y_true, y_qml)

        # --- таблица ---
        st.markdown("### Model Comparison")

        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "MAPE"],
            "LSTM": lstm_metrics,
            "QML": qml_metrics
        })

        st.dataframe(style_table(metrics_df), width="stretch")

        best = "LSTM" if lstm_metrics[1] < qml_metrics[1] else "QML"
        st.success(f"Best model: {best}")

        # --- прогноз ---
        forecast_steps = int(len(df_raw) * (forecast_percent / 100))

        lstm_df = lstm_model.predict(df_raw, steps=forecast_steps)
        qml_df = qml_model.predict(df_raw, steps=forecast_steps)

        fig = plot_timeseries(df_raw, lstm_df, qml_df)
        st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()