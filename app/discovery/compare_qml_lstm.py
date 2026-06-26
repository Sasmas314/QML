import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.data.pipeline import DataPipeline
from app.ml.models.lstm_model import LSTMModel
from app.qml.models.qml_model import QMLModel


# =========================
# 🔹 МЕТРИКИ
# =========================
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return mae, rmse, mape


# =========================
# 🔹 ЗАГРУЗКА ДАННЫХ
# =========================
def load_data():
    pipeline = DataPipeline()

    end = datetime.now()
    start = end - timedelta(days=30)

    df, _, _ = pipeline.load_from_moex(
        ticker="SBER",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=60  # 1 hour
    )

    return df


# =========================
# 🔹 MAIN
# =========================
def run_experiment():
    df = load_data()

    print(f"Loaded {len(df)} points")

    # split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    steps = len(test_df)

    # =========================
    # 🔹 МОДЕЛИ (РАВНЫЕ УСЛОВИЯ)
    # =========================

    # LSTM с hidden_size = 4
    lstm = LSTMModel(window_size=10, epochs=30)
    lstm.model.lstm.hidden_size = 8  # если нужно, можно задать в классе

    # QML (оптимальная конфигурация)
    qml = QMLModel(
        window_size=10,
        epochs=30,
        n_qubits=6,
        n_layers=3,
        hidden_size=4
    )

    print("Training LSTM...")
    lstm.fit(train_df)

    print("Training QML...")
    qml.fit(train_df)

    # =========================
    # 🔹 ПРЕДСКАЗАНИЕ
    # =========================
    lstm_pred = lstm.predict(train_df, steps=steps)
    qml_pred = qml.predict(train_df, steps=steps)

    y_true = test_df["close"].values
    y_lstm = lstm_pred["close"].values[:len(y_true)]
    y_qml = qml_pred["close"].values[:len(y_true)]

    # =========================
    # 🔹 МЕТРИКИ
    # =========================
    lstm_metrics = calculate_metrics(y_true, y_lstm)
    qml_metrics = calculate_metrics(y_true, y_qml)

    # =========================
    # 🔹 ТАБЛИЦА
    # =========================
    results = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE"],
        "LSTM": lstm_metrics,
        "QML": qml_metrics
    })

    print("\n📊 Comparison Results:")
    print(results)

    return results


if __name__ == "__main__":
    df = run_experiment()

    df.to_csv("comparison_results.csv", index=False)
    print("\nSaved to comparison_results.csv")