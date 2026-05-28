import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.data.pipeline import DataPipeline
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
# 🔹 GRID SEARCH
# =========================
def run_experiment():
    df = load_data()

    print(f"Loaded {len(df)} points")

    # split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    steps = len(test_df)

    results = []

    # 🔥 ПАРАМЕТРЫ ДЛЯ ИССЛЕДОВАНИЯ
    qubits_list = [2, 4, 6]
    layers_list = [1, 3, 5]
    hidden_list = [2, 4, 8]

    total = len(qubits_list) * len(layers_list) * len(hidden_list)
    counter = 0

    for n_qubits in qubits_list:
        for n_layers in layers_list:
            for hidden_size in hidden_list:

                counter += 1
                print(f"\n[{counter}/{total}] Testing: "
                      f"qubits={n_qubits}, layers={n_layers}, hidden={hidden_size}")

                try:
                    model = QMLModel(
                        window_size=10,
                        epochs=30,
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        hidden_size=hidden_size
                    )

                    model.fit(train_df)
                    pred_df = model.predict(train_df, steps=steps)

                    if pred_df is None:
                        continue

                    y_true = test_df["close"].values
                    y_pred = pred_df["close"].values[:len(y_true)]

                    mae, rmse, mape = calculate_metrics(y_true, y_pred)

                    results.append({
                        "qubits": n_qubits,
                        "layers": n_layers,
                        "hidden": hidden_size,
                        "MAE": mae,
                        "RMSE": rmse,
                        "MAPE": mape
                    })

                except Exception as e:
                    print("Error:", e)

    return pd.DataFrame(results)


# =========================
# 🔹 АНАЛИЗ
# =========================
def analyze(df):
    df = df.sort_values("RMSE")

    print("\n🔥 TOP RESULTS:")
    print(df.head())

    best = df.iloc[0]

    print("\n🏆 BEST CONFIG:")
    print(best)

    return df


# =========================
# 🔹 MAIN
# =========================
if __name__ == "__main__":
    results_df = run_experiment()

    results_df = analyze(results_df)

    # сохраняем
    results_df.to_csv("qml_results.csv", index=False)

    print("\nSaved to qml_results.csv")