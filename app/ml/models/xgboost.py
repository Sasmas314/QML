import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostModel:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1
        )

    def _create_dataset(self, series):
        X, y = [], []

        for i in range(len(series) - self.window_size):
            X.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size])

        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame):
        series = df["close"].values

        X, y = self._create_dataset(series)

        if len(X) == 0:
            return False

        self.model.fit(X, y)
        return True

    def predict(self, df: pd.DataFrame, steps=10):
        series = df["close"].values

        if len(series) < self.window_size:
            return None

        self.fit(df)

        input_seq = series[-self.window_size:].tolist()

        predictions = []

        for _ in range(steps):
            pred = self.model.predict([input_seq])[0]
            predictions.append(pred)

            input_seq.pop(0)
            input_seq.append(pred)

        last_ts = df["timestamp"].iloc[-1]
        step_size = df["timestamp"].diff().median()

        future_ts = [
            int(last_ts + step_size * (i + 1))
            for i in range(steps)
        ]

        return pd.DataFrame({
            "timestamp": future_ts,
            "close": predictions
        })