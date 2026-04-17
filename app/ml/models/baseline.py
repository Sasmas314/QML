import numpy as np
import pandas as pd


class BaselineModel:
    def __init__(self, window_size=30):
        self.window_size = window_size

    def predict(self, df: pd.DataFrame, steps=10):
        if len(df) < self.window_size:
            return None

        y = df["close"].values[-self.window_size:]
        x = np.arange(len(y))

        # линейная регрессия
        coef = np.polyfit(x, y, 1)

        future_x = np.arange(len(y), len(y) + steps)
        forecast = coef[0] * future_x + coef[1]

        last_ts = df["timestamp"].iloc[-1]
        step_size = df["timestamp"].diff().median()

        future_ts = [
            int(last_ts + step_size * (i + 1))
            for i in range(steps)
        ]

        return pd.DataFrame({
            "timestamp": future_ts,
            "close": forecast
        })
