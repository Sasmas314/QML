import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel:
    def __init__(self, window_size=30, epochs=30):
        self.window_size = window_size
        self.epochs = epochs
        self.model = LSTMNet()

        self.min = None
        self.max = None

    # --- нормализация ---
    def _normalize(self, data):
        self.min = data.min()
        self.max = data.max()
        return (data - self.min) / (self.max - self.min + 1e-8)

    def _denormalize(self, data):
        return data * (self.max - self.min + 1e-8) + self.min

    # --- датасет ---
    def _create_dataset(self, data):
        X, y = [], []

        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])

        return np.array(X), np.array(y)

    # --- обучение ---
    def fit(self, df):
        data = df["close"].values.reshape(-1, 1)
        data = self._normalize(data)

        X, y = self._create_dataset(data)

        if len(X) == 0:
            return False

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            self.model.train()

            output = self.model(X)
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return True

    # --- прогноз ---
    def predict(self, df, steps=10):
        data = df["close"].values.reshape(-1, 1)
        data = self._normalize(data)

        if len(data) < self.window_size:
            return None

        input_seq = data[-self.window_size:].tolist()
        preds = []

        for _ in range(steps):
            x = torch.tensor([input_seq], dtype=torch.float32)
            pred = self.model(x).item()

            # 🔥 стабилизация (чтобы не улетал)
            pred = max(min(pred, 1), 0)

            preds.append(pred)

            input_seq.pop(0)
            input_seq.append([pred])

        preds = self._denormalize(np.array(preds))

        # --- временные метки ---
        last_ts = df["timestamp"].iloc[-1]
        step_size = df["timestamp"].diff().median()

        future_ts = [
            int(last_ts + step_size * (i + 1))
            for i in range(steps)
        ]

        return pd.DataFrame({
            "timestamp": future_ts,
            "close": preds
        })