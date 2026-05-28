import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml


# =========================
# 🔹 QNODE FACTORY
# =========================
def create_qnode(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    return qlayer


# =========================
# 🔹 QUANTUM RNN CELL
# =========================
class QuantumRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, n_layers):
        super().__init__()

        self.hidden_size = hidden_size

        self.linear_in = nn.Linear(input_size + hidden_size, n_qubits)

        self.q_layer = create_qnode(n_qubits, n_layers)

        self.linear_out = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)

        q_in = torch.tanh(self.linear_in(combined))

        q_out = self.q_layer(q_in)

        h_new = torch.tanh(self.linear_out(q_out))

        return h_new


# =========================
# 🔹 QML RNN MODEL
# =========================
class QRecurrentRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, n_layers):
        super().__init__()

        self.hidden_size = hidden_size

        self.cell = QuantumRNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            n_layers=n_layers
        )

        self.post = nn.Linear(hidden_size, 1)

    def forward(self, x_seq):
        batch_size = x_seq.shape[0]
        seq_len = x_seq.shape[1]

        h = torch.zeros(batch_size, self.hidden_size, device=x_seq.device)

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            h = self.cell(x_t, h)

        return self.post(h)


# =========================
# 🔹 DATASET
# =========================
def make_sequences(series, window_size):
    X, y = [], []

    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return X, y


# =========================
# 🔹 WRAPPER
# =========================
class QMLModel:
    def __init__(self,
                 window_size=10,
                 epochs=50,
                 n_qubits=4,
                 n_layers=3,
                 hidden_size=4):

        self.window_size = window_size
        self.epochs = epochs

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.model = QRecurrentRegressor(
            input_size=1,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            n_layers=n_layers
        )

        self.min = None
        self.max = None

    # =========================
    # 🔹 NORMALIZATION
    # =========================
    def _normalize(self, data):
        self.min = data.min()
        self.max = data.max()
        return (data - self.min) / (self.max - self.min + 1e-8)

    def _denormalize(self, data):
        return data * (self.max - self.min + 1e-8) + self.min

    # =========================
    # 🔹 TRAIN
    # =========================
    def fit(self, df):
        data = df["close"].values

        data = self._normalize(data)

        X, y = make_sequences(data, self.window_size)

        if len(X) == 0:
            return False

        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()

            pred = self.model(X)

            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

        return True

    # =========================
    # 🔹 PREDICT
    # =========================
    def predict(self, df, steps=10):
        data = df["close"].values

        data = self._normalize(data)

        if len(data) < self.window_size:
            return None

        input_seq = list(data[-self.window_size:])
        preds = []

        for _ in range(steps):
            x = torch.tensor([input_seq], dtype=torch.float32).unsqueeze(-1)

            pred = self.model(x).item()

            preds.append(pred)

            input_seq.pop(0)
            input_seq.append(pred)

        preds = self._denormalize(np.array(preds))

        # timestamps
        last_ts = df["timestamp"].iloc[-1]
        step = df["timestamp"].diff().median()

        future_ts = [
            int(last_ts + step * (i + 1))
            for i in range(steps)
        ]

        return pd.DataFrame({
            "timestamp": future_ts,
            "close": preds
        })