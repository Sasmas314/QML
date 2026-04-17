import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = ["open", "high", "low", "close", "volume"]

        scaled = self.scaler.fit_transform(df[features])

        df_scaled = pd.DataFrame(scaled, columns=features)
        df_scaled["timestamp"] = df["timestamp"].values

        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = ["open", "high", "low", "close", "volume"]

        scaled = self.scaler.transform(df[features])

        df_scaled = pd.DataFrame(scaled, columns=features)
        df_scaled["timestamp"] = df["timestamp"].values

        return df_scaled