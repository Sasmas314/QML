import pandas as pd


class DataValidator:

    @staticmethod
    def validate(df: pd.DataFrame):
        DataValidator._check_columns(df)
        DataValidator._check_prices(df)
        DataValidator._check_volume(df)

    @staticmethod
    def _check_columns(df):
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    @staticmethod
    def _check_prices(df):
        if not ((df["low"] <= df["open"]) & (df["low"] <= df["close"])).all():
            raise ValueError("Invalid low values")

        if not ((df["high"] >= df["open"]) & (df["high"] >= df["close"])).all():
            raise ValueError("Invalid high values")

    @staticmethod
    def _check_volume(df):
        if (df["volume"] < 0).any():
            raise ValueError("Negative volume detected")