import requests
import pandas as pd


class MOEXDataLoader:
    BASE_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities"

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()

    def load(self, start: str, end: str, interval: int = 24) -> pd.DataFrame:
        url = f"{self.BASE_URL}/{self.ticker}/candles.json"

        params = {
            "from": start,
            "till": end,
            "interval": interval
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception(f"MOEX API error: {response.status_code}")

        data = response.json()

        candles = data["candles"]["data"]
        columns = data["candles"]["columns"]

        if not candles:
            raise ValueError("No data returned (wrong ticker or dates)")

        df = pd.DataFrame(candles, columns=columns)

        return self._transform(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={
            "begin": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64") // 10**6

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        df = df.dropna()
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df