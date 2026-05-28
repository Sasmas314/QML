import requests
import pandas as pd
from datetime import datetime, timedelta


class DataPipeline:
    def load_from_moex(self, ticker: str, start: str, end: str, interval: int):
        # 🔥 адаптация диапазона под частоту
        now = datetime.now()

        if interval <= 10:
            start = (now - timedelta(days=5)).strftime("%Y-%m-%d")

        elif interval <= 60:
            start = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        # иначе оставляем как есть (дневные данные)

        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"

        params = {
            "from": start,
            "till": end,
            "interval": interval
        }

        response = requests.get(url, params=params)
        data = response.json()

        candles = data.get("candles", {}).get("data", [])
        columns = data.get("candles", {}).get("columns", [])

        df = pd.DataFrame(candles, columns=columns)

        if df.empty:
            return df, None, None

        # 🔥 timestamp
        df["timestamp"] = pd.to_datetime(df["begin"]).astype("int64") // 10**6

        # оставляем только нужное
        df = df[["timestamp", "close"]].dropna()

        return df, None, None