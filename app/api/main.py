from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

from app.data.pipeline import DataPipeline
from app.ml.models.baseline import BaselineModel
from app.ml.models.xgboost import XGBoostModel

app = FastAPI()


class RequestModel(BaseModel):
    ticker: str
    start: str
    end: str
    interval: int
    model: str
    steps: int


@app.post("/predict")
def predict(data: RequestModel):
    pipeline = DataPipeline()

    df, _, _ = pipeline.load_from_moex(
        ticker=data.ticker,
        start=data.start,
        end=data.end,
        interval=data.interval
    )

    if data.model == "Baseline":
        model = BaselineModel()
    else:
        model = XGBoostModel()

    forecast = model.predict(df, steps=data.steps)

    return {
        "data": df.to_dict(orient="records"),
        "forecast": forecast.to_dict(orient="records") if forecast is not None else []
    }