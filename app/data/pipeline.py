from .moex_loader import MOEXDataLoader
from .validator import DataValidator
from .preprocess import Preprocessor


class DataPipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()

    def load_from_moex(self, ticker: str, start: str, end: str, interval: int = 24):
        # 1. загрузка
        loader = MOEXDataLoader(ticker)
        df = loader.load(start=start, end=end, interval=interval)

        # 2. валидация
        DataValidator.validate(df)

        # 3. нормализация
        df_scaled = self.preprocessor.fit_transform(df)

        return df, df_scaled, self.preprocessor