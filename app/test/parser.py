from app.data.pipeline import DataPipeline


def main():
    pipeline = DataPipeline()

    df_raw, df_scaled, _ = pipeline.load_from_moex(
        ticker="SBER",
        start="2024-01-01",
        end="2024-03-01"
    )

    print("\n=== RAW DATA ===")
    print(df_raw.head())

    print("\n=== SCALED DATA ===")
    print(df_scaled.head())

    print("\nRows:", len(df_raw))


if __name__ == "__main__":
    main()