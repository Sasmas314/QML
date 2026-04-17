from fastapi import FastAPI, UploadFile, File, HTTPException
from app.data.pipeline import DataPipeline

app = FastAPI()


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()

        pipeline = DataPipeline()
        df_raw, df_scaled, preprocessor = pipeline.process(content)

        return {
            "status": "success",
            "rows": len(df_raw),
            "columns": list(df_raw.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))