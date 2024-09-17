from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
from processing import preprocess_input
from model import load_model

app = FastAPI()

# Load the model
model = load_model('models/catboost_model.pkl')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    # Preprocess the data
    try:
        df_processed = preprocess_input(df)
    except Exception as e:
        return JSONResponse(content={"error": f"Preprocessing failed: {str(e)}"}, status_code=400)

    # Predict
    try:
        predictions = model.predict(df_processed)
    except Exception as e:
        return JSONResponse(content={"error": f"Prediction failed: {str(e)}"}, status_code=500)

    return JSONResponse(content={"predictions": predictions.tolist()}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
