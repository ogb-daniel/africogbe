from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
import joblib
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "https://africogfe.vercel.app/",
    "http://localhost:3000",
]
app = FastAPI(title="Working Memory Model API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None

class PredictionRequest(BaseModel):
    Age: int
    WorkingMemory_Score: int

class PredictionResponse(BaseModel):
    predicted: int

@app.on_event("startup")
async def load_model():
    global model
    model_path = Path("workingmemorymodel.pkl")
    try:
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_score(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_data = np.array([[request.Age, request.WorkingMemory_Score]])
        prediction = model.predict(input_data)[0]
        
        return PredictionResponse(
            predicted=int(prediction),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)