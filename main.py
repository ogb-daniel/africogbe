from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
import joblib
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "https://africogfe.vercel.app",
    "http://localhost:3000",
]
app = FastAPI(title="Cognitive Assessment Models API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


working_memory_model = None
attention_model = None
auditory_processing_model = None
processing_speed_model = None

class WorkingMemoryRequest(BaseModel):
    Age: int
    WorkingMemory_Score: int

class AttentionRequest(BaseModel):
    Age: int
    Attention_Score: int

class AuditoryProcessingRequest(BaseModel):
    Age: int
    AuditoryProcessing_Score: int

class ProcessingSpeedRequest(BaseModel):
    Age: int
    ProcessingSpeed_Score: int

class PredictionResponse(BaseModel):
    predicted: int

@app.on_event("startup")
async def load_models():
    global working_memory_model, attention_model, auditory_processing_model, processing_speed_model
    
    models = {
        "working_memory_model": "workingmemorymodel.pkl",
        "attention_model": "attentionmodel.pkl",
        "auditory_processing_model": "auditoryprocessingmodel.pkl",
        "processing_speed_model": "processingspeedmodel.pkl"
    }
    
    for model_name, file_name in models.items():
        model_path = Path(file_name)
        try:
            with open(model_path, "rb") as f:
                globals()[model_name] = joblib.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Model file not found at {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {model_name}: {str(e)}")

@app.post("/predict/working-memory", response_model=PredictionResponse)
async def predict_working_memory(request: WorkingMemoryRequest):
    if working_memory_model is None:
        raise HTTPException(status_code=500, detail="Working memory model not loaded")
    
    try:
        input_data = np.array([[request.Age, request.WorkingMemory_Score]])
        prediction = working_memory_model.predict(input_data)[0]
        
        return PredictionResponse(
            predicted=int(prediction),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/attention", response_model=PredictionResponse)
async def predict_attention(request: AttentionRequest):
    if attention_model is None:
        raise HTTPException(status_code=500, detail="Attention model not loaded")
    
    try:
        input_data = np.array([[request.Age, request.Attention_Score]])
        prediction = attention_model.predict(input_data)[0]
        
        return PredictionResponse(
            predicted=int(prediction),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/auditory-processing", response_model=PredictionResponse)
async def predict_auditory_processing(request: AuditoryProcessingRequest):
    if auditory_processing_model is None:
        raise HTTPException(status_code=500, detail="Auditory processing model not loaded")
    
    try:
        input_data = np.array([[request.Age, request.AuditoryProcessing_Score]])
        prediction = auditory_processing_model.predict(input_data)[0]
        
        return PredictionResponse(
            predicted=int(prediction),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/processing-speed", response_model=PredictionResponse)
async def predict_processing_speed(request: ProcessingSpeedRequest):
    if processing_speed_model is None:
        raise HTTPException(status_code=500, detail="Processing speed model not loaded")
    
    try:
        input_data = np.array([[request.Age, request.ProcessingSpeed_Score]])
        prediction = processing_speed_model.predict(input_data)[0]
        
        return PredictionResponse(
            predicted=int(prediction),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": {
            "working_memory": working_memory_model is not None,
            "attention": attention_model is not None,
            "auditory_processing": auditory_processing_model is not None,
            "processing_speed": processing_speed_model is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)