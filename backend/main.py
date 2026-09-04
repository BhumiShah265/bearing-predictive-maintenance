"""
main.py - FastAPI Backend Application for Bearing Fault Diagnostics
"""
import sys
import os
import io
import json
import numpy as np
from scipy.io import loadmat
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Include root src & backend paths in Python module search
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.dirname(__file__))

from pipeline import BearingFaultPredictor
from schemas import PredictionResponse, SignalArrayPayload
from data_loader import generate_synthetic_cwru_file, LABEL_NAMES

app = FastAPI(
    title="AI Bearing Predictive Maintenance API",
    description="Machine Learning & Deep Learning API for Rotating Machinery Vibration Diagnosis",
    version="1.0.0"
)

# Enable CORS for frontend web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = BearingFaultPredictor(models_dir="models")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": predictor.is_loaded,
        "supported_faults": list(LABEL_NAMES.values())
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_signal_payload(payload: SignalArrayPayload):
    """
    Predict bearing condition from a raw 1D vibration signal float array.
    """
    try:
        signal = np.array(payload.signal, dtype=np.float64)
        result = predictor.predict_signal(signal, fs=payload.sampling_rate)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-file", response_model=PredictionResponse)
async def predict_uploaded_file(file: UploadFile = File(...)):
    """
    Predict bearing condition from an uploaded vibration file (.mat, .csv, .npy, .txt, .json).
    """
    filename = file.filename.lower()
    content = await file.read()
    
    signal = None

    try:
        if filename.endswith(".mat"):
            # Load MATLAB file from memory byte stream
            bytes_io = io.BytesIO(content)
            mat_data = loadmat(bytes_io)
            # Find time series array key
            for key in mat_data.keys():
                if "DE_time" in key or key.endswith("_time"):
                    signal = mat_data[key].squeeze()
                    break
            if signal is None:
                raise ValueError("Could not locate vibration time-series array key in MATLAB file.")

        elif filename.endswith(".npy"):
            bytes_io = io.BytesIO(content)
            signal = np.load(bytes_io).squeeze()

        elif filename.endswith(".csv") or filename.endswith(".txt"):
            text_str = content.decode("utf-8", errors="ignore")
            # Parse numbers separating by comma, space, or newline
            lines = [line.strip() for line in text_str.split("\n") if line.strip()]
            numbers = []
            for line in lines[:3000]: # max 3000 lines
                parts = line.replace(",", " ").split()
                for p in parts:
                    try:
                        numbers.append(float(p))
                    except ValueError:
                        pass
            signal = np.array(numbers, dtype=np.float64)

        elif filename.endswith(".json"):
            data = json.loads(content)
            if isinstance(data, list):
                signal = np.array(data, dtype=np.float64)
            elif isinstance(data, dict) and "signal" in data:
                signal = np.array(data["signal"], dtype=np.float64)

        else:
            # Fallback text parsing
            text_str = content.decode("utf-8", errors="ignore")
            numbers = [float(val) for val in text_str.replace(",", " ").split() if val.replace(".", "", 1).replace("-", "", 1).isdigit()]
            signal = np.array(numbers, dtype=np.float64)

        if signal is None or len(signal) == 0:
            raise ValueError("No numeric vibration data points could be extracted from uploaded file.")

        result = predictor.predict_signal(signal)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file '{file.filename}': {str(e)}")


@app.get("/api/sample/{fault_type}", response_model=PredictionResponse)
def get_sample_prediction(fault_type: str):
    """
    Generates and evaluates a real sample signal for instantaneous demonstration.
    Supported types: 'normal', 'inner_race', 'outer_race', 'ball'
    """
    type_map = {
        "normal": "Normal",
        "inner_race": "Inner_Race",
        "outer_race": "Outer_Race",
        "ball": "Ball"
    }

    key = fault_type.lower()
    if key not in type_map:
        raise HTTPException(status_code=400, detail=f"Invalid fault type '{fault_type}'. Choose from {list(type_map.keys())}")

    target_type = type_map[key]
    temp_path = f"data/raw/temp_{key}.mat"
    os.makedirs("data/raw", exist_ok=True)
    
    generate_synthetic_cwru_file(temp_path, target_type, num_samples=4096)
    from data_loader import load_cwru_mat_file
    signal = load_cwru_mat_file(temp_path)
    
    result = predictor.predict_signal(signal)
    return result


# Mount Static directory for Frontend UI
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_frontend():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                return f.read()
        return "<h1>AI Bearing Predictive Maintenance API Running. UI loading...</h1>"
