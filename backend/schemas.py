"""
schemas.py - FastAPI Pydantic Models for Bearing Predictive Maintenance
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class SignalArrayPayload(BaseModel):
    signal: List[float] = Field(..., description="1D array of vibration signal measurements (minimum 512 samples)")
    sampling_rate: Optional[float] = Field(12000.0, description="Sensor sampling rate in Hz (default 12kHz CWRU)")


class SignalFeaturesModel(BaseModel):
    mean: float
    std: float
    rms: float
    peak: float
    peak_to_peak: float
    kurtosis: float
    skewness: float
    crest_factor: float
    shape_factor: float
    impulse_factor: float
    margin_factor: float
    spectral_centroid: float
    spectral_energy: float
    spectral_std: float
    dominant_frequency: float
    dominant_amplitude: float


class PredictionResponse(BaseModel):
    predicted_condition: str
    confidence: float
    confidence_percentage: str
    class_probabilities: Dict[str, float]
    signal_features: SignalFeaturesModel
    time_series_data: Dict[str, List[float]]
    fft_data: Dict[str, List[float]]
    spectrogram_data: Dict[str, Any]
