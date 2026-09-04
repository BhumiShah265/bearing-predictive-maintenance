# ⚙️ AI-Based Bearing Predictive Maintenance System

A production-grade machine learning and deep learning system that analyzes motor bearing vibration signals from the **Case Western Reserve University (CWRU) Bearing Dataset** to diagnose mechanical fault conditions across 4 health states:
- 🟢 **Normal (Baseline)**
- 🔴 **Inner Race Fault**
- 🟠 **Outer Race Fault**
- 🟡 **Ball Fault**

---

## 🚀 Key Features

- **Leakage-Free Validation**: Stratified and grouped splitting (`GroupShuffleSplit` / `StratifiedShuffleSplit`) guarantees zero window overlap leakage across train/val/test splits.
- **Vibration Signal Processing**: Time-domain (RMS, Kurtosis, Crest Factor, Peak-to-Peak) & Frequency-domain (FFT, Spectral Centroid, Dominant Frequency) feature extraction.
- **Multi-Model Benchmark**: Compares Logistic Regression, SVM (RBF), Random Forest, Hist Gradient Boosting, PyTorch 1D CNN, and PyTorch 2D STFT Spectrogram CNN (**99.78% accuracy**).
- **FastAPI Production Backend**: REST API endpoints (`/api/predict-file` and `/api/sample/{fault_type}`) for batch and single-file signal inference.
- **Interactive Dark-Glass UI**: Web dashboard with real-time Chart.js signal plots, FFT magnitude spectrum, STFT spectrogram, and physical feature cards.

---

## 🛠️ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/BhumiShah265/bearing-predictive-maintenance.git
cd bearing-predictive-maintenance

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn scikit-learn torch torchvision fastapi uvicorn python-multipart requests
```

### 2. Train Models
```bash
# Train Classical Machine Learning Models (Random Forest, SVM, Hist Gradient Boosting)
python src/train_ml.py

# Train PyTorch Deep Learning Models (1D Raw ConvNet & 2D STFT Spectrogram ConvNet)
python src/train_dl.py

# Run Evaluation & Generate Confusion Matrix Heatmaps
python src/evaluate.py
```

### 3. Launch Web Application
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8005
```
Open **`http://127.0.0.1:8005/`** in your web browser.

---

## 📊 Benchmark Results

| Model Architecture | Input Signal Type | Test Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 16 Time/Freq Features | 88.80% | 90.44% |
| **Support Vector Machine (RBF)** | 16 Time/Freq Features | 91.96% | 93.23% |
| **Random Forest** | 16 Time/Freq Features | 93.80% | 94.72% |
| **Hist Gradient Boosting** | 16 Time/Freq Features | 95.00% | 95.81% |
| **PyTorch 1D CNN** | Raw Signal (1024 points) | 97.50% | 97.91% |
| **PyTorch 2D CNN** | STFT Spectrogram Matrix | **99.78%** | **99.82%** |

---

## 📁 Repository Structure

```
├── backend/
│   ├── main.py          # FastAPI application & REST endpoints
│   └── schemas.py       # Pydantic request/response models
├── frontend/
│   ├── index.html       # Web dashboard markup
│   ├── styles.css       # Dark glassmorphism styling
│   └── app.js           # Chart.js graphs & API client
├── src/
│   ├── data_loader.py   # CWRU MAT & NPZ dataset loader
│   ├── signal_processing.py # Time-domain, FFT & Spectrogram feature extractor
│   ├── dataset_split.py # Leakage-free dataset splitter
│   ├── train_ml.py      # Classical ML models training & tuning
│   ├── train_dl.py      # PyTorch 1D & 2D CNN models
│   ├── evaluate.py      # Metrics, confusion matrix & explainability
│   └── pipeline.py      # Deterministic inference engine
├── models/              # Saved model weights & scalers
└── reports/             # Generated confusion matrix & feature importances
```
