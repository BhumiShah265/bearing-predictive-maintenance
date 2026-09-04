"""
data_loader.py - CWRU Bearing Dataset Acquisition, NPZ Loader & Windowing
"""
import os
import ssl
import urllib.request
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Dict, List, Tuple, Optional

# CWRU Drive-End 12k Sampling Rate Dataset URL mapping
CWRU_FILES = {
    "Normal": {
        "0HP": ("97.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/97.mat"),
        "1HP": ("98.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/98.mat"),
        "2HP": ("99.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/99.mat"),
        "3HP": ("100.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/100.mat"),
    },
    "Inner_Race": {
        "0HP": ("105.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/105.mat"),
        "1HP": ("106.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/106.mat"),
        "2HP": ("169.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/169.mat"),
        "3HP": ("209.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/209.mat"),
    },
    "Outer_Race": {
        "0HP": ("130.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/130.mat"),
        "1HP": ("131.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/131.mat"),
        "2HP": ("197.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/197.mat"),
        "3HP": ("234.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/234.mat"),
    },
    "Ball": {
        "0HP": ("118.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/118.mat"),
        "1HP": ("119.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/119.mat"),
        "2HP": ("185.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/185.mat"),
        "3HP": ("222.mat", "http://engineering.case.edu/bearingdatacenter/files/mat/222.mat"),
    }
}

LABEL_MAP = {
    "Normal": 0,
    "Inner_Race": 1,
    "Outer_Race": 2,
    "Ball": 3
}

LABEL_NAMES = {0: "Normal", 1: "Inner Race Fault", 2: "Outer Race Fault", 3: "Ball Fault"}


def load_cwru_npz_dataset(npz_path: str = "data/raw/CWRU_48k_load_1_CNN_data.npz") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads pre-processed CWRU 48k NPZ dataset (4600 samples x 1024 points).
    Maps 10 sub-fault types to 4 core health conditions:
      - 'Normal' -> 0 (Normal)
      - 'IR_007', 'IR_014', 'IR_021' -> 1 (Inner Race)
      - 'OR_007', 'OR_014', 'OR_021' -> 2 (Outer Race)
      - 'Ball_007', 'Ball_014', 'Ball_021' -> 3 (Ball Fault)
    """
    print(f"Loading user uploaded NPZ dataset from: {npz_path}")
    data_dict = np.load(npz_path)
    X_data = data_dict['data'] # shape: (4600, 32, 32)
    str_labels = data_dict['labels'] # shape: (4600,)

    # Flatten (32, 32) to 1D 1024 signal points
    N = len(X_data)
    X_raw = X_data.reshape(N, -1)

    y_labels = []
    metadata = []

    for idx, raw_label in enumerate(str_labels):
        label_str = str(raw_label)
        if label_str == "Normal":
            lbl = 0
        elif label_str.startswith("IR"):
            lbl = 1
        elif label_str.startswith("OR"):
            lbl = 2
        elif label_str.startswith("Ball"):
            lbl = 3
        else:
            lbl = 0

        y_labels.append(lbl)
        metadata.append({
            "file_id": f"npz_{label_str}",
            "load": "1HP",
            "label": lbl,
            "label_name": LABEL_NAMES[lbl],
            "window_idx": idx,
            "start_sample": 0,
            "end_sample": 1024
        })

    y = np.array(y_labels, dtype=int)
    meta_df = pd.DataFrame(metadata)
    print(f"NPZ Dataset Loaded Successfully: {X_raw.shape} samples across 4 classes.")
    return X_raw, y, meta_df


def download_cwru_dataset(data_dir: str = "data/raw") -> Dict[str, str]:
    """
    Downloads CWRU .mat files if not present locally.
    """
    os.makedirs(data_dir, exist_ok=True)
    downloaded_paths = {}

    for fault_type, loads in CWRU_FILES.items():
        for load, (filename, url) in loads.items():
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename} for {fault_type} ({load}) from CWRU server...")
                try:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=ctx) as response, open(filepath, 'wb') as out_file:
                        out_file.write(response.read())
                    print(f"Successfully downloaded {filename}")
                except Exception as e:
                    print(f"Direct download failed for {url}: {e}.")
            downloaded_paths[f"{fault_type}_{load}"] = filepath

    return downloaded_paths


def load_cwru_mat_file(filepath: str) -> np.ndarray:
    """
    Parses MATLAB .mat file and extracts Drive-End vibration array.
    """
    mat_data = loadmat(filepath)
    de_key = None
    for key in mat_data.keys():
        if "DE_time" in key or key.endswith("_time"):
            de_key = key
            break

    if de_key is None:
        raise KeyError(f"Could not find Drive-End vibration key in {filepath}")

    return mat_data[de_key].squeeze()


def create_sliding_windows(
    signal: np.ndarray,
    label: int,
    file_id: str,
    load: str,
    window_size: int = 2048,
    overlap: float = 0.5
) -> Tuple[np.ndarray, List[Dict]]:
    step = int(window_size * (1.0 - overlap))
    num_windows = (len(signal) - window_size) // step + 1
    
    windows = []
    metadata = []

    for i in range(num_windows):
        start = i * step
        end = start + window_size
        win = signal[start:end]
        windows.append(win)
        metadata.append({
            "file_id": file_id,
            "load": load,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "window_idx": i,
            "start_sample": start,
            "end_sample": end
        })

    return np.array(windows), metadata


def prepare_cwru_dataset(
    data_dir: str = "data/raw",
    window_size: int = 2048,
    overlap: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Primary dataset builder: Checks if custom NPZ file is present, otherwise builds from .mat files.
    """
    npz_path = os.path.join(data_dir, "CWRU_48k_load_1_CNN_data.npz")
    if os.path.exists(npz_path):
        return load_cwru_npz_dataset(npz_path)

    download_cwru_dataset(data_dir)
    
    all_windows = []
    all_labels = []
    all_metadata = []

    for fault_type, loads in CWRU_FILES.items():
        label = LABEL_MAP[fault_type]
        for load, (filename, _) in loads.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                signal = load_cwru_mat_file(filepath)
                windows, meta = create_sliding_windows(
                    signal, label, file_id=filename.replace(".mat", ""), load=load,
                    window_size=window_size, overlap=overlap
                )
                all_windows.append(windows)
                all_labels.append(np.full(len(windows), label))
                all_metadata.extend(meta)

    X_raw = np.vstack(all_windows)
    y = np.concatenate(all_labels)
    meta_df = pd.DataFrame(all_metadata)

    return X_raw, y, meta_df


if __name__ == "__main__":
    X, y, df = prepare_cwru_dataset()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE:")
    print(f"Total Samples (X): {X.shape}")
    print(f"Class Distribution:\n{pd.Series(y).map(LABEL_NAMES).value_counts()}")
    print("=" * 60)
