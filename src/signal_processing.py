"""
signal_processing.py - Vibration Feature Extraction & Time-Frequency Analysis
"""
import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from typing import Dict, List, Tuple


def extract_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Computes 10 statistical time-domain features for a 1D vibration signal window.
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms_val = np.sqrt(np.mean(signal**2)) + 1e-12
    peak_val = np.max(np.abs(signal))
    peak_to_peak = np.ptp(signal)
    mean_abs = np.mean(np.abs(signal)) + 1e-12
    
    # High-order statistics
    kurt = kurtosis(signal, fisher=False) # Pearson definition (Normal dist = 3.0)
    skw = skew(signal)
    
    # Dimensionless shape parameters
    crest_factor = peak_val / rms_val
    shape_factor = rms_val / mean_abs
    impulse_factor = peak_val / mean_abs
    margin_factor = peak_val / ((np.mean(np.sqrt(np.abs(signal))) + 1e-12) ** 2)

    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "rms": float(rms_val),
        "peak": float(peak_val),
        "peak_to_peak": float(peak_to_peak),
        "kurtosis": float(kurt),
        "skewness": float(skw),
        "crest_factor": float(crest_factor),
        "shape_factor": float(shape_factor),
        "impulse_factor": float(impulse_factor),
        "margin_factor": float(margin_factor),
    }


def compute_fft(signal: np.ndarray, fs: float = 12000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes single-sided Fast Fourier Transform (FFT) amplitude spectrum.
    Returns:
      - freqs: Positive frequency bins (Hz)
      - amplitudes: Magnitude spectrum (|X(f)|)
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    amplitudes = np.abs(fft_vals) / N
    amplitudes[1:-1] *= 2.0  # Conservation of energy for single-sided FFT
    return freqs, amplitudes


def extract_frequency_domain_features(signal: np.ndarray, fs: float = 12000.0) -> Dict[str, float]:
    """
    Computes frequency-domain features from the FFT amplitude spectrum.
    """
    freqs, amp = compute_fft(signal, fs=fs)
    total_amp = np.sum(amp) + 1e-12
    
    # Spectral Centroid (Center of Mass of frequencies)
    spectral_centroid = np.sum(freqs * amp) / total_amp
    
    # Spectral Energy & Variance
    spectral_energy = np.sum(amp**2)
    spectral_variance = np.sum(((freqs - spectral_centroid) ** 2) * amp) / total_amp
    spectral_std = np.sqrt(max(0.0, spectral_variance))
    
    # Dominant Frequency (Frequency with highest peak magnitude)
    dom_freq_idx = np.argmax(amp)
    dominant_frequency = freqs[dom_freq_idx]
    dominant_amplitude = amp[dom_freq_idx]

    return {
        "spectral_centroid": float(spectral_centroid),
        "spectral_energy": float(spectral_energy),
        "spectral_std": float(spectral_std),
        "dominant_frequency": float(dominant_frequency),
        "dominant_amplitude": float(dominant_amplitude),
    }


def extract_all_features(signal: np.ndarray, fs: float = 12000.0) -> Dict[str, float]:
    """
    Combines time-domain and frequency-domain features into a single dictionary.
    """
    time_feat = extract_time_domain_features(signal)
    freq_feat = extract_frequency_domain_features(signal, fs=fs)
    return {**time_feat, **freq_feat}


def compute_spectrogram(
    signal: np.ndarray,
    fs: float = 12000.0,
    nperseg: int = 128,
    noverlap: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Short-Time Fourier Transform (STFT) log-spectrogram matrix for 2D Deep Learning.
    Returns:
      - f: Frequency array
      - t: Time array
      - Sxx_log: 2D Log-magnitude spectrogram (shape: [freq_bins, time_steps])
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_magnitude = np.abs(Zxx)
    Sxx_log = np.log10(Sxx_magnitude + 1e-10) # Log scale for dynamic range
    return f, t, Sxx_log


if __name__ == "__main__":
    # Smoke test on synthetic signal
    dummy_signal = np.random.normal(0, 1, 2048)
    feats = extract_all_features(dummy_signal)
    print("Extracted Features Keys:", list(feats.keys()))
    print(f"Sample Features: RMS={feats['rms']:.4f}, Kurtosis={feats['kurtosis']:.4f}, DomFreq={feats['dominant_frequency']:.1f}Hz")
    f, t, Sxx = compute_spectrogram(dummy_signal)
    print(f"Spectrogram Matrix Shape: {Sxx.shape}")
