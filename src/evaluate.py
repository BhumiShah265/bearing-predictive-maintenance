"""
evaluate.py - Model Evaluation, Error Analysis & Physical Feature Explainability
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from data_loader import prepare_cwru_dataset, LABEL_NAMES
from signal_processing import extract_all_features
from dataset_split import split_dataset_by_groups


def run_full_evaluation(models_dir: str = "models", reports_dir: str = "reports"):
    """
    Evaluates final trained model on test set, generates confusion matrix plot,
    computes feature importances, and outputs physical error analysis report.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load saved model, scaler, and feature names
    model_path = os.path.join(models_dir, "best_classical_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    features_path = os.path.join(models_dir, "feature_names.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, "r") as f:
        feature_names = json.load(f)

    # Load dataset & extract features
    X_raw, y, meta_df = prepare_cwru_dataset()
    feature_list = [extract_all_features(win) for win in X_raw]
    X_feats = pd.DataFrame(feature_list).values

    # Split using same group split
    _, _, _, _, X_test_raw, y_test, _ = split_dataset_by_groups(
        X_feats, y, meta_df, test_size=0.2, val_size=0.15
    )
    
    X_test = scaler.transform(X_test_raw)
    y_pred = model.predict(X_test)

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[LABEL_NAMES[i] for i in range(4)],
        yticklabels=[LABEL_NAMES[i] for i in range(4)]
    )
    plt.title("Bearing Fault Classification — Confusion Matrix")
    plt.xlabel("Predicted Health Condition")
    plt.ylabel("Actual Health Condition")
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Feature Importances (Random Forest or Linear Coefficients)
    feature_importances = {}
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        feature_importances = dict(zip(feature_names, imps.tolist()))
    elif hasattr(model, "coef_"):
        imps = np.mean(np.abs(model.coef_), axis=0)
        feature_importances = dict(zip(feature_names, imps.tolist()))

    # Sort feature importances
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    # Feature Importance Plot
    if sorted_importances:
        plt.figure(figsize=(10, 6))
        top_feats = list(sorted_importances.keys())[:10]
        top_scores = [sorted_importances[k] for k in top_feats]
        sns.barplot(x=top_scores, y=top_feats, palette="viridis")
        plt.title("Top 10 Physical & Spectral Vibration Features")
        plt.xlabel("Relative Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "feature_importances.png"), dpi=300)
        plt.close()

    # 3. Physical Feature Insights & Error Diagnostics
    explainability_report = {
        "classification_summary": classification_report(
            y_test, y_pred, labels=[0, 1, 2, 3], target_names=[LABEL_NAMES[i] for i in range(4)], output_dict=True
        ),
        "top_features": sorted_importances,
        "physical_explanations": {
            "kurtosis": "Measures signal tail peakedness. Spikes (> 5.0) indicate localized impact shocks from race or ball defects.",
            "rms": "Measures total vibration kinetic energy. Increases monotonically with fault severity and spalling surface damage.",
            "crest_factor": "Ratio of peak amplitude to RMS. Sensitive to early-stage transient impacts before overall energy increases.",
            "dominant_frequency": "Identifies primary spectral peak corresponding to bearing impact frequencies (BPFI, BPFO, BSF)."
        }
    }

    with open(os.path.join(reports_dir, "explainability_report.json"), "w") as f:
        json.dump(explainability_report, f, indent=2)

    print("=" * 60)
    print("EVALUATION & EXPLAINABILITY REPORT COMPLETED:")
    print(f"Confusion Matrix saved to: {cm_path}")
    print("Top 5 Physical Features:")
    for feat, score in list(sorted_importances.items())[:5]:
        print(f"  - {feat:20s}: {score:.4f}")
    print("=" * 60)
    return explainability_report


if __name__ == "__main__":
    run_full_evaluation()
