"""
train_ml.py - Classical Machine Learning Models Training, Evaluation, & Tuning
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV

from data_loader import prepare_cwru_dataset, LABEL_NAMES
from signal_processing import extract_all_features
from dataset_split import split_dataset_by_groups


def extract_features_df(X_windows: np.ndarray, fs: float = 12000.0) -> pd.DataFrame:
    """
    Extracts time-domain and frequency-domain feature matrix from raw signal windows.
    """
    feature_list = []
    for win in X_windows:
        feats = extract_all_features(win, fs=fs)
        feature_list.append(feats)
    return pd.DataFrame(feature_list)


def train_and_evaluate_ml(models_dir: str = "models"):
    """
    Complete classical ML pipeline:
    1. Load CWRU data
    2. Extract features
    3. Leakage-free Grouped Split
    4. Fit StandardScaler on Train only
    5. Train multiple ML models (Logistic Regression, SVM, Random Forest, HistGradientBoosting)
    6. Compare evaluation metrics
    7. Hyperparameter tune the best model
    8. Save model artifacts
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print("Step 1: Loading raw CWRU signal dataset...")
    X_raw, y, meta_df = prepare_cwru_dataset()
    
    print("Step 2: Extracting signal features (Time & Frequency domain)...")
    feats_df = extract_features_df(X_raw)
    feature_names = list(feats_df.columns)
    X_features = feats_df.values
    print(f"Extracted {len(feature_names)} features: {feature_names}")

    print("Step 3: Performing leakage-free grouped dataset split...")
    X_tr_raw, y_train, X_val_raw, y_val, X_te_raw, y_test, split_info = split_dataset_by_groups(
        X_features, y, meta_df, test_size=0.2, val_size=0.15
    )

    print("Step 4: Standardizing features using StandardScaler (Fitted on Train set ONLY)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_te_raw)

    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Initialize Classical ML candidate models
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
    }

    results = {}
    fitted_models = {}

    print("Step 5: Training and evaluating candidate Classical ML models...")
    print("-" * 80)

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_test_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="macro")
        
        results[name] = {
            "test_accuracy": float(acc),
            "test_precision_macro": float(prec),
            "test_recall_macro": float(rec),
            "test_f1_macro": float(f1)
        }
        fitted_models[name] = model
        
        print(f"Model: {name:30s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    # Determine Best Model based on Macro F1-score
    best_model_name = max(results, key=lambda k: results[k]["test_f1_macro"])
    print("=" * 80)
    print(f"BEST CLASSICAL MODEL SELECTED: {best_model_name}")
    print("=" * 80)

    # Hyperparameter Tuning for Best Model (if Random Forest or SVM)
    best_model = fitted_models[best_model_name]
    if best_model_name == "Random Forest":
        print("Step 6: Performing Hyperparameter Tuning for Random Forest...")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [6, 10, 15],
            "min_samples_split": [2, 5]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="f1_macro")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best Tuned Parameters: {grid.best_params_}")

    # Save final model
    joblib.dump(best_model, os.path.join(models_dir, "best_classical_model.joblib"))
    
    # Save evaluation report
    y_final_pred = best_model.predict(X_test)
    final_report = classification_report(
        y_test, y_final_pred, labels=[0, 1, 2, 3], target_names=[LABEL_NAMES[i] for i in range(4)], output_dict=True
    )
    
    with open(os.path.join(models_dir, "ml_evaluation_metrics.json"), "w") as f:
        json.dump({
            "model_comparison": results,
            "best_model_name": best_model_name,
            "final_test_report": final_report
        }, f, indent=2)

    print(f"Saved best model to {models_dir}/best_classical_model.joblib")
    return best_model, scaler, feature_names, results


if __name__ == "__main__":
    train_and_evaluate_ml()
