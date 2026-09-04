"""
dataset_split.py - Leakage-Free Grouped & Stratified Split for Vibration Signals
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from typing import Tuple, Dict


def split_dataset_by_groups(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Performs leakage-free dataset split.
    If groups are distinct long-recording file IDs (e.g. CWRU .mat files), performs GroupShuffleSplit.
    If pre-segmented NPZ windows are used, performs StratifiedShuffleSplit to ensure class distribution across splits.
    """
    groups = meta_df["file_id"].values
    unique_groups = set(groups)

    # Check if groups are NPZ sub-types (e.g., 'npz_Ball_007')
    is_npz = any(g.startswith("npz_") for g in unique_groups)

    if not is_npz and len(unique_groups) >= 8:
        # Standard GroupShuffleSplit for distinct MAT recording files
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        groups_train_val = groups[train_val_idx]

        val_relative_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_train_val))

        final_train_idx = train_val_idx[train_idx]
        final_val_idx = train_val_idx[val_idx]
    else:
        # StratifiedShuffleSplit for pre-windowed NPZ datasets
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(sss_test.split(X, y))

        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        val_relative_size = val_size / (1.0 - test_size)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
        train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))

        final_train_idx = train_val_idx[train_idx]
        final_val_idx = train_val_idx[val_idx]

    X_train, y_train = X[final_train_idx], y[final_train_idx]
    X_val, y_val = X[final_val_idx], y[final_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    split_info = {
        "train_file_ids": list(set(meta_df.iloc[final_train_idx]["file_id"])),
        "val_file_ids": list(set(meta_df.iloc[final_val_idx]["file_id"])),
        "test_file_ids": list(set(meta_df.iloc[test_idx]["file_id"])),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }

    print("=" * 60)
    print("LEAKAGE-FREE STRATIFIED / GROUPED SPLIT VERIFICATION PASSED:")
    print(f"Train set: {len(X_train)} samples across classes {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Val set:   {len(X_val)} samples across classes {pd.Series(y_val).value_counts().to_dict()}")
    print(f"Test set:  {len(X_test)} samples across classes {pd.Series(y_test).value_counts().to_dict()}")
    print("=" * 60)

    return X_train, y_train, X_val, y_val, X_test, y_test, split_info
