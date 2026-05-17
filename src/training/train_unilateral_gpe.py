"""
Train a 1-D CNN that estimates gait phase (x,y) for one thigh.
The network is trained in a canonical “right-leg” format and can
be applied to BOTH legs at run-time by mirroring left-leg inputs.

Usage:
python unilateral_train_final.py 
"""

from __future__ import annotations
import argparse, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, BatchNormalization, Conv1D,
                                     Flatten, Dense, Dropout, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import r2_score

# ────────────────────────── hyper-params ──────────────────────────────
WINDOW       = 80           # 0.80 s @ 100 Hz
BATCH_SIZE   = 128
EPOCHS       = 50
EARLY_PAT    = 10           # stop if val_loss⊝10 epochs
REDLR_PAT    = 3            # halve LR after 3 stagnant epochs
MIN_LR       = 1e-5

# ───────── column templates ─────────
SENS_TPL = [                 # {S}=L or R
    "Thigh_{S}_Acc_X", "Thigh_{S}_Acc_Y", "Thigh_{S}_Acc_Z",
    "Thigh_{S}_Gyr_X", "Thigh_{S}_Gyr_Y", "Thigh_{S}_Gyr_Z",
    "mtr_pos_{S}", "mtr_vel_{S}"
]
LAB_TPL  = ["{s}_x", "{s}_y"]           # {s}=left / right  (lower-case)

# sensor channels that need sign inversion when mirrored
_SIGN_FLIP_SUFFIXES = ("_Acc_Y", "_Gyr_X", "_Gyr_Z", "mtr_pos_", "mtr_vel_")

# canonical side → always train in “R” format
CANON_SIDE = "R"
SENS_CANON = [c.replace("{S}", CANON_SIDE) for c in SENS_TPL]


# ───────────────────────── helper functions ───────────────────────────
def make_windows(sensor_data: np.ndarray, labels: np.ndarray, window: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (X,y) windows of shape (N, window, C) and (N, 2)."""
    idx = np.arange(len(sensor_data) - window + 1) # [0, 1, ..., total - window]
    center_idx = idx + window // 2 # one gait phase prediction per window
    X = np.stack([sensor_data[i:i+window] for i in idx]).astype(np.float32)
    
    y = labels[center_idx].astype(np.float32)
    return X, y


def _needs_flip(col: str) -> bool:
    """True if this column changes sign when mirrored L↔R"""
    return col.endswith(_SIGN_FLIP_SUFFIXES) or col.startswith(_SIGN_FLIP_SUFFIXES)


def mirrored_df(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Create a virtual copy of *df* that looks like it came from `side`
    ('L' or 'R') even though it actually came from the opposite thigh.
    Numeric values that change sign across sides are negated.
    """
    opp = "R" if side == "L" else "L"
    
    # ex. if side == "R", copy left-leg data and turn it into right-leg
    # src -> dst
    src_cols = [c.replace("{S}", opp) for c in SENS_TPL]
    dst_cols = [c.replace("{S}", side) for c in SENS_TPL]

    # renaming columns (ex. Thigh_L_Acc_X -> Thigh_R_Acc_X)
    out = (
        df[src_cols] # slices the DataFrame down to the src columns only
        .copy() # detaches it from the original frame
        .set_axis(dst_cols, axis=1)) # renames those columns to the dst names

    # flip sign where needed
    for col in dst_cols:
        if _needs_flip(col):
            out[col] *= -1.0
    return out

def windows_from_side(df: pd.DataFrame, side: str, win: int):
    """Create (X,y) windows for one physical leg, always in right leg format"""
    if side == CANON_SIDE:
        sens = df[[c.replace("{S}", side) for c in SENS_TPL]].values
    else:
        sens = mirrored_df(df, CANON_SIDE).values
    leg = "right" if side.lower() == "r" else "left"
    labs = df[[f"{leg}_x", f"{leg}_y"]].values
    return make_windows(sens, labs, win)

def load_split(folder: Path, win: int):
    X_all, y_all = [], []
    for csv in folder.rglob("*.csv"):
        df = pd.read_csv(csv)
        for leg in ("R", "L"):
            X, y = windows_from_side(df, leg, win)
            X_all.append(X)
            y_all.append(y)
    X = np.concatenate(X_all); y = np.concatenate(y_all)
    shuf = np.random.permutation(len(X))
    print(f"{folder.name:15s} → {len(X):,} windows")
    return X[shuf], y[shuf]

def load_test(folder: Path, side:str, win: int):
    X_te, y_te = [], []
    for csv in folder.rglob("*.csv"):
        df = pd.read_csv(csv)
        X_all, y_all = windows_from_side(df, side, win)
        X_te.append(X_all); y_te.append(y_all)
    X = np.concatenate(X_te); y = np.concatenate(y_te)
    shuf = np.random.permutation(len(X))
    return X[shuf], y[shuf]

def build_model(window_size: int, num_channel: int) -> tf.keras.Model:
    k_local = 20
    k_global = window_size - k_local + 1
    
    model = tf.keras.Sequential([
        Input(shape=(window_size, num_channel)),
        BatchNormalization(), # normalize raw sensors

        # 1) local temporal features 
        # output = input - filter + 1 = 80 - 20 + 1 = 61
        # short-range detections
        Conv1D(16, k_local, use_bias=False), 
        BatchNormalization(), Activation("relu"),
            
        # 2) global kernel over the remaining 76 samples 
        # output = input - filter + 1 = 61 - 61 + 1 = 1
        # stride-level phase estimation
        Conv1D(32, k_global, use_bias=False), 
        BatchNormalization(), Activation("relu"),

        Flatten(), # now shape (batch, 32)
        
        # 'Dropout' used as a regularization technique to prevent overfitting
        Dense(64, activation="relu"), Dropout(0.30),
        Dense(32, activation="relu"), Dropout(0.30),
        Dense(2,  activation="linear") # output : x, y
    ])
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss="mse", 
        metrics=["mae"])
    return model


# ───────────────────────────── main ───────────────────────────────────
def main(args):
    root = args.root
    model_dir = "IND_Jimin"

    X_tr, y_tr = load_split(root/model_dir/"CNN_Train",      WINDOW)
    X_va, y_va = load_split(root/model_dir/"CNN_Validation", WINDOW)
    
    X_te, y_te = load_split(root/model_dir/"CNN_Test",       WINDOW)
    X_te_r, y_te_r = load_test(root/model_dir/"CNN_Test", "R", WINDOW)
    X_te_l, y_te_l = load_test(root/model_dir/"CNN_Test", "L", WINDOW)
    

    model = build_model(WINDOW, len(SENS_CANON))
    model.summary()

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(patience=EARLY_PAT, monitor="val_loss", 
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=REDLR_PAT, factor=0.5, 
                              min_lr=MIN_LR, monitor="val_loss", verbose=1)
        ],
        verbose=2)

    model_path = root/model_dir/"gpe_cnn_jimin.h5"
    
    test_path = root/model_dir/"gpe_test_jimin.npz"
    test_path_r = root/model_dir/"gpe_test_jimin_r.npz"
    test_path_l = root/model_dir/"gpe_test_jimin_l.npz"
    
    model.save(model_path)
    np.savez(test_path, X_test=X_te, y_test=y_te)
    np.savez(test_path_r, X_test=X_te_r, y_test=y_te_r)
    np.savez(test_path_l, X_test=X_te_l, y_test=y_te_l)
    
    print(f"✓ saved model → {model_path}")

    # optional loss curves
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"],     label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.grid(alpha=.3)
    plt.legend(); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent,
                    help="project root (default = script folder)")
    main(ap.parse_args())
