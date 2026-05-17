#!/usr/bin/env python3
"""
Evaluate a single canonical-right gait-phase CNN.

Outputs
-------
• RMSE (%) and R² for:
    1. all test windows,
    2. right-leg windows,
    3. left-leg windows (mirrored).
• Two phase-trace plots (right & left) from one CSV.
"""

from __future__ import annotations
import glob, argparse
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf

# ─────────────── constants ───────────────────────────────────────────
WINDOW   = 80
CANON    = "R"
MODEL_FN = "gpe_cnn_ryan.h5"
NPZ_ALL  = "gpe_test_ryan.npz"
NPZ_R    = "gpe_test_ryan_r.npz"
NPZ_L    = "gpe_test_ryan_l.npz"

SENS_TPL = [
    "Thigh_{S}_Acc_X", "Thigh_{S}_Acc_Y", "Thigh_{S}_Acc_Z",
    "Thigh_{S}_Gyr_X", "Thigh_{S}_Gyr_Y", "Thigh_{S}_Gyr_Z",
    "mtr_pos_{S}", "mtr_vel_{S}"
]
_SIGN_FLIP_SUFFIXES = ("_Acc_Y", "_Gyr_X", "_Gyr_Z", "mtr_pos_", "mtr_vel_")
SENS_CANON = [c.replace("{S}", CANON) for c in SENS_TPL]

# ─────────────── utilities ───────────────────────────────────────────
def needs_flip(col: str) -> bool:
    return col.endswith(_SIGN_FLIP_SUFFIXES) or col.startswith(_SIGN_FLIP_SUFFIXES)

def preprocess(win: np.ndarray, leg: str) -> np.ndarray:
    """Mirror left-leg window → canonical right format."""
    if leg.upper() == CANON:
        return win.astype(np.float32)
    win = win.copy()
    flip_idx = [i for i, c in enumerate(SENS_CANON) if needs_flip(c)]
    win[:, flip_idx] *= -1.0
    return win.astype(np.float32)

def cart2theta(x, y):          # rad ∈ [0, 2π)
    return (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)

def cart2pct(x, y):            # % ∈ [0, 100)
    return cart2theta(x, y) * 100 / (2*np.pi)

def circ_rmse(th_true, th_pred):
    d = np.abs(th_pred - th_true)
    d = np.minimum(d, 2*np.pi - d)
    return np.sqrt(np.mean(d**2))

def evaluate(model, X, y):
    y_hat = model.predict(X, verbose=0)
    th_p  = cart2theta(y_hat[:,0], y_hat[:,1])
    th_t  = cart2theta(y[:,0],    y[:,1])
    rmse  = circ_rmse(th_t, th_p) * 100 / (2*np.pi)
    r2    = r2_score(th_t, th_p)
    return rmse, r2

def phase_plot(df: pd.DataFrame, model, leg: str):
    sensor_cols = [c.replace("{S}", leg.upper()[0]) for c in SENS_TPL]
    label_cols  = [f"{leg}_x", f"{leg}_y"]

    sensor = df[sensor_cols].values
    labels = df[label_cols].values

    preds, gts, t_idx = [], [], []
    for i in range(len(sensor) - WINDOW):
        win = preprocess(sensor[i:i+WINDOW], leg.upper()[0])
        preds.append(model.predict(win[None, ...], verbose=0)[0])
        center = i + WINDOW//2
        gts.append(labels[center]); t_idx.append(center)

    preds = np.asarray(preds); gts = np.asarray(gts); t_idx = np.asarray(t_idx)
    pct_p = cart2pct(preds[:,0], preds[:,1])
    pct_t = cart2pct(gts[:,0],   gts[:,1])

    plt.figure(figsize=(10,4))
    plt.plot(t_idx/100, pct_t, label="ground truth", lw=1.2)
    plt.plot(t_idx/100, pct_p, "--", label="CNN")
    plt.ylabel("phase (%)"); plt.xlabel("time [s]")
    plt.title(f"{leg.capitalize()}-leg phase trace")
    plt.ylim(-5,105); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.show()

# ─────────────── main ────────────────────────────────────────────────
def main(args):
    root = Path(__file__).resolve().parent
    folder = root / "IND_Ryan"          # adjust if needed

    # load model
    model = tf.keras.models.load_model(folder/MODEL_FN, compile=False)
    model.compile(loss="mse", optimizer="sgd", metrics=["mae"])

    # evaluate three NPZ sets
    for tag, fname in [("ALL", NPZ_ALL), ("RIGHT", NPZ_R), ("LEFT", NPZ_L)]:
        X, y = np.load(folder/fname)["X_test"], np.load(folder/fname)["y_test"]
        rmse, r2 = evaluate(model, X, y)
        print(f"{tag:5s}  RMSE = {rmse:5.2f}%   R² = {r2:.3f}")

    # qualitative plots from one CSV
    # csv_path = sorted(
    #     glob.glob(str(folder/"CNN_Test/**/*.csv"), recursive=True)
    # )[-1]
    csv_path = "AB11_Ryan_transient_15sec1.csv"
    df = pd.read_csv(csv_path)

    for leg in ("right", "left"):
        phase_plot(df, model, leg)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None,
                    help="(optional) override project root")
    main(ap.parse_args())
