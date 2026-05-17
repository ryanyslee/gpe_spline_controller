"""
Quick visual check of GRF, heel-strike events, and gait-phase %
----------------------------------------------------------------
Call example
    python plot_gait_phase.py \
        --trial-dir "N14_Data_Sync/AB01_Jimin/0p8mps/trial_1" \
        --foot left
"""

import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from io import StringIO
import scipy.signal as signal

# ---------- ——— reuse helpers from your main module ——— ---------- #
data_weights = {  # (copy from gait_data_processor.py)
    "AB01_Jimin": 80, "AB02_Rajiv": 61, "AB03_Amy": 52,  "AB04_Changseob": 72,
    "AB05_Maria": 57, "AB06_Vaidehi": 42, "AB07_Leo": 74, "AB08_Adrian": 86,
    "AB09_Crystal": 86,"AB10_Pragya": 60, "AB11_Ryan": 72, "AB12_Ray": 79,
    "AB13_Hridayam": 75,"AB14_Evy": 68,
}

def butterworth_filter(data, cutoff, samp_freq, order=5):
    b, a = signal.butter(order, cutoff / (samp_freq / 2), btype='low')
    return signal.filtfilt(b, a, data, axis=0)

def downsample(arr, factor): return arr[::factor]

def find_heel_contacts(grf, body_mass):
    thresh = body_mass * 9.80665 * 0.05
    return [i for i in range(1, len(grf)) if grf[i-1] < thresh and grf[i] >= thresh]

def gait_phase_percentage(hc_idx):
    """return list of (index, %) pairs for *each* stride"""
    gpe_seqs = []
    for s, e in zip(hc_idx[:-1], hc_idx[1:]):
        pct = np.linspace(0, 99, e-s)
        gpe_seqs.append([(s+i, p) for i, p in enumerate(pct)])
    return gpe_seqs
# ------------------------------------------------------------------ #

def load_grf(label_csv_path, trial_time_s):
    """copy-pasted and slimmed from original"""
    with open(label_csv_path, encoding="utf-8") as f: lines = f.readlines()
    hdr = next(i for i,l in enumerate(lines) if l.lstrip().startswith("Frame"))
    header = lines[hdr].strip().split(",")
    n_cols = len(header)
    block  = lines[hdr+1 : hdr+2 + trial_time_s*1000]   # 1000 Hz rows
    if block[0].split(",")[2].strip().upper() == "N": block = block[1:]
    body = [",".join(r.strip().split(",")[:n_cols]) for r in block
            if len(r.strip().split(",")) == n_cols]
    df = pd.read_csv(StringIO("\n".join([",".join(header)] + body)),
                     dtype=float, low_memory=False, on_bad_lines="skip")
    return -df["Fz.1"].to_numpy(), -df["Fz"].to_numpy()    # L, R (sign flip)

def plot_one_side(time, grf, hc_idx, gp_list, foot):
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(time, grf, lw=1.2, label="GRF", color="steelblue")
    ax1.set_ylabel("Force (N)", color="steelblue"); ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="steelblue")

    for hc in hc_idx:
        ax1.axvline(hc/100, ls="--", color="green", alpha=.5)

    ax2 = ax1.twinx()
    for stride in gp_list:
        idx, pct = zip(*stride)
        ax2.plot(np.array(idx)/100, pct, color="crimson", alpha=.6, lw=1)
    ax2.set_ylabel("Gait phase (%)", color="crimson")
    ax2.tick_params(axis='y', labelcolor="crimson")

    ax1.set_title(f"{foot.capitalize()} foot – GRF, heel contacts, gait phase")
    ax1.legend(["GRF", "Heel contact"], loc="upper right")
    fig.tight_layout(); plt.show()

def main(trial_dir, foot):
    subject = os.path.basename(os.path.dirname(os.path.dirname(trial_dir)))
    if subject not in data_weights:
        raise ValueError(f"Body-mass of {subject} unknown in data_weights dict")

    # ---------- paths ---------- #
    label_dir = os.path.join(trial_dir, "Label")
    label_csv = next(f for f in os.listdir(label_dir) if f.endswith(".csv"))
    trial_len = 14 if "15sec" in trial_dir else 29

    # ---------- load & preprocess ---------- #
    grf_L_raw, grf_R_raw = load_grf(os.path.join(label_dir, label_csv), trial_len)
    grf_L_f = butterworth_filter(grf_L_raw, 6, 1000)
    grf_R_f = butterworth_filter(grf_R_raw, 6, 1000)
    grf_L   = downsample(grf_L_f, 10)     # -> 100 Hz
    grf_R   = downsample(grf_R_f, 10)

    if foot == "left":
        grf = grf_L
        hc_idx = find_heel_contacts(grf_L, data_weights[subject])
    else:
        grf = grf_R
        hc_idx = find_heel_contacts(grf_R, data_weights[subject])

    gp_pct  = gait_phase_percentage(hc_idx)
    t = np.arange(len(grf)) / 100        # 100 Hz ⇒ seconds

    # ---------- plot ---------- #
    plot_one_side(t, grf, hc_idx, gp_pct, foot)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", required=True,
                    help="…/subject/speed/trial_N directory")
    ap.add_argument("--foot", choices=["left","right"], default="left")
    main(**vars(ap.parse_args()))
