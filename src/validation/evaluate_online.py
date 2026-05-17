# ───────────────── online_gpe_vs_vicon.py ──────────────────
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path
from io import StringIO
from sklearn.metrics import r2_score

# ---------- helpers (unchanged) ------------------------------------------------
def butterworth_filter(d, fc, fs, order=5):
    b,a = signal.butter(order, fc/(fs/2), 'low'); return signal.filtfilt(b,a,d)
def downsample(x, n): return x[::n]

def find_heel_contacts(grf, mass):
    thr = mass*9.80665*0.05
    return [i for i in range(1,len(grf)) if grf[i-1]<thr and grf[i]>=thr]

def heel_contacts_jetson(p, hi=90, lo=10):
    p=np.asarray(p); wraps=(p[:-1]>hi)&(p[1:]<lo); return list(np.where(wraps)[0]+1)

def load_grf(csv_path:str, seconds:int, fs:int=1000):
    with open(csv_path,'rb') as f: raw=f.read()
    txt = raw.decode('utf-16-le' if raw[:2]==b'\xff\xfe' else
                     'utf-16-be' if raw[:2]==b'\xfe\xff' else 'utf-8','replace')
    lines = txt.splitlines()
    hdr = next(i for i,l in enumerate(lines) if l.lstrip().startswith('Frame'))
    header = lines[hdr].split(','); ncol=len(header)
    rows = lines[hdr+1:hdr+2+seconds*fs]
    if rows and rows[0].split(',')[2].strip().upper()=='N': rows=rows[1:]
    good = [",".join(r.split(',')[:ncol]) for r in rows
            if len(r.split(','))==ncol]
    df = pd.read_csv(StringIO("\n".join([",".join(header)]+good)),dtype=float)
    return -df["Fz.1"].to_numpy(), -df["Fz"].to_numpy()   # sign-flip

def align_last_k_cycles(hc_v, hc_j, k=10, tol=40, verbose=False):
    """
    Align the last *k* strides between Vicon (hc_v) and Jetson (hc_j)
    """
    pairs = []
    i, j = len(hc_v) - 1, len(hc_j) - 1          # start from the end

    while i > 0 and j > 0 and len(pairs) < k:
        if abs(hc_v[i] - hc_j[j]) <= tol:        # match found
            pairs.append((hc_v[i-1], hc_v[i],
                           hc_j[j-1], hc_j[j]))

            if verbose:
                # Ordinal heel-contact numbers (1-based counting)
                ord_v = i + 1        # because i is 0-based index into hc_v
                ord_j = j + 1
                print(f"[{len(pairs):>2}]  Vicon HC #{ord_v:<4} ↔ "
                      f"Jetson HC #{ord_j:<4}   (Δ = {hc_v[i]-hc_j[j]:+d} samples)")

            i -= 1
            j -= 1
        elif hc_v[i] > hc_j[j]:
            i -= 1                   # advance Vicon pointer
        else:
            j -= 1                   # advance Jetson pointer

    return pairs[::-1]               # earliest → latest
                                         # earliest→latest

# ───── add a tiny helper just above main() ────────────────────────────
def pct_to_theta(p):                      # 0‒100 %  →  0‒2π rad
    return (np.asarray(p) % 100) * 2*np.pi / 100

def circular_err_deg(a, b):
    """
    a, b already in rad.  Return the wrapped difference in **percent**.
    """
    d = np.angle(np.exp(1j*(a-b)))        # shortest signed diff, −π…π
    return np.abs(d) * 100 / (2*np.pi)    # back to percent (0…50 %)


# ------------------------------------------------------------------------------
def main(root:Path, weight:float=75, k: int=10, foot:str="right"):

    # vicon_csv  = root/"Online_Validations/Vicon/gpe_test_ryan_assi.csv"
    # jetson_csv = root/f"Online_Validations/Jetson/gpe_log_ryan_{foot}_assi.csv"
    
    vicon_csv  = root/"Online_Validations/Vicon/gpe_test_hridayam_assi.csv"
    jetson_csv = root/f"Online_Validations/Jetson/gpe_log_hridayam_{foot}_assi.csv"
    
    # vicon_csv  = root/"Online_Validations/Vicon/gpe_test_sync_assi.csv"
    # jetson_csv = root/f"Online_Validations/Jetson/gpe_log_sync_{foot}_assi.csv"
    
    # vicon_csv  = root/"Online_Validations/Vicon/gpe_test_unsync_assi.csv"
    # jetson_csv = root/f"Online_Validations/Jetson/gpe_log_unsync_{foot}_assi.csv"

    # ---------- load signals ---------------------------------------------------
    grf_L_raw, grf_R_raw = load_grf(vicon_csv, seconds=30)
    grf_f = butterworth_filter(grf_R_raw if foot=="right" else grf_L_raw, 6, 1000)
    grf   = downsample(grf_f, 10)                                                  # 100 Hz

    phase_pct = pd.read_csv(jetson_csv)["phase_pct"].to_numpy()

    # ---------- heel contacts --------------------------------------------------
    hc_v = find_heel_contacts(grf, weight)                   # Vicon @100 Hz
    hc_j = heel_contacts_jetson(phase_pct)                   # Jetson @100 Hz
    
    pairs = align_last_k_cycles(hc_v, hc_j, k=k, tol=60, verbose=True)
    if len(pairs)<k:
        print(f"⚠ Only {len(pairs)} cycles matched (tol = 40 samples)")

    # ---------- build concatenated arrays on a TIME axis -----------------
    # ----- build arrays with REAL samples per stride --------------------
    v_all, j_all, t_axis = [], [], []
    t_cursor = 0.0                        # 0 s at the chosen alignment HC

    v_total = j_total = 0
    for pv, nv, pj, nj in pairs:          # (prev HC, next HC) in Vicon/Jetson
        v_cycle = np.linspace(0, 99, nv - pv)          # one value per *real* sample
        j_cycle   = phase_pct[pj:nj]                     # Jetson has (nj-pj) samples
        
        v_all.append(v_cycle)
        j_all.append(j_cycle)
        
        v_total += len(v_cycle)
        j_total += len(j_cycle)
        
    v_all = np.concatenate(v_all)
    j_all = np.concatenate(j_all)
    
    min_len = min(v_all.size, j_all.size)
    v_all = v_all[:min_len]
    j_all = j_all[:min_len]
    
    t_axis = np.arange(min_len) * 0.01
    
    theta_v = pct_to_theta(v_all)
    theta_j = pct_to_theta(j_all)
    
    err_pct = circular_err_deg(theta_v, theta_j)

    # ---------- metrics --------------------------------------------------
    rmse = np.sqrt(np.mean((err_pct) ** 2))
    r2 = r2_score(theta_v, theta_j)
    # r2   = 1 - np.sum((err_pct) ** 2) / np.sum((v_all - v_all.mean()) ** 2)
    print(f"After trim  →  {min_len} samples  "
      f"(RMSE = {rmse:.2f} %,  R² = {r2:.3f})")

    # ---------- plot -----------------------------------------------------
    plt.figure(figsize=(10,4))
    plt.plot(t_axis, v_all, '-', label="Vicon (ground-truth)")
    plt.plot(t_axis, j_all, '-', label="Jetson GPE", alpha=.8)
    for split in np.unique(t_axis[:-1][np.diff(t_axis)<0]):   # dashed HCs
        plt.axvline(split, ls="--", color="gray", lw=.5)

    plt.xlabel("Time [s]")
    plt.ylabel("Gait-phase (%)")
    plt.title(f"{foot.capitalize()} – last {len(pairs)} aligned cycles\n"
            f"RMSE = {rmse:.2f} %,  R² = {r2:.3f}")
    plt.legend(); plt.tight_layout(); plt.show()

# -------------- CLI -----------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent)
    ap.add_argument("--foot", choices=["left","right"], default="right")
    ap.add_argument("--cycles", type=int, default=10, help="how many to compare")
    args = ap.parse_args()
    main(args.root, foot=args.foot, k=args.cycles)
