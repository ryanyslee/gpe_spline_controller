import os
import pandas as pd
import numpy as np
from io import StringIO
import scipy.signal as signal

'''
Preprocesses raw input data (IMU sensor readings and motor position data) along
with label data (ground reaction force (GRF) data) to generate structured CSV
files containing gait phase data suitable for training CNN models. The preprocessing
includes filtering, downsampling, detecting heel contacts, and computing gait
phases in both percentage (0~99%) and polar coordinates (x, y, θ).

Usage:
    python gait_data_processor.py
'''

# Subject weights
# These are used to determine subject-specific GRF heel contact thresholds
data_weights = {
    "AB01_Jimin": 80,
    "AB02_Rajiv": 61,
    "AB03_Amy": 52,
    "AB04_Changseob": 72,
    "AB05_Maria": 57,
    "AB06_Vaidehi": 42,
    "AB07_Leo": 74,
    "AB08_Adrian": 86,
    "AB09_Crystal": 86,
    "AB10_Pragya": 60,
    "AB11_Ryan": 72,
    "AB12_Ray": 79,
    "AB13_Hridayam": 75,
    "AB14_Evy": 68,
}

def load_grf(path: str, trial_time: int, samp_freq: int):
    '''
    Loads and parses the GRF label data file. Extracts vertical ground reaction
    forces for left and right foot, corrects for negative orientation,
    and returns them
    '''
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # locate the first “Frame, …” header
    hdr_idx = next(i for i, ln in enumerate(lines)
                   if ln.lstrip().startswith("Frame"))
    header  = lines[hdr_idx].strip().split(",")
    n_cols  = len(header)

    # rows = 1 (units) + 1000 Hz × trial_time s
    block   = lines[hdr_idx + 1 : hdr_idx + 2 + trial_time*samp_freq]

    # discard the units row (contains 'N' in Fx column)
    if block and block[0].split(",")[2].strip().upper() == "N":
        block = block[1:]

    # keep only rows with the correct column count
    body = [",".join(r.strip().split(",")[:n_cols])
            for r in block if len(r.strip().split(",")) == n_cols]

    df = pd.read_csv(
        StringIO("\n".join([",".join(header)] + body)),
        dtype=float,          # <───────── forces float64 everywhere
        low_memory=False,
        on_bad_lines="skip",  # silently drop any malformed row
    )

    # file stores downward force as negative
    GRF_l  = -df["Fz.1"].to_numpy()
    GRF_r = -df["Fz"  ].to_numpy()
    return GRF_l, GRF_r

def load_input_data(input_dir):
    '''
    Loads IMU and motor sensor data from CSV files into structured pandas
    DataFrames. Splits IMU data by sensor location (pelvis, thigh) and separates
    accelerometer (Acc) and gyroscope (Gyr) readings.
    Returns them grouped in dictionaries.
    '''
    imu_path = os.path.join(input_dir, "imu_data.csv")
    motor_path = os.path.join(input_dir, "motor_positions.csv")
    
    df_imu = pd.read_csv(imu_path)
    df_motor = pd.read_csv(motor_path)
    
    # IMU Data: Pelvis, Left thigh, Right thigh
    imu_data = {
        "pelvis_acc": df_imu[["Pelvis_Acc_X", "Pelvis_Acc_Y", "Pelvis_Acc_Z"]],
        "pelvis_gyr": df_imu[["Pelvis_Gyr_X", "Pelvis_Gyr_Y", "Pelvis_Gyr_Z"]],
        "left_thigh_acc": df_imu[["Thigh_L_Acc_X", "Thigh_L_Acc_Y", "Thigh_L_Acc_Z"]],
        "left_thigh_gyr": df_imu[["Thigh_L_Gyr_X", "Thigh_L_Gyr_Y", "Thigh_L_Gyr_Z"]],
        "right_thigh_acc": df_imu[["Thigh_R_Acc_X", "Thigh_R_Acc_Y", "Thigh_R_Acc_Z"]],
        "right_thigh_gyr": df_imu[["Thigh_R_Gyr_X", "Thigh_R_Gyr_Y", "Thigh_R_Gyr_Z"]],
    }
    
    # Motor Position & Velocity
    motor_data = {
        "hip_pos": df_motor[["mtr_pos_L", "mtr_pos_R"]],
        "hip_vel": df_motor[["mtr_vel_L", "mtr_vel_R"]],
    }
    
    return imu_data, motor_data

def butterworth_filter(data, cutoff, samp_freq, order=5):
    '''
    Applies a low-pass Butterworth filter to remove high-frequency noise.
    Returns a smoothed, filtered data
    '''
    b, a = signal.butter(order, cutoff / (samp_freq / 2), btype='low')
    return signal.filtfilt(b, a, data, axis=0)

def downsample(data, factor):
    '''
    Reduces data sampling rate by selecting every nth element
    '''
    return data[::factor]

def find_heel_contacts(grf_data, weight):
    '''
    Identifies heel contacts by threshold crossing in the vertical GRF signal.
    Returns a list of indices where heel strikes occur.
    '''
    heel_contacts = []
    threshold = weight * 9.80665 * 0.05
    for i in range(1, len(grf_data)):
        if grf_data[i-1] < threshold and grf_data[i] >= threshold:
            heel_contacts.append(i)
    return heel_contacts

def process_GRF(grf_l_raw, grf_r_raw):
    '''
    Smooths and downsamples raw GRF data for both feet.
    Returns 100 Hz processed signals.
    '''
    # Appying butterworth filter on GRF for smoothness
    grf_l_f = butterworth_filter(grf_l_raw, 6, 1000)
    grf_r_f = butterworth_filter(grf_r_raw, 6, 1000)
    
    # Downsampling GRF (1000Hz -> 100Hz)
    grf_l = downsample(grf_l_f, factor=10)
    grf_r = downsample(grf_r_f, factor=10)
    
    return grf_l, grf_r
    
def gait_phase_percentage(heel_contacts):
    '''
    Computes the gait phase in percentage (0~99%) between each pair of
    consecutive heel strikes.
    Returns a list of lists with tuples (index, %)
    '''
    gait_phase_percentages = []
    for i in range (len(heel_contacts) - 1):
        index_1 = heel_contacts[i]
        index_2 = heel_contacts[i+1]
        gpe = np.linspace(0, 99, index_2-index_1)
        
        current_gait_phase = []
        for x in range (index_2 - index_1):
            current_gait_phase.append((index_1 + x, gpe[x]))
            
        gait_phase_percentages.append(current_gait_phase)
    return gait_phase_percentages

def percent_to_polar(gait_phases):
    '''
    Converts gait phase percentages into polar coordinates (
    Returns a list of lists with tuples (index, x, y, θ)
    '''
    # list that stores tuple lists for all gait cycles
    polar_gait_phases = []
    for phase in gait_phases:
        # list that stores tuples for a single gait cycle
        polar_phase = []
        for index, percentage in phase:
            theta = (percentage / 100) * 2 * np.pi
            x = np.cos(theta)
            y = np.sin(theta)
            polar_phase.append((index, x, y, theta))
        polar_gait_phases.append(polar_phase)
    return polar_gait_phases

def create_complete_gait_df(imu_data, motor_data, gp_percent, gp_polar):
    '''
    Constructs a final pandas DataFrame for CNN training by merging sensor data
    and gait phase labels (both percentage and polar coordinates). 
    Ensures data alignment using valid index range intersection.
    '''
    # Unpack and flatten percent and polar data
    left_percent, right_percent = gp_percent[0], gp_percent[1]
    left_polar, right_polar = gp_polar[0], gp_polar[1]
    
    # Creating a single list with all the (index, percent) tuples
    # eg. [(index1, percent1),...]
    flat_left_percent = [item for phase in left_percent for item in phase]
    flat_right_percent = [item for phase in right_percent for item in phase]
    
    # Creating a single list with all the (index, x, y, θ) tuples
    # eg. [(index1, x1, y1, θ1)...]
    flat_left_polar = [item for phase in left_polar for item in phase]
    flat_right_polar = [item for phase in right_polar for item in phase]
    
    # Finding valid index range for the DataFrame
    start_index = max(flat_left_percent[0][0], flat_right_percent[0][0])
    end_index = min(flat_left_percent[-1][0], flat_right_percent[-1][0])
    time = np.arange(start_index, end_index + 1) / 100
    
    def slice_df(df):
        return df.iloc[start_index:end_index + 1].reset_index(drop=True)
    
    # Slice all sensor streams to align by index
    pelv_acc = slice_df(imu_data["pelvis_acc"])
    pelv_gyr = slice_df(imu_data["pelvis_gyr"])
    left_thigh_acc = slice_df(imu_data["left_thigh_acc"])
    left_thigh_gyr = slice_df(imu_data["left_thigh_gyr"])
    right_thigh_acc = slice_df(imu_data["right_thigh_acc"])
    right_thigh_gyr = slice_df(imu_data["right_thigh_gyr"])
    hip_pos = slice_df(motor_data["hip_pos"])
    hip_vel = slice_df(motor_data["hip_vel"])
    
    # Extract valid gait labels in range
    left_percent = [p[1] for p in flat_left_percent if start_index <= p[0] <= end_index]
    left_x = [p[1] for p in flat_left_polar if start_index <= p[0] <= end_index]
    left_y = [p[2] for p in flat_left_polar if start_index <= p[0] <= end_index]
    left_theta = [p[3] for p in flat_left_polar if start_index <= p[0] <= end_index]

    right_percent = [p[1] for p in flat_right_percent if start_index <= p[0] <= end_index]
    right_x = [p[1] for p in flat_right_polar if start_index <= p[0] <= end_index]
    right_y = [p[2] for p in flat_right_polar if start_index <= p[0] <= end_index]
    right_theta = [p[3] for p in flat_right_polar if start_index <= p[0] <= end_index]

    
    df = pd.DataFrame({
        'Time (s)': time,
        'Pelvis_Acc_X': pelv_acc.iloc[:, 0],
        'Pelvis_Acc_Y': pelv_acc.iloc[:, 1],
        'Pelvis_Acc_Z': pelv_acc.iloc[:, 2],
        'Pelvis_Gyr_X': pelv_gyr.iloc[:, 0],
        'Pelvis_Gyr_Y': pelv_gyr.iloc[:, 1],
        'Pelvis_Gyr_Z': pelv_gyr.iloc[:, 2],
        
        'Thigh_L_Acc_X': left_thigh_acc.iloc[:, 0],
        'Thigh_L_Acc_Y': left_thigh_acc.iloc[:, 1],
        'Thigh_L_Acc_Z': left_thigh_acc.iloc[:, 2],
        'Thigh_L_Gyr_X': left_thigh_gyr.iloc[:, 0],
        'Thigh_L_Gyr_Y': left_thigh_gyr.iloc[:, 1],
        'Thigh_L_Gyr_Z': left_thigh_gyr.iloc[:, 2],
        
        'Thigh_R_Acc_X': right_thigh_acc.iloc[:, 0],
        'Thigh_R_Acc_Y': right_thigh_acc.iloc[:, 1],
        'Thigh_R_Acc_Z': right_thigh_acc.iloc[:, 2],
        'Thigh_R_Gyr_X': right_thigh_gyr.iloc[:, 0],
        'Thigh_R_Gyr_Y': right_thigh_gyr.iloc[:, 1],
        'Thigh_R_Gyr_Z': right_thigh_gyr.iloc[:, 2],
        
        'mtr_pos_L' : hip_pos.iloc[:, 0],
        'mtr_pos_R' : hip_pos.iloc[:, 1],
        'mtr_vel_L' : hip_vel.iloc[:, 0],
        'mtr_vel_R' : hip_vel.iloc[:, 1],
        
        'left_percent': left_percent,
        'left_x': left_x,
        'left_y': left_y,
        'left_θ' : left_theta,
        
        'right_percent': right_percent,
        'right_x': right_x,
        'right_y': right_y,
        'right_θ' : right_theta
    })
    return df

def batch_process_all(data_root="./RARD_Data", output_root="./CNN_RARD"):
    '''
    Iterates through all subjects, speeds, and trials, preprocesses each trial
    using the full pipeline, and saves the result as a single CSV for
    CNN training.
    '''
    speeds = ["0p2mps", "0p4mps", "0p6mps", "0p8mps", "1p0mps", "1p2mps", 
              "1p4mps", "transient_15sec", "transient_30sec"]

    for subject in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject)
        if subject not in data_weights:
            continue
        weight = data_weights[subject]

        for speed in speeds:
            speed_path = os.path.join(subject_path, speed)
            trial_time = 14 if "15sec" in speed else 29

            if not os.path.exists(speed_path):
                continue

            for trial_num in ["1", "2", "3"]:
                trial_path = os.path.join(speed_path, f"trial_{trial_num}")
                if not os.path.exists(trial_path):
                    continue

                input_path, label_path = os.path.join(trial_path, "Input"), os.path.join(trial_path, "Label")
                if not os.path.exists(input_path) or not os.path.exists(label_path):
                    continue

                label_file = next((f for f in os.listdir(label_path) if f.endswith(".csv")), None)
                if not label_file:
                    continue

                print(f"subject: {subject}, speed: {speed}, trial: {trial_num}")
                imu_data, motor_data = load_input_data(input_path)
                grf_l_raw, grf_r_raw = load_grf(os.path.join(label_path, label_file), trial_time, 1000)
                grf_l, grf_r = process_GRF(grf_l_raw, grf_r_raw)
                l_hc, r_hc = find_heel_contacts(grf_l, weight), find_heel_contacts(grf_r, weight)

                gp_percent = [gait_phase_percentage(l_hc), gait_phase_percentage(r_hc)]
                gp_polar = [percent_to_polar(gp) for gp in gp_percent]

                df_gait = create_complete_gait_df(imu_data, motor_data, gp_percent, gp_polar)
                filename = f"{subject}_{speed}{trial_num}.csv"

                subject_output = os.path.join(output_root, subject)
                os.makedirs(subject_output, exist_ok=True)
                df_gait.to_csv(os.path.join(subject_output, filename), index=False)

                print(f"Saved: {filename}")


def _iter_trial_paths(data_root: str):
    """
    Yields tuples describing *every* trial folder it can find:

    (subject, subject_path, segment_label, angle, speed, ramp_tag,
     trial_num, trial_path)

    segment_label —— "level", "RA", or "RD"
    angle          —— e.g. "5deg" or "" for level ground
    ramp_tag       —— "RA", "RD", or ""   (used in filename)
    """
    # Speeds we expect to exist inside each level / incline folder
    speeds = [
        "0p4mps", "0p6mps", "0p8mps",
        "1p0mps", "transient_15sec", "transient_30sec",
    ]

    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue

        # ── 1) level-ground folders (speed directories live directly here)
        for speed in speeds:
            speed_path = os.path.join(subject_path, speed)
            if not os.path.isdir(speed_path):
                continue
            for trial in ("1", "2", "3"):
                tp = os.path.join(speed_path, f"trial_{trial}")
                if os.path.isdir(tp):
                    yield (subject, subject_path, "level", "", speed, "", trial, tp)

        # ── 2) RA / RD folders
        for ramp_tag in ("RA", "RD"):
            ramp_path = os.path.join(subject_path, ramp_tag)
            if not os.path.isdir(ramp_path):
                continue
            for angle in sorted(os.listdir(ramp_path)):          # 5deg, 10deg, …
                angle_path = os.path.join(ramp_path, angle)
                if not os.path.isdir(angle_path):
                    continue
                for speed in speeds:
                    speed_path = os.path.join(angle_path, speed)
                    if not os.path.isdir(speed_path):
                        continue
                    for trial in ("1", "2", "3"):
                        tp = os.path.join(speed_path, f"trial_{trial}")
                        if os.path.isdir(tp):
                            yield (subject, subject_path, ramp_tag,
                                   angle, speed, ramp_tag, trial, tp)


def batch_process_rard(data_root="./RARD_Data", output_root="./CNN_RARD"):
    """
    Walks through *every* trial returned by _iter_trial_paths, runs the full
    preprocessing pipeline and saves a CSV in CNN_RARD/Subject/.
    """
    # trial length: 14 s for the 15-sec transient blocks, else 29 s
    def _trial_secs(speed_folder: str) -> int:
        return 14 if "15sec" in speed_folder else 29

    for (subject, _subj_path, segment, angle, speed, ramp_tag,
         trial_num, trial_path) in _iter_trial_paths(data_root):

        if subject not in data_weights:         # skip unknown-weight subject
            continue
        weight = data_weights[subject]

        # locate input / label folders
        input_dir  = os.path.join(trial_path, "Input")
        label_dir  = os.path.join(trial_path, "Label")
        if not (os.path.isdir(input_dir) and os.path.isdir(label_dir)):
            continue

        # single CSV file inside Label/
        label_csv = next((f for f in os.listdir(label_dir) if f.endswith(".csv")), None)
        if label_csv is None:
            continue

        print(f"subject: {subject}, angle: {angle}, speed: {speed}, ramp: {ramp_tag}, trial: {trial_num}")

        # ── run preprocessing pipeline ────────────────────────────────
        imu_d, mtr_d = load_input_data(input_dir)
        grf_l_raw, grf_r_raw = load_grf(
            os.path.join(label_dir, label_csv),
            _trial_secs(speed), 1000
        )
        grf_l, grf_r = process_GRF(grf_l_raw, grf_r_raw)
        l_hc = find_heel_contacts(grf_l, weight)
        r_hc = find_heel_contacts(grf_r, weight)

        gp_percent = [gait_phase_percentage(l_hc),
                      gait_phase_percentage(r_hc)]
        gp_polar   = [percent_to_polar(gp) for gp in gp_percent]

        df = create_complete_gait_df(imu_d, mtr_d, gp_percent, gp_polar)

        fname = f"{subject}_{angle}_{speed}_{ramp_tag}_{trial_num}.csv"

        out_dir = os.path.join(output_root, subject)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, fname), index=False)
        print(f"✓ Saved  {fname}")

if __name__ == "__main__":
    # batch_process_all()
    batch_process_rard()
