# Real-Time CNN Gait-Phase Estimator & Spline Controller
A real-time, CNN-based bilateral gait-phase estimator and Spline Controller for hip exoskeletons, achieving $R^2 > 0.80$ and $RMSE < 4.0%$ via TensorRT.

## 🎥 Live Demonstration
### ▶️ Click the image below to watch the Spline Controller GUI in action:
[![Spline Controller GUI Demo](https://img.youtube.com/vi/JjngNyeef_U/maxresdefault.jpg)](https://www.youtube.com/watch?v=JjngNyeef_U)

## 📌 Overview & System Pipeline
Traditional Time-Based Estimation (TBE) for robotic gait phase suffers from critical latency and discontinuities, especially during sudden speed transitions. This project solves that by replacing mathematical TBE models with a **data-driven 1D-CNN pipeline** that directly bridges raw sensor telemetry to physical actuation.

**1. Input Architecture:** Extracts features from an 80-step time-window at 100Hz across 8 channels (IMU Kinematics + Motor Encoder Position/Velocity).

**2. CNN Topology:** Utilizes a local kernel (size 20) for short-range temporal features, followed by a global kernel (size 61) for stride-level phase estimation.

**3. Cartesian Conversion:** The network outputs continuous Cartesian coordinates $(x, y)$, avoiding the $0 \rightarrow 100\%$ wrap-around discontinuity. This is converted to a continuous percentage via:

$$Gait Phase = \left(\left(\tan^{-1}\left(\frac{y}{x}\right)+2\pi\right)\bmod 2\pi\right)\times\frac{100}{2\pi}$$

**4. Bilateral Mirroring (Data Engineering):** To maximize efficiency, a single "canonical-right" model was trained. During live inference, left-leg sensor channels (e.g., Acc_Y, Gyr_X, Gyr_Z) are negated on the fly to simulate right-leg inputs, allowing one model to power both actuators.


## 📂 Dataset & Training Methodology
To ensure robust generalization, the models were trained on a diverse dataset comprising biomechanical ground-truth data (GRF and kinematics) captured via a Vicon Motion Cpature system, paired with simultaneous Jetson-logged IMU and encoder telemetry.

### Experimental Walking Conditions
Data was collected across multiple speed profiles, with each subject completing **3 trials per condition**:
- **Steady-State (30-second trials):** 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, and 1.4 m/s.
- **Transient (Acceleration/Deceleration):** Ramping from 0 m/s $/rightarrow$ 1.2 m/s $/rightarrow$ 0 m/s.
    - 15s Transient: Faster acceleration/deceleration profile.
    - 30s Transient: More gradual acceleration/deceleration profile.
 
### Model Types & Data Splits
Two distinct types of models were trained to evaluate different aspects of system robustness:

**1. Subject-Independent Models (IND)**
- Dataset: 14 total subjects.
- Split: 9 subjects for Training, 4 for Validation, and 1 left out entirely for testing (Leave-One-Subject-Out cross-validation).

**2. Hardware Synchronization Models (Sync vs. Unsync)**
- **The Context:** During data collection, the Vicon system triggered approximately 0.27 seconds earlier than the Jetson data logger.
- **The Models:** *The **Sync** dataset explicity accounts for and corrects this 0.27s hardware delay.
    - The **Unsync** dataset trains on the raw, unshifted timestamps.
- **Dataset:** 10 total subjects (randomized split).
- _Note on impact_: The 0.27s offset does not significantly impact offline validation, as the train/test sets are internally consistent. For online validation, comparing the final 10 gait cycles inherently aligns the data as long as the oscillation frequency remains constant.


## 📊 Key Results (Online & Offline Validation)
Online metrics were captured in real-time while the exoskeleton actively applied the spline torque assistance profile. _(Note: The subject utilized for the Sync/Unsync online validation was drawn from the validation split)._

| Model Architecture | Offline RMSE (%) | Offline R² | Online RMSE (%)* | Online R²* |
| :--- | :--- | :--- | :--- | :--- |
| **IND_AB01 (Subject-Independent)** | 1.87 | 0.908 | 2.80 | 0.827 |
| **IND_AB02 (Subject-Independent)** | 3.67 | 0.795 | 2.92 | 0.865 |
| **Synchronized (Sync)** | 4.90 | 0.820 | 3.63 | 0.790 |
| **Unsynchronized (Unsync)** | 5.09 | 0.849 | 3.10 | 0.776 |

*\*Online metrics represent the average bilateral (Left + Right leg) performance during active torque assistance.*


## ⚡ Hardware Integration & TensorRT Deployment
This repository is optimized for embedded hardware (e.g., NVIDIA Jetson) to ensure minimal mechanical latency and a stable **5ms control loop**.
- **TRT Engine Optimization:** The Keras CNN is exported to ONNX and compiled into a highly optimized TensorRT engine. The custom ```trt_runner.py``` wrapper ensures seamless execution across both TensorRT 8.x and 10.x APIs.
- **Batch-2 Simultaneous Inference:** Rather than processing each leg sequentially, '''run_spline_controller_batch2.py''' stacks the standard right-leg window and the mirrored left-leg window into a single ```(2, 80, 8)``` tensor. This halves the network inference overhead.
- **Vicon Synchronization:** Integrated TCp triggers automatically start and stop Vicon Nexus recording to perfectly align exoskeleton telemtry with ground-truth biomechanical data.


## 🎛️ Real-Time Spline GUI
An interactive PyQt5 interface (```spline_controller_gui.py```) was built to monitor and tune the exoskeleton without recompiling the controller.

### Key Features:
- **20Hz Live Telemetry:** Monitors motor position, velocity, commanded torque, and estimated gait phase for both legs.
- **On-the-Fly Spline Tuning:** Adjust the parameters of the Cubic Hermite Spline (extension/flexion phases and maximum torque) in real-time.
- **Automated Data Logging:** Effortlessly records synchronized CSV logs of all experiment states.


## 🚀 How to Run (Quickstart)
**1. Process Raw Biomechanical Data**

Convert 1000Hz GRF and 100Hz IMU and Motor Encoder data into structured 100Hz training windows:
```bash
python src/data_processing/gait_data_processor.py
```
**2. Train the CNN Model**
```bash
python src/training/train_unilateral_gpe.py
```
_(Note: Use ```src/deployment/export_tf_to_onnx.py``` and ```src/deployment/export_onnx_to_trt.py``` on your target hardware to generate the TensorRT engine)._

**3. Launch the Controller & GUI**

Initialize the TensorRT Batch-2 controller and open the interactive dashboard:
```bash
# Terminal 1: Start the hardware control loop
python src/controller/run_spline_batch2.py

# Terminal 2: Launch the visual interface
python src/controller/spline_controller_gui.py
```
