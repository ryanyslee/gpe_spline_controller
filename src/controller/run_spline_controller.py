"""
Real-time bilateral hip-exo controller (TensorRT, static batch).

• One canonical-right CNN powers **both** motors.
• Left-leg sensor channels that change sign are mirrored before inference.
• Automatically starts/stops Vicon (Nexus) recording via TCP trigger.
"""

import time, enum, socket, os
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import CubicHermiteSpline

# --- New Epicpower imports ---
from epicpower_tmotorV3.actuator_group import ActuatorGroup
from epicpower_tmotorV3.tmotor_v3 import TMotorV3

from Header_ICM20948_I2C_pcb2 import ICM20948_I2C_IMUs
from Header_Mocap_trigger import Mocap_trigger
from trt_runner_v2 import TrtRunner  # .infer(window) → (1, 2)

# ───────────── experiment timing ────────────────────────────────────
TRIAL_DUR_SEC = 35 # total length  (warm-up + active) -> 30s trials
WARM_UP_SEC   = 5
LOOP_HZ       = 100
ACTIVE_SEC    = TRIAL_DUR_SEC - WARM_UP_SEC
EXPECTED_SAMPLES = int(ACTIVE_SEC * LOOP_HZ)

# ───────────── Vicon trigger setup ──────────────────────────────────
USE_MOCAP = False
MOCAP_IP   = "172.24.44.177"
MOCAP_PORT = 10
TRIGGER_MODE = "mocap"

# ───────────── Teleplot streaming ───────────────────────────────────
_TELE_ADDR = ("127.0.0.1", 47269)
_UDP_SOCK  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send_tele(name: str, val: float):
    ts = int(time.time()*1000)
    _UDP_SOCK.sendto(f"{name}:{ts}:{val}|g".encode(), _TELE_ADDR)

# ───────────── enums & data classes ─────────────────────────────────
class ExoSide(enum.Enum):
    LEFT  = 1
    RIGHT = 2

@dataclass
class ControlData:
    can_id: int
    side:   ExoSide
    motors: ActuatorGroup
    runner: TrtRunner

    # internal state
    window: np.ndarray = field(init=False)
    filled: int        = 0

    # constants
    WSIZE = 80                         # 0.80 s @ 100 Hz
    SHIFT = -15.0                      # phase shift
    MAX_TQ = 2.0                       # set >0 Nm if you want assist torque

    FLIP_IDX = (1, 3, 5, 6, 7)        # channels to negate for L→R mirror
    spline: CubicHermiteSpline = field(init=False)
    log:    list = field(default_factory=list)

    def __post_init__(self):
        self.window = np.zeros((self.WSIZE, 8), np.float32)
        x   = np.array([0,20,30,65,80,85,100])
        y   = np.array([-1,0,0,1,0,0,-1]) * self.MAX_TQ
        dyd = np.array([0,1e-4,1e-4,0,-1e-4,-1e-4,0])
        self.spline = CubicHermiteSpline(x, y, dyd, extrapolate='periodic')

    # ───────── core helpers ────────────────────────────────────────
    def push_sample(self, imu6, pos, vel):
        if imu6.size != 6:
            imu6 = np.zeros(6, np.float32)
        s = np.hstack((imu6, [pos, vel])).astype(np.float32)
        if self.side == ExoSide.LEFT:
            s[list(self.FLIP_IDX)] *= -1.0
        self.window[:-1] = self.window[1:]
        self.window[-1] = s
        
        self.filled = min(self.filled+1, self.WSIZE)

    def phase_pct(self):
        xy = self.runner.infer(self.window)[0]
        return float((np.arctan2(xy[1], xy[0]) % (2*np.pi)) * 100/(2*np.pi))

    def step(self, imu6, pos, vel):
        self.push_sample(imu6, pos, vel)
        if self.filled < self.WSIZE: # Need to wait for 80 data points to be logged
            send_tele(f"phase_{self.side.name[0]}_%", 0.0)
            return 0.0, 0.0 # No torque applied, No gait phase estimation 
        pct = (self.phase_pct() + self.SHIFT) % 100.0
        send_tele(f"phase_{self.side.name[0]}_%", pct)
        tq  = float(self.spline(pct))
        return tq, pct

# ───────────── controller factory ───────────────────────────────────
def build_controllers(root: Path):
    mtr_L = TMotorV3(1, "AK80-9")
    mtr_R = TMotorV3(2, "AK80-9")
    
    motors = ActuatorGroup([mtr_L, mtr_R])
    
    runner = TrtRunner(root / "trt_models_new" / "gpe_cnn_ryan.trt")
    return {s: ControlData(s.value, s, motors, runner) for s in ExoSide}

# ───────────── main ────────────────────────────────────────────────
def main():
    os.system("echo nc -u -w0 127.0.0.1 47269")
    # "model_dir" -> "trt_models" -> "model_name"
    root = Path(__file__).parent / "IND_model"
    ctrls = build_controllers(root)
    imu   = ICM20948_I2C_IMUs()
    
    mocap = None

    # ── Vicon trigger ------------------------------------------------
    if USE_MOCAP:
        mocap = Mocap_trigger(MOCAP_IP, MOCAP_PORT)
        mocap.start_client()
        if TRIGGER_MODE == "mocap":
            print("Waiting for Vicon trigger …")
            mocap.wait_for_trigger()
        else:
            input("Press ⏎ to start …")
        try:
            mocap.client.sendall(b"exo on")
            print("▶ Sent “exo on” to Vicon")
        except Exception as e:
            print(f"Failed to send 'exo on' to Vicon: {e}")

    t0        = time.time()
    t_start   = t0 + WARM_UP_SEC
    t_end     = t_start + ACTIVE_SEC
    sample_ct = 0
    
    # Native Python clock to replace epicpower.utils
    loop_period = 1.0 / LOOP_HZ
    next_loop_time = time.time()

    # ── control loop -------------------------------------------------
    try:
        while imu.IMUs_are_on:
            while time.time() < next_loop_time:
                pass
            next_loop_time += loop_period
        
            now = time.time()
            if USE_MOCAP:
                if   now <  t_start: continue          # still warming up
                elif now >= t_end:   break             # finished

            # ---------- sensors -------------------------------------
            imu_L = np.asarray(imu.read_IMU(imu.IMU_mux_ports["IMU_THIGH_LEFT" ]), np.float32)
            imu_R = np.asarray(imu.read_IMU(imu.IMU_mux_ports["IMU_THIGH_RIGHT"]), np.float32)
            
            # Added degrees=False kwarg for V3 API
            posL = ctrls[ExoSide.LEFT ].motors.get_position(1, degrees=False)
            velL = ctrls[ExoSide.LEFT ].motors.get_velocity(1, degrees=False)
            posR = ctrls[ExoSide.RIGHT].motors.get_position(2, degrees=False)
            velR = ctrls[ExoSide.RIGHT].motors.get_velocity(2, degrees=False)

            tq_L, pct_L = ctrls[ExoSide.LEFT ].step(imu_L, posL, velL)
            tq_R, pct_R = ctrls[ExoSide.RIGHT].step(imu_R, posR, velR)
            
            ctrls[ExoSide.LEFT ].motors.set_torque(1, tq_L)
            ctrls[ExoSide.RIGHT].motors.set_torque(2, tq_R)
            
            rel_ms = int((now - t_start)*1000)     # 0-based time
            ctrls[ExoSide.LEFT ].log.append((rel_ms, pct_L))
            ctrls[ExoSide.RIGHT].log.append((rel_ms, pct_R))
            
            sample_ct += 1

            for name, val in (("mtr_pos_L", posL), ("mtr_pos_R", posR), 
                              ("mtr_vel_L", velL), ("mtr_vel_R", velR), 
                              ("cmd_L", tq_L),    ("cmd_R", tq_R)):
                send_tele(name, val)

    finally:
        # ---- clean-up ---------------------------------------------
        try:
            if USE_MOCAP and mocap is not None and getattr(mocap, "client", None):
                mocap.client.sendall(b"exo off")
                mocap.client.close()
                print("▶ Sent “exo off” to Vicon")
        except Exception:
            pass
        
        # New safe shutdown built into ActuatorGroup
        # Both sides share the same ActuatorGroup, so we only need to shut down one
        ctrls[ExoSide.RIGHT].motors.disable_actuators()

        # saves estimated phase percent (%) - useful for ground truth comparisons
        if USE_MOCAP:
            save_root = Path(__file__).parent
            for side, c in ctrls.items():
                fn = save_root / f"gpe_log_{side.name.lower()}.csv"
                np.savetxt(fn, c.log,
                        fmt="%.0f,%.4f",
                        header="time_ms,phase_pct", comments="")
                print(f"✓ saved {len(c.log):,} samples → {fn}")

            print(f"Expected {EXPECTED_SAMPLES} | logged {sample_ct}")

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
