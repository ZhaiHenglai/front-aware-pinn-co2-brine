"""
GPU-adaptive paper-ablation script for 2D brine-CO2 two-phase flow PINN.

M0-M7 study for publication-oriented ablation:
M0 : minimal SingleNet PINN baseline
M1 : M0 + TwoNet pressure/saturation decoupling
M2 : M1 + multiscale Fourier saturation features + pressure/saturation input decoupling
M3 : M2 + coarse/detail saturation branch
M4 : M3 + unified front/plume-aware supervised saturation loss
M5 : M4 + pair-wise front-gradient supervised saturation loss
M6 : M5 + residual-biased resampling (RAR-style)
M7 : M6 + delayed local finite-volume phase-mass conservation loss
     (dominant CO2 FV plus weak water-phase FV regularization)

Training-strategy switches are deliberately separated from M0-M7:
BASE      : no beta curriculum, no weak diffusion continuation
BETA      : BASE + beta curriculum
DIFF      : BASE + weak diffusion continuation
BETA_DIFF : BASE + beta curriculum + weak diffusion continuation

Excluded from the main M0-M7 path:
- single-scale Fourier as a final model;
- the previous duplicate front-weighted saturation loss;
- beta curriculum and weak diffusion continuation as structural M0-M7 increments.

Key semantics
-------------
- phase2::PhaseVolumeFraction is treated as the physical CO2 saturation Sco2 in [0, 1].
- The network output for Sco2 is bounded to [eps, 1-Sw_irr].
  This enforces the maximum mobile CO2 saturation implied by Sw_irr.
- Relative permeability still uses effective saturation built from the physical Sco2.
- Initial condition uses Sco2 = 0.0 everywhere; residual non-wetting saturation Snr = 0.0.
- Injection boundary uses Sco2 = 0.8 and total normal Darcy velocity U_in = 5.4e-5 m/s,
  where Sco2 = 0.8 corresponds to the maximum mobile CO2 saturation 1 - Sw_irr for Sw_irr = 0.2.
  The total normal Darcy velocity becomes vn_in = 1.0 in nondimensional form because U_ref = U_in.
- With Snr = 0.0, the front band Sco2 in [0.05, 0.30] lies in the mobile CO2 region.
- M7 uses delayed local FV phase-mass conservation: CO2 is the dominant term and water is weakly weighted.
- Checkpoints save both state_dict and config.
"""

# -----------------------------------------------------------------------------
# Imports and numeric setup
# -----------------------------------------------------------------------------
import os
import glob
import math
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# User controls
# -----------------------------------------------------------------------------
EXP_NAME = os.environ.get("PINN_EXP_NAME", "M7")   # choose from: M0, M1, M2, M3, M4, M5, M6, M7
TRAINING_STRATEGY = os.environ.get("PINN_TRAINING_STRATEGY", "BASE")  # choose from: BASE, BETA, DIFF, BETA_DIFF
# Leave-one-out switch for final-model ablation.
# Recommended final LOO runs use EXP_NAME="M7" and TRAINING_STRATEGY="BASE".
# Choices: NONE, PLAIN_TWONET, FRONT_PLUME, PAIRGRAD, RAR, FV, COARSE_DETAIL.
# Optional internal checks: WATER_FV, MSFF.
LEAVE_OUT = os.environ.get("PINN_LEAVE_OUT", "NONE")
SEED = int(os.environ.get("PINN_SEED", "0"))   # env-controlled for multi-seed; default 0 (backward compatible)
GPU_ID = os.environ.get("PINN_GPU_ID", "0")

# put these toggles here, as requested
FLOAT_DTYPE = "float64"          # choose from: "float64", "float32"; float64 is recommended for second-order PINN derivatives
ENABLE_LBFGS_FINE_TUNE = False      # independent user switch; set True only for optional post-Adam refinement
ENABLE_PERIODIC_CHECKPOINT = False
CHECKPOINT_EVERY = 1000          # save every N Adam iterations if periodic checkpoint is enabled

# MS Fourier feature controls (used from M2 onward)
MSFF_M_PER_SCALE = 24
MSFF_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
MSFF_BASE_SCALE = 3.0
# Examples:
#   MSFF_SCALES = (1.0, 2.0, 4.0, 16.0)
#   MSFF_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)


# M3 coarse/detail saturation branch controls
M3_COARSE_M_PER_SCALE = 24
M3_COARSE_SCALES = (1.0, 2.0, 4.0, 8.0)
M3_COARSE_BASE_SCALE = 3.0
M3_DETAIL_M_PER_SCALE = 24
M3_DETAIL_SCALES = (4.0, 8.0, 16.0, 32.0)
M3_DETAIL_BASE_SCALE = 3.0
M3_COARSE_DEPTH = 6
M3_DETAIL_DEPTH = 6
M3_DETAIL_GAIN = 1.0
M3_DETAIL_TANH = True

# Optional training-strategy beta curriculum controls
BETA_MAX = 20.0
BETA_WARM_STEPS = 5000.0

# Optional training-strategy weak diffusion continuation controls
DIFF_EPS0 = 3e-4
DIFF_EPS1 = 1e-6
DIFF_DECAY_STEPS = 8000.0

# Training loss weights (kept at top so config and training stay consistent)
W_DATA_P = 1.0*5
W_DATA_S = 2.0*10
W_INJ_FLUX = 1.0*10
W_INJ_SAT = 2.0*10
W_OUT_P = 1.0*5
# Weak outlet backflow penalty.  This is a one-sided diagnostic/regularizer, not a replacement for outlet pressure BC.
W_OUT_BACKFLOW = 0.1*10
W_NOFLOW = 0.2*10
W_IC_P = 1.0*5
W_IC_S = 2.0*10

# M7 finite-volume phase-mass-conservation controls
# M7 adds a delayed local control-volume conservation loss.
# Main setting: CO2 FV is dominant; water FV is weakly weighted as auxiliary regularization.
USE_FV_MASS_LOSS = False          # overwritten by configure_experiment(); True only for M7
W_FV_MASS_MAX = 0.30              # ramped maximum FV loss weight for M7
FV_START = 8000                   # delayed FV activation to avoid early front distortion
FV_RAMP_STEPS = 6000              # smoother FV weight ramp
FV_N_CELLS = 1024                 # local control volumes per Adam iteration
FV_H_TILDE = 1.0 / 64.0           # nondimensional local square-cell width
FV_DT_TILDE_FALLBACK = 1.0e-2     # used only if data time spacing cannot be inferred
FV_W_WATER = 0.05                 # weak water-phase FV penalty; M7 is CO2-dominant phase-mass FV; set 0.0 for CO2-only FV ablation

# CO2-displacing-brine front-band definition in physical CO2 saturation space.
# The "front band" is a saturation transition band, not a spatial coordinate band.
# For the current data, the resolved displacement front is mainly Sco2 in [0.05, 0.30]. With Snr=0.0 this whole band is mobile CO2.
CO2_FRONT_S_MIN = 0.05
CO2_FRONT_S_MAX = 0.30
CO2_FRONT_CENTER = 0.5 * (CO2_FRONT_S_MIN + CO2_FRONT_S_MAX)  # 0.175
CO2_FRONT_SIGMA = 0.075  # Gaussian band width; endpoints retain ~25% of peak weight.

# M7 weak/data-range/front-band FV controls
# Front-band weighting avoids applying the FV constraint uniformly in regions
# where saturation is essentially constant and the constraint mainly excites high-frequency detail.
USE_FV_FRONT_BAND_WEIGHT = True
FV_FRONT_CENTER = CO2_FRONT_CENTER
FV_FRONT_SIGMA = CO2_FRONT_SIGMA
FV_FRONT_WEIGHT_FLOOR = 0.05      # keeps a weak background FV constraint outside the front band

# Saturation initial-condition and injection controls (soft IC; hard-IC gate removed).
S_IC_CO2 = 0.0                    # physical initial CO2 saturation at t=0; consistent with Snr=0.0
S_INJ_CO2 = 0.8                   # physical injection-boundary CO2 saturation; equals 1 - Sw_irr for Sw_irr=0.2
FORCE_DATA_INITIAL_S_TO_S_IC = True  # make supervised t=t_min rows consistent with S_IC_CO2
# Safety guard: only force the earliest supervised time level to S_IC_CO2 if the
# earliest physical data time is genuinely the initial time.  This prevents
# corrupting the first transient output when the dataset does not include t=0.
FORCE_DATA_INITIAL_ONLY_IF_TMIN_ZERO = True
DATA_IC_TIME_ATOL_PHYS = 1.0e-10

# Time-reference and early-time controls.
# If USE_DATA_MAX_TIME_AS_T_REF is True, T_ref is overwritten after the dataset is
# loaded by max(arrays["t"]).  A_time and _T are then recomputed consistently.
USE_DATA_MAX_TIME_AS_T_REF = True
T_REF_FALLBACK = 1.0e5

# INJ_BC_T_START_TILDE is initialized here and then replaced after FV_DT_TILDE
# is inferred from the loaded data:
#   INJ_BC_T_START_TILDE  = INJ_BC_START_DT_MULTIPLIER * FV_DT_TILDE
# This avoids imposing an O(1000-10000 s) artificial early-time injection delay
# when T_ref is large or when the dataset duration is much shorter than 1e5 s.
INJ_BC_T_START_TILDE = 1.0e-2
INJ_BC_START_DT_MULTIPLIER = 1.00

USE_DATA_FRONT_LOSS = False       # overwritten by configure_experiment(); True from M4 onward
W_DATA_S_FRONT_REL = 1.0          # relative multiplier inside W_DATA_S * (...)
DATA_FRONT_CENTER = CO2_FRONT_CENTER
DATA_FRONT_SIGMA = CO2_FRONT_SIGMA

USE_DATA_PLUME_LOSS = False       # overwritten by configure_experiment(); True from M4 onward
W_DATA_S_PLUME_REL = 1.0          # relative multiplier inside W_DATA_S * (...)
DATA_PLUME_THRESHOLD = CO2_FRONT_S_MIN  # plume support begins at the observed front lower bound

# M5 pair-wise front-gradient controls.
# This is a supervised local slope constraint on the CO2 saturation front.
# It is introduced after M4 and then inherited by M6/M7.  A delayed ramp is
# used so that the model first learns the point-wise front/plume saturation
# field before the local front-slope constraint is allowed to sharpen it.
USE_DATA_PAIRWISE_FRONT_GRAD = False  # overwritten by configure_experiment(); True from M5 onward
W_DATA_S_PAIRGRAD_REL = 0.03          # maximum relative multiplier in W_DATA_S * w_pairgrad * L_pairgrad
PAIRGRAD_START = 4000                 # Adam iteration at which the pair-gradient term starts to ramp
PAIRGRAD_RAMP_STEPS = 4000            # smooth ramp length for the pair-gradient multiplier
PAIRGRAD_FRONT_CENTER = CO2_FRONT_CENTER
PAIRGRAD_FRONT_SIGMA = CO2_FRONT_SIGMA
PAIRGRAD_FRONT_WEIGHT_FLOOR = 0.01    # weak background only; avoids diluting the actual front band
PAIRGRAD_S_MIN = CO2_FRONT_S_MIN      # candidate-pair lower saturation bound; excludes near-zero tails
PAIRGRAD_S_MAX = CO2_FRONT_S_MAX      # candidate-pair upper saturation bound; excludes plateau/core regions
PAIRGRAD_N_TIME_SLICES = 8            # time slices sampled per Adam iteration
PAIRGRAD_POINTS_PER_TIME = 256        # candidate points per selected time slice
PAIRGRAD_MAX_PAIRS = 2048             # upper bound on local pairs per iteration
PAIRGRAD_MIN_DIST_TILDE = 1.0e-3      # avoids excessive finite-difference amplification at tiny distances
PAIRGRAD_MAX_DIST_TILDE = 4.0 * FV_H_TILDE  # local distance scale tied to FV cell width (=0.0625 by default)
PAIRGRAD_EPS_DIST = 1.0e-6
PAIRGRAD_GRAD_SCALE = 30.0            # expected front slope scale in nondimensional coordinates

# dataset
DATA_PT_CANDIDATES = [
    "<LOCAL_PATH>s_cache_0_723_step1.pt",
    "<LOCAL_PATH>sst/PINN3/tables_cache_tensor.pt",
    "tables_cache.pt",
]
DATA_GLOB = "./tables/*.csv"
EOS_FIT_SAMPLE = 2_000_000

# training length
MAX_ADAM_ITERS = 20000

# Residual-biased resampling controls.  RAR-style resampling is enabled only from M6 onward.
RAR_START = 3500
RAR_EVERY = 100

# experiment switches; overwritten by configure_experiment() and configure_training_strategy()
USE_RAR = False
USE_RAR_RESIDUAL = False
USE_BETA_CURRICULUM = False
USE_DIFFUSION_DECAY = False
USE_TWONET = False
USE_MSFOURIER = False
USE_LBFGS = False
DECOUPLE_PS_INPUT = False
USE_TIME_STRATIFIED_DATA = True
USE_COARSE_DETAIL_S_BRANCH = False
USE_FV_MASS_LOSS = False
USE_DATA_FRONT_LOSS = False
USE_DATA_PLUME_LOSS = False
USE_DATA_PAIRWISE_FRONT_GRAD = False

EPS_CONST = 0.0  # BASE uses no artificial diffusion; DIFF/BETA_DIFF use diffusion continuation
BETA_CONST = 1.0
S_EPS = 1.0e-6  # numerical safety for physical Sco2 in (0,1)

# set GPU visibility before device selection
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))


# -----------------------------------------------------------------------------
# Helpers: seeds and runtime config
# -----------------------------------------------------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_dtype_from_user_choice():
    s = str(FLOAT_DTYPE).strip().lower()
    if s in ("float64", "fp64", "double"):
        return torch.float64
    if s in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported FLOAT_DTYPE={FLOAT_DTYPE!r}; use 'float64' or 'float32'.")


def _detect_runtime_config():
    """
    Build a small runtime profile that adapts the heavy training settings to the available device.

    Important choice:
    - dtype is user-controlled at the top of this file via FLOAT_DTYPE.
    - Only adapt batch sizes / RAR sizes / model width / L-BFGS usage by GPU type.
    """
    cpu_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    base = {
        "dtype": _resolve_dtype_from_user_choice(),
        "cpu_threads": max(1, cpu_threads),
    }

    if not torch.cuda.is_available():
        base.update({
            "profile": "cpu",
            "device": torch.device("cpu"),
            "gpu_name": "CPU",
            "gpu_mem_gb": 0.0,
            "gpu_cc": "n/a",
            "gpu_sms": 0,
            "width": 128,
            "N_BASE": 12000,
            "N_POOL": 36000,
            "N_CAND": 10000,
            "N_SEL_R": 7500,
            "N_SEL_G": 2500,
            "DATA_BATCH": 4096,
            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 50,
            "INJ_ARC_N": 1200,
            "OUTLET_ARC_N": 1200,
            "OUTER_BC_N": 500,
            "INIT_N": 3000,
        })
        return base

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    mem_gb = props.total_memory / (1024 ** 3)
    cc = f"{props.major}.{props.minor}"
    gpu_name_l = gpu_name.lower()

    if "a100" in gpu_name_l:
        base.update({
            "profile": "v100",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 160,
            "N_BASE": 45000,
            "N_POOL": 120000,
            "N_CAND": 32000,
            "N_SEL_R": 24000,
            "N_SEL_G": 8000,
            "DATA_BATCH": 24576,
            "USE_LBFGS": True,
            "LBFGS_MAX_ITER": 1400,
            "LBFGS_HISTORY_SIZE": 40,
            "INJ_ARC_N": 2600,
            "OUTLET_ARC_N": 2600,
            "OUTER_BC_N": 1100,
            "INIT_N": 9000,
        })
        return base
   
    if "v100" in gpu_name_l:
        base.update({
            "profile": "v100",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,

            "width": 160,
            "N_BASE": 38000,
            "N_POOL": 120000,
            "N_CAND": 30000,
            "N_SEL_R": 22500,
            "N_SEL_G": 7500,
            "DATA_BATCH": 24576,

            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 30,

            "INJ_ARC_N": 2600,
            "OUTLET_ARC_N": 2600,
            "OUTER_BC_N": 1100,
            "INIT_N": 9000,
        })
        return base



    if "v100_160_24G" in gpu_name_l:
        base.update({
            "profile": "v100",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,

            "width": 160,
            "N_BASE": 32000,
            "N_POOL": 110000,
            "N_CAND": 24000,
            "N_SEL_R": 18000,
            "N_SEL_G": 6000,
            "DATA_BATCH": 20480,

            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 30,

            "INJ_ARC_N": 2400,
            "OUTLET_ARC_N": 2400,
            "OUTER_BC_N": 1000,
            "INIT_N": 8000,
        })
        return base

    if "v100_144" in gpu_name_l:
        base.update({
            "profile": "v100",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,

        # mildly reduced from the original V100 profile
            "width": 144,
            "N_BASE": 28000,
            "N_POOL": 90000,
            "N_CAND": 20000,
            "N_SEL_R": 15000,
            "N_SEL_G": 5000,
            "DATA_BATCH": 16384,

        # disable runtime L-BFGS on V100 for these heavy RAR/FV (M6/M7) settings
           "USE_LBFGS": False,
           "LBFGS_MAX_ITER": 0,
           "LBFGS_HISTORY_SIZE": 30,

           "INJ_ARC_N": 2200,
           "OUTLET_ARC_N": 2200,
           "OUTER_BC_N": 900,
           "INIT_N": 7000,
        })
        return base



    if "2080 ti" in gpu_name_l or ("rtx" in gpu_name_l and mem_gb <= 12.5):
        base.update({
            "profile": "2080ti",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 96,
            "N_BASE": 8000,
            "N_POOL": 24000,
            "N_CAND": 6000,
            "N_SEL_R": 4500,
            "N_SEL_G": 1500,
            "DATA_BATCH": 2048,
            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 30,
            "INJ_ARC_N": 800,
            "OUTLET_ARC_N": 800,
            "OUTER_BC_N": 300,
            "INIT_N": 2000,
        })
        return base

    if "4060 ti" in gpu_name_l:
        base.update({
            "profile": "4060ti",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 128,
            "N_BASE": 20000,
            "N_POOL": 60000,
            "N_CAND": 16000,
            "N_SEL_R": 12000,
            "N_SEL_G": 4000,
            "DATA_BATCH": 8192,
            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 1600,
            "LBFGS_HISTORY_SIZE": 50,
            "INJ_ARC_N": 1500,
            "OUTLET_ARC_N": 1500,
            "OUTER_BC_N": 600,
            "INIT_N": 4000,
        })
        return base

    if mem_gb >= 30.0:
        base.update({
            "profile": "gpu_32gb_like",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 128,
            "N_BASE": 18000,
            "N_POOL": 54000,
            "N_CAND": 14000,
            "N_SEL_R": 10500,
            "N_SEL_G": 3500,
            "DATA_BATCH": 6144,
            "USE_LBFGS": True,
            "LBFGS_MAX_ITER": 1200,
            "LBFGS_HISTORY_SIZE": 50,
            "INJ_ARC_N": 1400,
            "OUTLET_ARC_N": 1400,
            "OUTER_BC_N": 600,
            "INIT_N": 3500,
        })
        return base

    if mem_gb >= 20.0:
        base.update({
            "profile": "gpu_24gb_like",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 128,
            "N_BASE": 14000,
            "N_POOL": 42000,
            "N_CAND": 11000,
            "N_SEL_R": 8250,
            "N_SEL_G": 2750,
            "DATA_BATCH": 4096,
            "USE_LBFGS": True,
            "LBFGS_MAX_ITER": 900,
            "LBFGS_HISTORY_SIZE": 50,
            "INJ_ARC_N": 1200,
            "OUTLET_ARC_N": 1200,
            "OUTER_BC_N": 500,
            "INIT_N": 3000,
        })
        return base

    if mem_gb >= 10.0:
        base.update({
            "profile": "gpu_12gb_like",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 96,
            "N_BASE": 9000,
            "N_POOL": 27000,
            "N_CAND": 7000,
            "N_SEL_R": 5250,
            "N_SEL_G": 1750,
            "DATA_BATCH": 2048,
            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 30,
            "INJ_ARC_N": 900,
            "OUTLET_ARC_N": 900,
            "OUTER_BC_N": 350,
            "INIT_N": 2200,
        })
        return base

    base.update({
        "profile": "gpu_small",
        "device": torch.device("cuda"),
        "gpu_name": gpu_name,
        "gpu_mem_gb": mem_gb,
        "gpu_cc": cc,
        "gpu_sms": props.multi_processor_count,
        "width": 64,
        "N_BASE": 5000,
        "N_POOL": 15000,
        "N_CAND": 4000,
        "N_SEL_R": 3000,
        "N_SEL_G": 1000,
        "DATA_BATCH": 1024,
        "USE_LBFGS": False,
        "LBFGS_MAX_ITER": 0,
        "LBFGS_HISTORY_SIZE": 20,
        "INJ_ARC_N": 600,
        "OUTLET_ARC_N": 600,
        "OUTER_BC_N": 250,
        "INIT_N": 1500,
    })
    return base


def configure_experiment(exp_name: str):
    """Configure the paper-oriented M0-M7 structural ablation.

    Training schedules such as beta curriculum and weak diffusion continuation are
    intentionally controlled by configure_training_strategy(), not by M0-M7.
    """
    global USE_RAR, USE_RAR_RESIDUAL
    global USE_BETA_CURRICULUM, USE_DIFFUSION_DECAY
    global USE_TWONET, USE_MSFOURIER, USE_LBFGS
    global DECOUPLE_PS_INPUT, USE_TIME_STRATIFIED_DATA, USE_COARSE_DETAIL_S_BRANCH
    global USE_FV_MASS_LOSS, USE_DATA_FRONT_LOSS, USE_DATA_PLUME_LOSS
    global USE_DATA_PAIRWISE_FRONT_GRAD
    global EPS_CONST, BETA_CONST

    exp_name = str(exp_name).upper().strip()

    USE_RAR = False
    USE_RAR_RESIDUAL = False
    USE_BETA_CURRICULUM = False
    USE_DIFFUSION_DECAY = False
    USE_TWONET = False
    USE_MSFOURIER = False
    USE_LBFGS = bool(ENABLE_LBFGS_FINE_TUNE)
    DECOUPLE_PS_INPUT = False
    USE_TIME_STRATIFIED_DATA = True
    USE_COARSE_DETAIL_S_BRANCH = False
    USE_FV_MASS_LOSS = False
    USE_DATA_FRONT_LOSS = False
    USE_DATA_PLUME_LOSS = False
    USE_DATA_PAIRWISE_FRONT_GRAD = False

    EPS_CONST = 0.0
    BETA_CONST = 1.0

    if exp_name == "M0":
        pass
    elif exp_name == "M1":
        USE_TWONET = True
    elif exp_name == "M2":
        USE_TWONET = True
        USE_MSFOURIER = True
        DECOUPLE_PS_INPUT = True
    elif exp_name == "M3":
        USE_TWONET = True
        USE_MSFOURIER = True
        DECOUPLE_PS_INPUT = True
        USE_COARSE_DETAIL_S_BRANCH = True
    elif exp_name == "M4":
        USE_TWONET = True
        USE_MSFOURIER = True
        DECOUPLE_PS_INPUT = True
        USE_COARSE_DETAIL_S_BRANCH = True
        USE_DATA_FRONT_LOSS = True
        USE_DATA_PLUME_LOSS = True
    elif exp_name == "M5":
        USE_TWONET = True
        USE_MSFOURIER = True
        DECOUPLE_PS_INPUT = True
        USE_COARSE_DETAIL_S_BRANCH = True
        USE_DATA_FRONT_LOSS = True
        USE_DATA_PLUME_LOSS = True
        USE_DATA_PAIRWISE_FRONT_GRAD = True
    elif exp_name == "M6":
        USE_TWONET = True
        USE_MSFOURIER = True
        DECOUPLE_PS_INPUT = True
        USE_COARSE_DETAIL_S_BRANCH = True
        USE_DATA_FRONT_LOSS = True
        USE_DATA_PLUME_LOSS = True
        USE_DATA_PAIRWISE_FRONT_GRAD = True
        USE_RAR = True
        USE_RAR_RESIDUAL = True
    elif exp_name == "M7":
        USE_TWONET = True
        USE_MSFOURIER = True
        DECOUPLE_PS_INPUT = True
        USE_COARSE_DETAIL_S_BRANCH = True
        USE_DATA_FRONT_LOSS = True
        USE_DATA_PLUME_LOSS = True
        USE_DATA_PAIRWISE_FRONT_GRAD = True
        USE_RAR = True
        USE_RAR_RESIDUAL = True
        USE_FV_MASS_LOSS = True
    else:
        raise ValueError(f"Unknown EXP_NAME: {exp_name}. Use one of M0-M7.")

def configure_training_strategy(strategy: str):
    """
    Configure optional training schedules independently from the M0-M7 structural ablation.

    BASE:
        no beta curriculum and no weak diffusion continuation.

    BETA:
        beta curriculum only.

    DIFF:
        weak diffusion continuation only.

    BETA_DIFF:
        beta curriculum + weak diffusion continuation.

    This separation is intentional: M7 must isolate the FV effect. Training
    schedules should be evaluated through T0-T3 style auxiliary experiments,
    not mixed into the main M0-M7 sequence.
    """
    global USE_BETA_CURRICULUM, USE_DIFFUSION_DECAY

    st = str(strategy).upper().strip()
    USE_BETA_CURRICULUM = False
    USE_DIFFUSION_DECAY = False

    if st in ("BASE", "T0", "NONE", "OFF"):
        pass
    elif st in ("BETA", "T1"):
        USE_BETA_CURRICULUM = True
    elif st in ("DIFF", "DIFFUSION", "T2"):
        USE_DIFFUSION_DECAY = True
    elif st in ("BETA_DIFF", "BETA+DIFF", "BETA_DIFFUSION", "T3"):
        USE_BETA_CURRICULUM = True
        USE_DIFFUSION_DECAY = True
    else:
        raise ValueError(
            f"Unknown TRAINING_STRATEGY={strategy!r}; use BASE, BETA, DIFF, or BETA_DIFF."
        )


def configure_leave_one_out(name: str):
    """Apply final-model leave-one-out ablations after M0-M7 and training strategy are configured.

    Intended use:
        EXP_NAME = "M7"
        TRAINING_STRATEGY = "BASE"
        LEAVE_OUT in {NONE, PLAIN_TWONET, FRONT_PLUME, PAIRGRAD, RAR, FV, COARSE_DETAIL}

    This function applies named final-model ablations and architecture comparison
    variants after the base experiment is configured.  Each run should be
    trained from scratch, not evaluated by toggling a trained checkpoint.
    """
    global USE_DATA_FRONT_LOSS, USE_DATA_PLUME_LOSS
    global USE_DATA_PAIRWISE_FRONT_GRAD
    global USE_RAR, USE_RAR_RESIDUAL
    global USE_FV_MASS_LOSS, FV_W_WATER
    global USE_COARSE_DETAIL_S_BRANCH
    global USE_TWONET, USE_MSFOURIER, DECOUPLE_PS_INPUT

    loo = str(name).upper().strip()
    if loo in ("", "NONE", "OFF", "NO", "FALSE"):
        return

    if loo in ("PLAIN_TWONET", "PLAIN_TWO_NET", "PLAINTWONET", "PLAIN_TWO", "PLAIN"):
        # Architecture-baseline LOO (replaces the former hard-IC ablation):
        # keep separate pressure and saturation networks, but remove the Fourier
        # feature maps and the coarse/detail saturation branch.  Physics, data,
        # boundary, initial-condition, RAR, and FV losses remain as configured by
        # the selected experiment.
        USE_COARSE_DETAIL_S_BRANCH = False
        USE_TWONET = True
        USE_MSFOURIER = False
        DECOUPLE_PS_INPUT = False

    elif loo in ("FRONT_PLUME", "FRONTPLUME", "FRONT", "PLUME"):
        # Remove M4 point-wise front/plume weighting.  Pairwise front-gradient
        # remains active, so this tests whether local slope supervision can
        # replace point-wise front/plume supervision.
        USE_DATA_FRONT_LOSS = False
        USE_DATA_PLUME_LOSS = False

    elif loo in ("PAIRGRAD", "PAIRWISE", "FRONT_GRAD", "FRONTGRAD"):
        # Remove M5 pair-wise front-gradient supervision.
        USE_DATA_PAIRWISE_FRONT_GRAD = False

    elif loo == "RAR":
        # Remove M6 residual-biased resampling.
        USE_RAR = False
        USE_RAR_RESIDUAL = False

    elif loo == "FV":
        # Remove the entire M7 FV training loss: CO2 FV, weak-water FV, and
        # front-band FV weighting all become inactive because w_fv=0.
        # This is the correct LOO for testing whether local FV conservation is
        # necessary.  It is not the same as setting FV_W_WATER=0.
        USE_FV_MASS_LOSS = False

    elif loo in ("COARSE_DETAIL", "COARSEDETAIL", "CD"):
        # Remove the M3 coarse/detail saturation decomposition while keeping
        # TwoNet, MS Fourier, and p/S input decoupling.
        USE_COARSE_DETAIL_S_BRANCH = False
        USE_TWONET = True
        USE_MSFOURIER = True
        DECOUPLE_PS_INPUT = True

    elif loo == "WATER_FV":
        # Optional FV-internal ablation: retain CO2 FV, remove weak water FV.
        # This is not L5; it tests CO2-only FV vs CO2+weak-water FV.
        FV_W_WATER = 0.0

    elif loo == "MSFF":
        # Optional architecture stress test.  Not one of the core L1-L6 runs.
        USE_MSFOURIER = False

    else:
        raise ValueError(
            f"Unknown LEAVE_OUT={name!r}; use NONE, PLAIN_TWONET, FRONT_PLUME, "
            "PAIRGRAD, RAR, FV, COARSE_DETAIL, WATER_FV, or MSFF."
        )


RUNTIME = _detect_runtime_config()
DEVICE = RUNTIME["device"]
DTYPE = RUNTIME["dtype"]

torch.set_default_dtype(DTYPE)
torch.set_num_threads(RUNTIME["cpu_threads"])
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


def print_runtime_report():
    print(
        f"[Runtime] EXP={EXP_NAME} | strategy={TRAINING_STRATEGY} | leave_out={LEAVE_OUT} | profile={RUNTIME['profile']} | "
        f"device={DEVICE} | dtype={DTYPE} | cpu_threads={RUNTIME['cpu_threads']} | "
        f"visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )
    print(
        f"[UserSwitch] FLOAT_DTYPE={FLOAT_DTYPE} | ENABLE_PERIODIC_CHECKPOINT={ENABLE_PERIODIC_CHECKPOINT} | "
        f"CHECKPOINT_EVERY={CHECKPOINT_EVERY} | ENABLE_LBFGS_FINE_TUNE={ENABLE_LBFGS_FINE_TUNE}"
    )

    if DEVICE.type == "cuda":
        print(
            f"[GPU] name={RUNTIME.get('gpu_name', 'unknown')} | total_mem={RUNTIME.get('gpu_mem_gb', 0.0):.1f} GB | "
            f"compute_capability={RUNTIME.get('gpu_cc', 'unknown')} | SMs={RUNTIME.get('gpu_sms', 'unknown')} | "
            f"cuda_version={torch.version.cuda}"
        )
    else:
        print("[GPU] CUDA not available; running on CPU.")

    print(
        f"[TrainConfig] width={RUNTIME['width']} | DATA_BATCH={RUNTIME['DATA_BATCH']} | "
        f"N_BASE={RUNTIME['N_BASE']} | N_POOL={RUNTIME['N_POOL']} | N_CAND={RUNTIME['N_CAND']} | "
        f"N_SEL_R={RUNTIME['N_SEL_R']} | USE_LBFGS_RUNTIME={RUNTIME['USE_LBFGS']} | "
        f"LBFGS_MAX_ITER={RUNTIME['LBFGS_MAX_ITER']}"
    )
    print(
        f"[ExpSwitches] twonet={USE_TWONET} | ms_fourier={USE_MSFOURIER} | "
        f"decouple_ps_input={DECOUPLE_PS_INPUT} | coarse_detail_s={USE_COARSE_DETAIL_S_BRANCH} | "
        f"front_loss={USE_DATA_FRONT_LOSS} | plume_loss={USE_DATA_PLUME_LOSS} | "
        f"pairgrad={USE_DATA_PAIRWISE_FRONT_GRAD} | RAR={USE_RAR} | FV={USE_FV_MASS_LOSS} | "
        f"beta_curr={USE_BETA_CURRICULUM} | "
        f"diff_decay={USE_DIFFUSION_DECAY}"
    )
    denom_relperm = float(1.0 - Sw_irr - Snr)
    if denom_relperm <= 0.0:
        raise ValueError(
            f"Invalid residual saturations: Sw_irr={Sw_irr}, Snr={Snr}; "
            "1 - Sw_irr - Snr must be positive."
        )
    s_front_mid = float(CO2_FRONT_CENTER)
    se_c_front_mid = max(0.0, min(1.0, (s_front_mid - float(Snr)) / denom_relperm))
    krc_front_mid = float(krc0) * (se_c_front_mid ** float(nc))
    se_c_front_min = max(0.0, min(1.0, (float(CO2_FRONT_S_MIN) - float(Snr)) / denom_relperm))
    se_c_front_max = max(0.0, min(1.0, (float(CO2_FRONT_S_MAX) - float(Snr)) / denom_relperm))
    print(
        f"[Physical] L_ref={L_ref:.6e} m | T_ref={T_ref:.6e} s | U_ref={U_ref:.6e} m/s | "
        f"P_ref={P_ref:.6e} Pa | A_time={A_time:.6e} | K={K:.3e} m2 | phi={phi:.3f}"
    )
    print(
        f"[PressureBC] p0={p0:.6e} Pa | p_out={p_out:.6e} Pa | p_out_tilde={p_out_tilde:.6e} | "
        f"U_in={U_in:.6e} m/s => vn_in_tilde=1.0"
    )
    se_c_inj = max(0.0, min(1.0, (float(S_INJ_CO2) - float(Snr)) / denom_relperm))
    se_w_inj = max(0.0, min(1.0, ((1.0 - float(S_INJ_CO2)) - float(Sw_irr)) / denom_relperm))
    print(
        f"[InjectionBC] S_INJ_CO2={S_INJ_CO2:.6f} | 1-Sw_irr={1.0 - Sw_irr:.6f} | "
        f"S_CO2_MAX={S_CO2_MAX:.6f} | Se_c(inj)={se_c_inj:.3f} | Se_w(inj)={se_w_inj:.3f}"
    )
    print(
        f"[RelPerm] Sw_irr={Sw_irr:.6f} | Snr={Snr:.6f} | mobile_denominator={denom_relperm:.6f} | "
        f"nw={nw:.3f} nc={nc:.3f} | krw0={krw0:.3f} krc0={krc0:.3f} | "
        f"Se_c(front_min/mid/max)={se_c_front_min:.3f}/{se_c_front_mid:.3f}/{se_c_front_max:.3f} | "
        f"krc(front_mid)={krc_front_mid:.3e}"
    )
    print(
        f"[SaturationIC] S_IC_CO2={S_IC_CO2:.6f} | S_EPS={S_EPS:.1e} | "
        f"network_Sco2_range=[{S_EPS:.1e},{S_CO2_MAX:.6f}] | "
        f"FORCE_DATA_INITIAL_S_TO_S_IC={FORCE_DATA_INITIAL_S_TO_S_IC} | "
        f"FORCE_ONLY_IF_TMIN_ZERO={FORCE_DATA_INITIAL_ONLY_IF_TMIN_ZERO} | "
        f"DATA_IC_TIME_ATOL_PHYS={DATA_IC_TIME_ATOL_PHYS:.1e} s | "
        f"INJ_BC_T_START_TILDE={INJ_BC_T_START_TILDE:.6e}"
    )
    try:
        print(
            f"[DataICReport] forced={DATA_IC_FORCED} | rows={DATA_IC_FORCED_ROWS} | "
            f"reason={DATA_IC_FORCE_REASON}"
        )
    except NameError:
        pass
    print(
        f"[FrontBand] common_range=[{CO2_FRONT_S_MIN:.3f},{CO2_FRONT_S_MAX:.3f}] | "
        f"common_center={CO2_FRONT_CENTER:.3f} sigma={CO2_FRONT_SIGMA:.3f} | "
        f"data_front=({DATA_FRONT_CENTER:.3f},{DATA_FRONT_SIGMA:.3f}) | "
        f"plume_threshold={DATA_PLUME_THRESHOLD:.3f} | "
        f"pairgrad_range=[{PAIRGRAD_S_MIN:.3f},{PAIRGRAD_S_MAX:.3f}] | "
        f"FV_front=({FV_FRONT_CENTER:.3f},{FV_FRONT_SIGMA:.3f})"
    )
    print(
        f"[LossWeights] dataP={W_DATA_P:.3e} dataS={W_DATA_S:.3e} | "
        f"injFlux={W_INJ_FLUX:.3e} injSat={W_INJ_SAT:.3e} | "
        f"outP={W_OUT_P:.3e} outBackflow={W_OUT_BACKFLOW:.3e} noflow={W_NOFLOW:.3e} | "
        f"icP={W_IC_P:.3e} icS={W_IC_S:.3e} | pairgrad_rel_max={W_DATA_S_PAIRGRAD_REL:.3e}"
    )
    print(
        f"[SamplingSizes] INJ_ARC_N={RUNTIME['INJ_ARC_N']} | OUTLET_ARC_N={RUNTIME['OUTLET_ARC_N']} | "
        f"OUTER_BC_N_per_side={RUNTIME['OUTER_BC_N']} | INIT_N={RUNTIME['INIT_N']}"
    )
    print(
        f"[RAR-resampling] enabled={USE_RAR and USE_RAR_RESIDUAL} | start={RAR_START} | every={RAR_EVERY} | "
        f"N_CAND={RUNTIME['N_CAND']} | N_SEL_R={RUNTIME['N_SEL_R']} | "
        f"select_ratio={RUNTIME['N_SEL_R'] / max(1, RUNTIME['N_CAND']):.3f}"
    )
    try:
        print(
            f"[DataTime] t_tilde_range=[{T_DATA_MIN_TILDE:.6e},{T_DATA_MAX_TILDE:.6e}] | "
            f"FV_DT_TILDE={FV_DT_TILDE:.6e}"
        )
        print(
            f"[DataTimePhys] t_phys_range=[{DATA_TIME_MIN_PHYS:.6e},{DATA_TIME_MAX_PHYS:.6e}] s | "
            f"span={DATA_TIME_SPAN_PHYS:.6e} s | T_ref={T_ref:.6e} s"
        )
    except NameError:
        pass

    if USE_TWONET and (not USE_MSFOURIER) and (not USE_COARSE_DETAIL_S_BRANCH):
        print("[Architecture] PlainTwoNet: separate pressure/saturation MLPs with raw coordinate inputs; no Fourier feature maps and no coarse/detail branch.")
    if USE_MSFOURIER:
        print(f"[MSFF] m_per_scale={MSFF_M_PER_SCALE} | scales={MSFF_SCALES} | base_scale={MSFF_BASE_SCALE}")
    if USE_COARSE_DETAIL_S_BRANCH:
        print(
            f"[M3-coarse-detail] coarse_scales={M3_COARSE_SCALES} coarse_m={M3_COARSE_M_PER_SCALE} "
            f"coarse_depth={M3_COARSE_DEPTH} | detail_scales={M3_DETAIL_SCALES} "
            f"detail_m={M3_DETAIL_M_PER_SCALE} detail_depth={M3_DETAIL_DEPTH} | "
            f"detail_gain={M3_DETAIL_GAIN} detail_tanh={M3_DETAIL_TANH}"
        )
    if USE_BETA_CURRICULUM:
        print(f"[Training-beta] beta_max={BETA_MAX} | warm_steps={BETA_WARM_STEPS}")
    if USE_DIFFUSION_DECAY:
        print(f"[Training-diffusion] eps0={DIFF_EPS0:.1e} | eps1={DIFF_EPS1:.1e} | decay_steps={DIFF_DECAY_STEPS}")
    if USE_DATA_PAIRWISE_FRONT_GRAD:
        print(
            f"[M5-pairgrad] W_rel_max={W_DATA_S_PAIRGRAD_REL:.3e} | start={PAIRGRAD_START} | "
            f"ramp={PAIRGRAD_RAMP_STEPS} | n_times={PAIRGRAD_N_TIME_SLICES} | "
            f"points_per_time={PAIRGRAD_POINTS_PER_TIME} | max_pairs={PAIRGRAD_MAX_PAIRS} | "
            f"dist=[{PAIRGRAD_MIN_DIST_TILDE:.1e}, {PAIRGRAD_MAX_DIST_TILDE:.3e}] | "
            f"front_center={PAIRGRAD_FRONT_CENTER:.3f} sigma={PAIRGRAD_FRONT_SIGMA:.3f} | "
            f"S_range=[{PAIRGRAD_S_MIN:.3f}, {PAIRGRAD_S_MAX:.3f}] | floor={PAIRGRAD_FRONT_WEIGHT_FLOOR:.3f} | "
            f"grad_scale={PAIRGRAD_GRAD_SCALE:.3e}"
        )
    if USE_FV_MASS_LOSS:
        fv_type = "CO2-only" if float(FV_W_WATER) == 0.0 else "CO2-dominant + weak-water phase-mass"
        print(f"[FVType] {fv_type} | FV_W_WATER={FV_W_WATER:.3e}")
        print(
            f"[M7-FV] W_max={W_FV_MASS_MAX:.3e} | start={FV_START} | ramp={FV_RAMP_STEPS} | "
            f"N_cells={FV_N_CELLS} | h_tilde={FV_H_TILDE:.3e} | water_weight={FV_W_WATER:.3e} | "
            f"front_band={USE_FV_FRONT_BAND_WEIGHT} center={FV_FRONT_CENTER:.3f} "
            f"sigma={FV_FRONT_SIGMA:.3f} floor={FV_FRONT_WEIGHT_FLOOR:.3f}"
        )


def get_gpu_memory_report():
    if DEVICE.type != "cuda":
        return "gpu_mem=n/a"
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserv = torch.cuda.memory_reserved() / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_reserv = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return (
        f"gpu_mem alloc={alloc:.2f}GB reserv={reserv:.2f}GB "
        f"max_alloc={max_alloc:.2f}GB max_reserv={max_reserv:.2f}GB"
    )


# -----------------------------------------------------------------------------
# Physical parameters and nondimensionalization
# -----------------------------------------------------------------------------
L_ref = 5.0
# Initial fallback only. If USE_DATA_MAX_TIME_AS_T_REF=True, this is overwritten
# after the dataset is loaded, before t_all, FV_DT_TILDE, and all training
# samplers are constructed.
T_ref = float(T_REF_FALLBACK)
K = 1.0e-14
phi = 0.2

rho_w_const = 1027.61
mu_w = 2.5e-4
mu_c = 2.25e-5

U_in = 5.4e-5
p0 = 10.0e6
p_out = 10.0e6

r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

mu_ref = mu_w
U_ref = U_in
k_ref = K

P_ref = mu_ref * U_ref * L_ref / k_ref
p_out_tilde = (p_out - p0) / P_ref
A_time = L_ref / (U_ref * T_ref)

Sw_irr = 0.2
Snr = 0.0  # residual non-wetting CO2 saturation; Sco2 is mobile for any positive saturation
S_CO2_MAX = 1.0 - Sw_irr  # physical upper bound for CO2 saturation; keeps Sw >= Sw_irr
krw0 = 1.0
krc0 = 1.0
nw = 2.0
nc = 2.0

# Geometry constants used by rejection sampling, boundary sampling, and FV cells.
# These are physical coordinates; network inputs are nondimensionalized by L_ref/T_ref.
_L = float(L_ref)
_T = float(T_ref)
_R = float(r_well)
_inj_cx, _inj_cy = float(inj_center[0]), float(inj_center[1])
_out_cx, _out_cy = float(out_center[0]), float(out_center[1])
_GEOM_EPS = 1.0e-6


# -----------------------------------------------------------------------------
# Dataset loading utilities
# -----------------------------------------------------------------------------
REQUIRED_COLS = [
    "X", "Y", "phase1::Pressure", "phase1::Time",
    "phase2::PhaseVolumeFraction", "phase2::Density",
    "phase1::Density", "phase1::Viscosity_0", "phase2::Viscosity_0",
]


def read_one_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True)

    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    df["__file__"] = os.path.basename(path)
    return df


def load_all_tables(glob_pattern: str):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    dfs = [read_one_table(f) for f in files]
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    return df_all, files


def _pick_dataset_path():
    for p in DATA_PT_CANDIDATES:
        if os.path.exists(p):
            return p
    envp = os.environ.get("PINN_DATA_PT", "").strip()
    if envp and os.path.exists(envp):
        return envp
    return None


def load_binary_cache(path: str):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or "arrays" not in pack:
        raise ValueError(f"{path} is not a valid dataset pack (missing 'arrays').")

    arrays = pack["arrays"]
    needed = ["x", "y", "t", "p", "Sco2", "rho_c", "rho_w", "mu_w", "mu_c"]
    missing = [k for k in needed if k not in arrays]
    if missing:
        raise ValueError(f"{path} is missing array keys: {missing}")

    file_id_to_name = pack.get("file_id_to_name", None)
    N = int(pack.get("N", len(arrays["x"])))

    time_unique = pack.get("time_unique", None)
    if time_unique is None:
        t = arrays["t"]
        if isinstance(t, torch.Tensor):
            time_unique = torch.unique(t.detach().cpu()).numpy()
        else:
            time_unique = np.unique(np.asarray(t))

    if isinstance(time_unique, torch.Tensor):
        time_unique = time_unique.detach().cpu().numpy()

    return arrays, file_id_to_name, N, time_unique


def load_dataset():
    path = _pick_dataset_path()
    if path is not None:
        arrays, file_id_to_name, N, time_unique = load_binary_cache(path)
        print(f"Loaded binary dataset: {os.path.abspath(path)} (N={N})")
        print("Unique times:", int(np.asarray(time_unique).shape[0]))
        return arrays, file_id_to_name, N, time_unique

    df, files = load_all_tables(DATA_GLOB)
    print(f"Loaded {len(files)} CSV files, total rows: {len(df)}")
    time_unique = np.unique(df["phase1::Time"].to_numpy(np.float32))
    print("Unique times:", int(time_unique.shape[0]))

    arr = {
        "x": df["X"].to_numpy(np.float32),
        "y": df["Y"].to_numpy(np.float32),
        "t": df["phase1::Time"].to_numpy(np.float32),
        "p": df["phase1::Pressure"].to_numpy(np.float32),
        "Sco2": df["phase2::PhaseVolumeFraction"].to_numpy(np.float32),
        "rho_c": df["phase2::Density"].to_numpy(np.float32),
        "rho_w": df["phase1::Density"].to_numpy(np.float32),
        "mu_w": df["phase1::Viscosity_0"].to_numpy(np.float32),
        "mu_c": df["phase2::Viscosity_0"].to_numpy(np.float32),
    }

    files_cat = df["__file__"].astype("category")
    arr["file_id"] = files_cat.cat.codes.to_numpy(np.int32)
    file_id_to_name = list(files_cat.cat.categories)
    return arr, file_id_to_name, int(len(df)), time_unique


arrays, file_id_to_name, N_rows, time_unique = load_dataset()


def _to_numpy_float64_1d(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().view(-1).to(torch.float64).numpy()
    return np.asarray(v, dtype=np.float64).reshape(-1)


def _configure_time_reference_from_loaded_data():
    """Set T_ref from the loaded physical data time and recompute dependent scales.

    T_ref affects every nondimensional time variable t_t=t/T_ref and the
    accumulation coefficient A_time=L_ref/(U_ref*T_ref).  Therefore, changing
    T_ref requires recomputing A_time and _T before t_all and all samplers are
    used.
    """
    global T_ref, A_time, _T

    t_phys = _to_numpy_float64_1d(arrays["t"])
    t_phys = t_phys[np.isfinite(t_phys)]
    if t_phys.size == 0:
        raise ValueError("Loaded dataset has no finite time values in arrays['t'].")

    t_min_phys = float(np.min(t_phys))
    t_max_phys = float(np.max(t_phys))
    if USE_DATA_MAX_TIME_AS_T_REF:
        if not (t_max_phys > 0.0):
            raise ValueError(
                f"USE_DATA_MAX_TIME_AS_T_REF=True but max data time is {t_max_phys:.6e}. "
                "Use a positive physical time range or set USE_DATA_MAX_TIME_AS_T_REF=False."
            )
        T_ref = t_max_phys
    else:
        T_ref = float(T_REF_FALLBACK)

    A_time = L_ref / (U_ref * T_ref)
    _T = float(T_ref)

    print(
        f"[TimeRef] USE_DATA_MAX_TIME_AS_T_REF={USE_DATA_MAX_TIME_AS_T_REF} | "
        f"data_time_range=[{t_min_phys:.6e}, {t_max_phys:.6e}] s | "
        f"T_ref={T_ref:.6e} s | A_time={A_time:.6e}"
    )


_configure_time_reference_from_loaded_data()

# Physical data-time diagnostics reused by DataIC safety checks, logging, and checkpoints.
_DATA_TIME_PHYS_ALL = _to_numpy_float64_1d(arrays["t"])
_DATA_TIME_PHYS_FINITE = _DATA_TIME_PHYS_ALL[np.isfinite(_DATA_TIME_PHYS_ALL)]
if _DATA_TIME_PHYS_FINITE.size == 0:
    raise ValueError("Loaded dataset has no finite physical time values in arrays['t'].")
DATA_TIME_MIN_PHYS = float(np.min(_DATA_TIME_PHYS_FINITE))
DATA_TIME_MAX_PHYS = float(np.max(_DATA_TIME_PHYS_FINITE))
DATA_TIME_SPAN_PHYS = float(DATA_TIME_MAX_PHYS - DATA_TIME_MIN_PHYS)


def _median(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().to(torch.float64).median().item())
    return float(np.median(np.asarray(x)))


print("Median brine rho, mu:", _median(arrays["rho_w"]), _median(arrays["mu_w"]))
print("Median CO2  rho, mu:", _median(arrays["rho_c"]), _median(arrays["mu_c"]))


# -----------------------------------------------------------------------------
# EOS (rho_co2(p)) fitting
# -----------------------------------------------------------------------------
def _sample_pair_numpy(p, rho, n_sample: int, seed: int = 0):
    if isinstance(p, torch.Tensor):
        n = int(p.numel())
    else:
        n = int(np.asarray(p).size)

    n_eff = min(n, int(n_sample))

    if isinstance(p, torch.Tensor):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randint(0, n, (n_eff,), generator=g, device="cpu", dtype=torch.int64)
        p_s = p.detach().cpu().view(-1).index_select(0, idx).to(torch.float64).numpy()
        rho_s = rho.detach().cpu().view(-1).index_select(0, idx).to(torch.float64).numpy()
        return p_s, rho_s

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n_eff)
    p_np = np.asarray(p).reshape(-1).astype(np.float64, copy=False)
    rho_np = np.asarray(rho).reshape(-1).astype(np.float64, copy=False)
    return p_np[idx], rho_np[idx]


def compress_by_pressure(p_pa: np.ndarray, rho: np.ndarray):
    dtmp = pd.DataFrame({"p": p_pa, "rho": rho})
    g = dtmp.groupby("p", as_index=False)["rho"].mean().sort_values("p")
    p_u = g["p"].to_numpy(dtype=np.float64)
    rho_u = g["rho"].to_numpy(dtype=np.float64)
    return p_u, rho_u


def fit_rho_cheb(p_pa: np.ndarray, rho: np.ndarray,
                 deg_list=(10, 12, 14, 16, 18, 20),
                 n_val=20000, seed=0):
    rng = np.random.default_rng(seed)

    p_pa = np.asarray(p_pa, dtype=np.float64).reshape(-1)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    valid = np.isfinite(p_pa) & np.isfinite(rho) & (rho > 0.0)
    if int(valid.sum()) < 2:
        raise ValueError(
            f"EOS fit needs at least two finite pressure-density samples with positive density; got {int(valid.sum())}."
        )
    if int(valid.sum()) < int(p_pa.size):
        print(
            f"[EOSCheck] discarded {int(p_pa.size - valid.sum())} non-finite or non-positive samples before Chebyshev fit."
        )
    p_pa = p_pa[valid]
    rho = rho[valid]

    p_u, rho_u = compress_by_pressure(p_pa, rho)
    if int(p_u.size) < 2:
        raise ValueError(
            f"EOS fit needs at least two distinct pressure values; got {int(p_u.size)}."
        )
    pmin = float(np.min(p_u))
    pmax = float(np.max(p_u))
    pmid = 0.5 * (pmin + pmax)
    prng = 0.5 * (pmax - pmin)
    if (not np.isfinite(prng)) or prng <= 0.0:
        raise ValueError(
            f"Pressure range is zero or invalid for Chebyshev EOS: pmin={pmin:.6e}, pmax={pmax:.6e}."
        )

    rho_ref = float(np.interp(pmid, p_u, rho_u))
    if (not np.isfinite(rho_ref)) or rho_ref <= 0.0:
        raise ValueError(f"Invalid EOS reference density rho_ref={rho_ref:.6e}.")
    rho_tilde = rho_u / rho_ref
    z = (p_u - pmid) / prng

    pv = rng.uniform(pmin, pmax, size=n_val)
    rho_true = np.interp(pv, p_u, rho_u)
    zv = (pv - pmid) / prng

    best = None
    for deg in deg_list:
        coeffs = np.polynomial.chebyshev.chebfit(z, rho_tilde, deg=deg)
        rho_pred = rho_ref * np.polynomial.chebyshev.chebval(zv, coeffs)
        rel = np.abs((rho_pred - rho_true) / rho_true)
        max_rel = float(np.max(rel))
        rms_rel = float(np.sqrt(np.mean(rel ** 2)))
        if best is None or max_rel < best[0]:
            best = (max_rel, rms_rel, deg, coeffs, rho_ref, pmin, pmax)

    max_rel, rms_rel, deg, coeffs, rho_ref, pmin, pmax = best
    print(
        f"[EOS] Cheb deg={deg}, max_rel={max_rel:.3e}, rms_rel={rms_rel:.3e}, "
        f"range=[{pmin/1e6:.3f},{pmax/1e6:.3f}] MPa (fit sample={len(p_pa)})"
    )
    return coeffs, rho_ref, pmin, pmax


p_s, rho_c_s = _sample_pair_numpy(arrays["p"], arrays["rho_c"], EOS_FIT_SAMPLE, seed=SEED)
cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax = fit_rho_cheb(p_s, rho_c_s)


class ChebRhoCO2(nn.Module):
    def __init__(self, coeffs_np, rho_ref, pmin, pmax, eps=1e-12):
        super().__init__()
        self.register_buffer("c", torch.tensor(coeffs_np, dtype=DTYPE))
        self.rho_ref = float(rho_ref)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.eps = float(eps)
        self.pmid = 0.5 * (self.pmin + self.pmax)
        self.prng = 0.5 * (self.pmax - self.pmin)

    def _chebval_clenshaw(self, z):
        c = self.c
        if c.numel() == 1:
            return c[0] + 0.0 * z
        b1 = torch.zeros_like(z)
        b2 = torch.zeros_like(z)
        for a in torch.flip(c[1:], dims=[0]):
            b0 = 2.0 * z * b1 - b2 + a
            b2 = b1
            b1 = b0
        return z * b1 - b2 + c[0]

    def forward(self, p_pa):
        p_pa = p_pa.to(dtype=DTYPE)
        z = (p_pa - self.pmid) / self.prng
        z = torch.clamp(z, -1.0 + self.eps, 1.0 - self.eps)
        rho_tilde = self._chebval_clenshaw(z)
        return self.rho_ref * rho_tilde


rho_co2_model = ChebRhoCO2(cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax).to(DEVICE)


# -----------------------------------------------------------------------------
# Relative permeability model
# -----------------------------------------------------------------------------
def relperm_from_Sco2(Sco2: torch.Tensor):
    Sw = 1.0 - Sco2
    denom = 1.0 - Sw_irr - Snr
    denom_t = torch.tensor(denom, device=Sco2.device, dtype=Sco2.dtype)

    Se_w = (Sw - Sw_irr) / denom_t
    Se_c = (Sco2 - Snr) / denom_t
    Se_w = torch.clamp(Se_w, 0.0, 1.0)
    Se_c = torch.clamp(Se_c, 0.0, 1.0)

    krw = krw0 * Se_w ** nw
    krc = krc0 * Se_c ** nc
    return krw, krc


# -----------------------------------------------------------------------------
# Neural networks
# -----------------------------------------------------------------------------
class MultiScaleFourierFeatures(nn.Module):
    def __init__(self, in_dim=3, m_per_scale=MSFF_M_PER_SCALE, scales=MSFF_SCALES, base_scale=MSFF_BASE_SCALE):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        B_list = []
        for s in self.scales:
            B_list.append(torch.randn(in_dim, m_per_scale, dtype=DTYPE) * (base_scale * s))
        self.register_buffer("B", torch.cat(B_list, dim=1))

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, x):
        x = x.to(device=self.B.device, dtype=self.B.dtype)
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        param = next(self.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        return self.net(x)



class SingleNetPINN(nn.Module):
    def __init__(self, width=128, use_msfourier=False):
        super().__init__()
        self.use_msfourier = bool(use_msfourier)
        if self.use_msfourier:
            self.ff = MultiScaleFourierFeatures(3, m_per_scale=MSFF_M_PER_SCALE, scales=MSFF_SCALES, base_scale=MSFF_BASE_SCALE)
            in_dim = self.ff.out_dim
        else:
            self.ff = None
            in_dim = 3

        self.backbone = MLP(in_dim, 2, width=width, depth=8)
        self.beta = 1.0

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Z = self.ff(X) if self.use_msfourier else X
        out = self.backbone(Z)
        p_tilde = out[:, 0:1]
        s_logit = out[:, 1:2]

        S_hat = torch.sigmoid(self.beta * s_logit)
        Sco2 = S_EPS + (float(S_CO2_MAX) - S_EPS) * S_hat
        return p_tilde, Sco2


class TwoNetPINN(nn.Module):
    def __init__(self, width=128, use_msfourier=False, decouple_ps_input=False):
        super().__init__()
        self.use_msfourier = bool(use_msfourier)
        self.decouple_ps_input = bool(decouple_ps_input)

        self.p_ff = None
        self.s_ff = None

        if self.use_msfourier:
            self.s_ff = MultiScaleFourierFeatures(3, m_per_scale=MSFF_M_PER_SCALE, scales=MSFF_SCALES, base_scale=MSFF_BASE_SCALE)
            s_in_dim = self.s_ff.out_dim
        else:
            s_in_dim = 3

        if self.decouple_ps_input:
            p_in_dim = 3
        else:
            self.p_ff = self.s_ff
            p_in_dim = s_in_dim

        self.pnet = MLP(p_in_dim, 1, width=width, depth=6)
        self.snet = MLP(s_in_dim, 1, width=width, depth=8)
        self.beta = 1.0

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward_pressure(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)
        Xp = self.p_ff(X) if self.p_ff is not None else X
        return self.pnet(Xp)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Xp = self.p_ff(X) if self.p_ff is not None else X
        Xs = self.s_ff(X) if self.s_ff is not None else X

        p_tilde = self.pnet(Xp)
        s_logit = self.snet(Xs)

        S_hat = torch.sigmoid(self.beta * s_logit)
        Sco2 = S_EPS + (float(S_CO2_MAX) - S_EPS) * S_hat
        return p_tilde, Sco2


class TwoNetPINNCoarseDetail(nn.Module):
    """M3-M7 saturation architecture with a coarse/detail decomposition.

    Pressure follows the M2 decoupled-input design. Saturation is represented by
    a low-frequency coarse branch plus a high-frequency Fourier-detail branch:

        s_logit = s_coarse + detail_gain * detail_term

    Pair-wise front-gradient supervision is implemented as an external supervised
    loss from M5 onward; it does not alter the network parameterization.
    """
    def __init__(self, width=128, decouple_ps_input=True):
        super().__init__()
        self.decouple_ps_input = bool(decouple_ps_input)
        self.beta = 1.0
        self.detail_tanh = bool(M3_DETAIL_TANH)
        self.detail_gain = float(M3_DETAIL_GAIN)

        self.p_ff = None
        if self.decouple_ps_input:
            p_in_dim = 3
        else:
            self.p_ff = MultiScaleFourierFeatures(3, m_per_scale=MSFF_M_PER_SCALE, scales=MSFF_SCALES, base_scale=MSFF_BASE_SCALE)
            p_in_dim = self.p_ff.out_dim

        self.s_coarse_ff = MultiScaleFourierFeatures(
            3,
            m_per_scale=M3_COARSE_M_PER_SCALE,
            scales=M3_COARSE_SCALES,
            base_scale=M3_COARSE_BASE_SCALE,
        )
        self.s_detail_ff = MultiScaleFourierFeatures(
            3,
            m_per_scale=M3_DETAIL_M_PER_SCALE,
            scales=M3_DETAIL_SCALES,
            base_scale=M3_DETAIL_BASE_SCALE,
        )

        self.pnet = MLP(p_in_dim, 1, width=width, depth=6)
        self.snet_coarse = MLP(self.s_coarse_ff.out_dim, 1, width=width, depth=M3_COARSE_DEPTH)
        self.snet_detail = MLP(self.s_detail_ff.out_dim, 1, width=width, depth=M3_DETAIL_DEPTH)

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward_pressure(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)
        Xp = self.p_ff(X) if self.p_ff is not None else X
        return self.pnet(Xp)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Xp = self.p_ff(X) if self.p_ff is not None else X
        Zc = self.s_coarse_ff(X)
        Zd = self.s_detail_ff(X)

        p_tilde = self.pnet(Xp)
        s_coarse = self.snet_coarse(Zc)
        s_detail = self.snet_detail(Zd)
        s_detail_term = torch.tanh(s_detail) if self.detail_tanh else s_detail

        s_logit = s_coarse + self.detail_gain * s_detail_term
        S_hat = torch.sigmoid(self.beta * s_logit)
        Sco2 = S_EPS + (float(S_CO2_MAX) - S_EPS) * S_hat
        return p_tilde, Sco2


def pressure_ic_loss(model, x0: torch.Tensor, y0: torch.Tensor, t0: torch.Tensor):
    """Pressure initial-condition helper.

    Evaluates only the pressure field at t=0.  TwoNet-style models expose
    forward_pressure(), which avoids an unnecessary saturation-branch forward
    pass for INIT_N points. SingleNet falls back to the normal forward()
    because pressure and saturation share one backbone.
    """
    if hasattr(model, "forward_pressure"):
        p_t_0 = model.forward_pressure(x0, y0, t0)
    else:
        p_t_0, _ = model(x0, y0, t0)
    return (p_t_0 ** 2).mean()


# -----------------------------------------------------------------------------
# Autograd helpers
# -----------------------------------------------------------------------------
def grad(u, x, create_graph=True, retain_graph=True):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]


def divergence(fx, fy, x, y):
    return grad(fx, x) + grad(fy, y)


def laplacian(u, x, y):
    ux = grad(u, x)
    uy = grad(u, y)
    uxx = grad(ux, x)
    uyy = grad(uy, y)
    return uxx + uyy


# -----------------------------------------------------------------------------
# PDE residual
# -----------------------------------------------------------------------------
def pde_residual(model, x_t, y_t, t_t, eps=0.0):
    p_tilde, Sco2 = model(x_t, y_t, t_t)
    Sw = 1.0 - Sco2

    p_phys = p0 + P_ref * p_tilde

    rho_c = rho_co2_model(p_phys)
    rho_w = torch.full_like(rho_c, rho_w_const)

    rho_c_tilde = rho_c / rho_c_ref
    rho_w_tilde = rho_w / rho_w_const

    krw, krc = relperm_from_Sco2(Sco2)

    dpdx = grad(p_tilde, x_t)
    dpdy = grad(p_tilde, y_t)

    K_tilde = K / k_ref

    vwx = -K_tilde * (mu_ref / mu_w) * krw * dpdx
    vwy = -K_tilde * (mu_ref / mu_w) * krw * dpdy
    vcx = -K_tilde * (mu_ref / mu_c) * krc * dpdx
    vcy = -K_tilde * (mu_ref / mu_c) * krc * dpdy

    d_rhoS_w_dt = grad(rho_w_tilde * Sw, t_t)
    d_rhoS_c_dt = grad(rho_c_tilde * Sco2, t_t)

    div_w = divergence(rho_w_tilde * vwx, rho_w_tilde * vwy, x_t, y_t)
    div_c = divergence(rho_c_tilde * vcx, rho_c_tilde * vcy, x_t, y_t)

    r_w = phi * A_time * d_rhoS_w_dt + div_w
    r_c = phi * A_time * d_rhoS_c_dt + div_c

    if eps is not None and float(eps) > 0.0:
        diff = float(eps) * laplacian(Sco2, x_t, y_t)
        r_c = r_c - diff
        r_w = r_w + diff

    vtx = vwx + vcx
    vty = vwy + vcy

    return r_w, r_c, p_tilde, Sco2, vtx, vty, p_phys


def total_velocity(model, x_t, y_t, t_t):
    """
    Cheaper helper for boundary conditions.
    Only computes p_tilde, Sco2, and total Darcy velocity components.
    No mass-balance residuals, no time derivative, no Laplacian.
    """
    p_tilde, Sco2 = model(x_t, y_t, t_t)

    krw, krc = relperm_from_Sco2(Sco2)
    dpdx = grad(p_tilde, x_t)
    dpdy = grad(p_tilde, y_t)

    K_tilde = K / k_ref
    vwx = -K_tilde * (mu_ref / mu_w) * krw * dpdx
    vwy = -K_tilde * (mu_ref / mu_w) * krw * dpdy
    vcx = -K_tilde * (mu_ref / mu_c) * krc * dpdx
    vcy = -K_tilde * (mu_ref / mu_c) * krc * dpdy

    vtx = vwx + vcx
    vty = vwy + vcy
    return p_tilde, Sco2, vtx, vty


def phase_mass_fluxes(model, x_t, y_t, t_t, include_diffusion=False, eps=0.0):
    """
    Return nondimensional phase mass fluxes used by the M7 local finite-volume phase-mass loss.

    Fw = rho_w_tilde * v_w
    Fc = rho_c_tilde * v_c

    Diffusion is disabled for the default M7 FV term because the optional weak diffusion
    term is a stabilization schedule rather than a physical diffusion model.
    """
    p_tilde, Sco2 = model(x_t, y_t, t_t)
    p_phys = p0 + P_ref * p_tilde

    rho_c = rho_co2_model(p_phys)
    rho_w = torch.full_like(rho_c, rho_w_const)

    rho_c_tilde = rho_c / rho_c_ref
    rho_w_tilde = rho_w / rho_w_const

    krw, krc = relperm_from_Sco2(Sco2)

    dpdx = grad(p_tilde, x_t)
    dpdy = grad(p_tilde, y_t)

    K_tilde = K / k_ref
    vwx = -K_tilde * (mu_ref / mu_w) * krw * dpdx
    vwy = -K_tilde * (mu_ref / mu_w) * krw * dpdy
    vcx = -K_tilde * (mu_ref / mu_c) * krc * dpdx
    vcy = -K_tilde * (mu_ref / mu_c) * krc * dpdy

    Fwx = rho_w_tilde * vwx
    Fwy = rho_w_tilde * vwy
    Fcx = rho_c_tilde * vcx
    Fcy = rho_c_tilde * vcy

    if include_diffusion and eps is not None and float(eps) > 0.0:
        dSdx = grad(Sco2, x_t)
        dSdy = grad(Sco2, y_t)
        Fcx = Fcx - float(eps) * dSdx
        Fcy = Fcy - float(eps) * dSdy
        Fwx = Fwx + float(eps) * dSdx
        Fwy = Fwy + float(eps) * dSdy

    return Fwx, Fwy, Fcx, Fcy, p_tilde, Sco2, rho_w_tilde, rho_c_tilde


# -----------------------------------------------------------------------------
# Supervised data batching
# -----------------------------------------------------------------------------
def _as_cpu_float32_col(v):
    if isinstance(v, torch.Tensor):
        t = v.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        return t.view(-1, 1)
    a = np.asarray(v)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return torch.from_numpy(a).view(-1, 1)


x_all = _as_cpu_float32_col(arrays["x"]) / float(L_ref)
y_all = _as_cpu_float32_col(arrays["y"]) / float(L_ref)
t_all = _as_cpu_float32_col(arrays["t"]) / float(T_ref)

p_all = _as_cpu_float32_col(arrays["p"])
Sco2_raw_all = _as_cpu_float32_col(arrays["Sco2"])

DATA_IC_FORCED = False
DATA_IC_FORCED_ROWS = 0
DATA_IC_FORCE_REASON = "disabled"

if FORCE_DATA_INITIAL_S_TO_S_IC:
    # Keep point-wise supervised data consistent with the prescribed initial
    # saturation only when the earliest physical data time is truly the initial
    # time.  Without this guard, the first transient output level would be
    # corrupted if the dataset does not contain t=0.
    allow_force_ic = (
        (not bool(FORCE_DATA_INITIAL_ONLY_IF_TMIN_ZERO))
        or abs(float(DATA_TIME_MIN_PHYS)) <= float(DATA_IC_TIME_ATOL_PHYS)
    )

    if allow_force_ic:
        t_min_loaded = float(t_all.min().item())
        ic_mask = torch.isclose(
            t_all.view(-1),
            torch.tensor(t_min_loaded, dtype=t_all.dtype),
            rtol=0.0,
            atol=1.0e-12,
        )
        n_ic_rows = int(ic_mask.sum().item())
        if n_ic_rows > 0:
            Sco2_raw_all = Sco2_raw_all.clone()
            Sco2_raw_all[ic_mask.view(-1), 0] = float(S_IC_CO2)
            DATA_IC_FORCED = True
            DATA_IC_FORCED_ROWS = n_ic_rows
            DATA_IC_FORCE_REASON = "forced earliest time level because physical t_min is zero or guard disabled"
            print(
                f"[DataIC] forced {n_ic_rows} rows at "
                f"t_min_tilde={t_min_loaded:.6e}, t_min_phys={DATA_TIME_MIN_PHYS:.6e} s "
                f"to Sco2={S_IC_CO2:.6f}"
            )
    else:
        DATA_IC_FORCE_REASON = "skipped because earliest physical data time is not zero"
        print(
            f"[DataIC] skipped forcing earliest data rows: "
            f"t_min_phys={DATA_TIME_MIN_PHYS:.6e} s exceeds "
            f"DATA_IC_TIME_ATOL_PHYS={DATA_IC_TIME_ATOL_PHYS:.1e}. "
            f"Set FORCE_DATA_INITIAL_ONLY_IF_TMIN_ZERO=False to override."
        )

def _report_and_build_saturation_training_targets(S_raw: torch.Tensor):
    """Validate physical CO2 saturation data and build bounded training targets.

    Network outputs are bounded to [S_EPS, S_CO2_MAX], so point-wise supervised
    saturation targets must use the same physical upper bound.  The raw data are
    not overwritten; this helper reports all out-of-range values and returns the
    clipped target tensor used by data and pair-gradient losses.
    """
    v = S_raw.detach().cpu().view(-1).to(torch.float64)
    finite = torch.isfinite(v)
    n_total = int(v.numel())
    n_nonfinite = int((~finite).sum().item())
    if n_nonfinite > 0:
        raise ValueError(f"Sco2 data contains {n_nonfinite}/{n_total} non-finite values.")

    n_below0 = int((v < 0.0).sum().item())
    n_above1 = int((v > 1.0).sum().item())
    n_above_phys = int((v > float(S_CO2_MAX)).sum().item())
    v_min = float(v.min().item())
    v_max = float(v.max().item())
    v_mean = float(v.mean().item())

    print(
        f"[DataSaturationCheck] raw_after_IC=[{v_min:.6e},{v_max:.6e}] mean={v_mean:.6e} | "
        f"allowed_math=[0,1] below0={n_below0} above1={n_above1} | "
        f"allowed_physical=[0,{float(S_CO2_MAX):.6f}] above_physical={n_above_phys}"
    )
    if n_below0 > 0 or n_above1 > 0 or n_above_phys > 0:
        print(
            f"[DataSaturationClamp] supervised targets and pair-gradient data will be clipped to "
            f"[0,{float(S_CO2_MAX):.6f}] to match the network output bound."
        )

    return torch.clamp(S_raw, 0.0, float(S_CO2_MAX))


Sco2_train_all = _report_and_build_saturation_training_targets(Sco2_raw_all)

N_data = int(x_all.shape[0])

# ============================================================================
# Optional training-side label holdout  (label-efficiency / generalization study)
# ----------------------------------------------------------------------------
# When PINN_DATA_HOLDOUT=1, a deterministic subset of the supervised (p, Sco2)
# rows is REMOVED from the data-fitting loss, so evaluate_heldout_and_front_timing
# can report genuinely out-of-sample accuracy on the SAME held-out rows. The split
# is computed by Results/heldout_split.py (the single source of truth shared with
# the evaluator); a self-contained replica keeps this script runnable if that
# module is not importable. PDE / BC / IC / FV collocation is sampled fresh in the
# domain (sample_interior, rejection sampling) and is therefore UNAFFECTED — only
# the labelled data are thinned. Default (PINN_DATA_HOLDOUT=0) reproduces the old
# in-sample behaviour exactly.
#
#   random_fraction : remove a random fraction of points (robustness to sparse labels)
#   time_holdout    : remove whole time levels (temporal generalisation; the front
#                     must be propagated through unobserved times by the physics)
# Both are invariant to the x/L_ref nondimensionalisation, so the train-side split
# matches the evaluator's split bit-for-bit for the same (mode, seed, fraction,
# hold_every, hold_start).
# ============================================================================
def _training_keep_indices(x_np, y_np, t_np, time_unique_np, spec):
    """Integer indices of supervised rows to KEEP (train side of the held-out
    split). Prefers Results/heldout_split.py; falls back to an inline replica for
    random_fraction / time_holdout if that module cannot be imported."""
    try:
        import sys as _sys
        _rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
        if _rd not in _sys.path:
            _sys.path.insert(0, _rd)
        import heldout_split as _HS
        return np.asarray(_HS.training_keep_indices(x_np, y_np, t_np, time_unique_np, spec), dtype=np.int64)
    except Exception as _e:
        print(f"[DataHoldout] heldout_split import failed ({_e!r}); using inline replica.")
        mode = str(spec.get("mode", "none")).lower()
        N = int(np.asarray(x_np).reshape(-1).shape[0])
        test = np.zeros(N, dtype=bool)
        if mode == "random_fraction":
            frac = float(spec.get("fraction", 0.2))
            if not (0.0 <= frac < 1.0):
                raise ValueError(f"fraction must be in [0,1); got {frac}")
            test = np.random.default_rng(int(spec.get("seed", 0))).random(N) < frac
        elif mode == "time_holdout":
            t = np.asarray(t_np, dtype=np.float64).reshape(-1)
            tu = np.unique(np.asarray(time_unique_np, dtype=np.float64).reshape(-1)); tu.sort()
            Tn = int(tu.shape[0])
            pos = np.clip(np.searchsorted(tu, t), 0, Tn - 1)
            prev = np.clip(pos - 1, 0, Tn - 1)
            time_id = np.where(np.abs(t - tu[prev]) < np.abs(t - tu[pos]), prev, pos)
            held = spec.get("hold_time_indices", None)
            if held is None:
                held = list(range(int(spec.get("hold_start", 2)), Tn, max(1, int(spec.get("hold_every", 5)))))
            held = sorted(set(int(i if i >= 0 else Tn + i) for i in held if -Tn <= int(i) < Tn))
            test = np.isin(time_id, np.asarray(held, dtype=np.int64))
        else:
            raise ValueError(f"inline holdout replica supports random_fraction/time_holdout only; got {mode!r}")
        return np.nonzero(~test)[0].astype(np.int64)


DATA_HOLDOUT = os.environ.get("PINN_DATA_HOLDOUT", "0").strip().lower() in ("1", "true", "yes", "on")
HOLDOUT_SPEC = None
if DATA_HOLDOUT:
    HOLDOUT_SPEC = dict(
        mode=os.environ.get("PINN_HOLDOUT_MODE", "time_holdout").strip().lower(),
        seed=int(os.environ.get("PINN_HOLDOUT_SEED", "0")),
        fraction=float(os.environ.get("PINN_HOLDOUT_FRACTION", "0.2")),
        hold_every=int(os.environ.get("PINN_HOLDOUT_HOLD_EVERY", "5")),
        hold_start=int(os.environ.get("PINN_HOLDOUT_HOLD_START", "2")),
        r_lo=float(os.environ.get("PINN_HOLDOUT_R_LO", "1.0")),
        r_hi=float(os.environ.get("PINN_HOLDOUT_R_HI", "2.5")),
        split_x=float(os.environ.get("PINN_HOLDOUT_SPLIT_X", "2.5")),
        split_y=float(os.environ.get("PINN_HOLDOUT_SPLIT_Y", "2.5")),
    )
    _time_unique_hold = np.unique(t_all.view(-1).cpu().numpy())
    _keep_np = _training_keep_indices(
        x_all.view(-1).cpu().numpy(), y_all.view(-1).cpu().numpy(),
        t_all.view(-1).cpu().numpy(), _time_unique_hold, HOLDOUT_SPEC)
    _keep_idx = torch.from_numpy(np.ascontiguousarray(_keep_np)).to(torch.int64)
    _n_before = N_data
    x_all = x_all.index_select(0, _keep_idx)
    y_all = y_all.index_select(0, _keep_idx)
    t_all = t_all.index_select(0, _keep_idx)
    p_all = p_all.index_select(0, _keep_idx)
    Sco2_raw_all = Sco2_raw_all.index_select(0, _keep_idx)
    Sco2_train_all = Sco2_train_all.index_select(0, _keep_idx)
    N_data = int(x_all.shape[0])
    print(
        f"[DataHoldout] ENABLED mode={HOLDOUT_SPEC['mode']} | kept {N_data}/{_n_before} "
        f"({100.0 * N_data / max(1, _n_before):.1f}%) supervised rows; held out "
        f"{_n_before - N_data} for out-of-sample eval | spec={HOLDOUT_SPEC}"
    )
else:
    print("[DataHoldout] disabled (PINN_DATA_HOLDOUT=0) -> all supervised rows used (in-sample).")


def _safe_filename_token(value) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value)).strip("-")


def _default_holdout_checkpoint_tag() -> str:
    if not DATA_HOLDOUT or not HOLDOUT_SPEC:
        return ""
    mode = str(HOLDOUT_SPEC.get("mode", "holdout"))
    if mode == "time_holdout":
        return "holdout-time-he{hold_every}-hs{hold_start}".format(**HOLDOUT_SPEC)
    if mode == "random_fraction":
        frac = _safe_filename_token(f"{float(HOLDOUT_SPEC.get('fraction', 0.2)):.3g}")
        seed = int(HOLDOUT_SPEC.get("seed", 0))
        return f"holdout-rf{frac}-s{seed}"
    return "holdout-" + _safe_filename_token(mode)


CHECKPOINT_DIR = os.environ.get("PINN_CKPT_DIR", ".").strip() or "."
CHECKPOINT_TAG = os.environ.get("PINN_CKPT_TAG", "").strip() or _default_holdout_checkpoint_tag()
CHECKPOINT_TAG = _safe_filename_token(CHECKPOINT_TAG)
print(f"[checkpoint] dir={CHECKPOINT_DIR} tag={CHECKPOINT_TAG or 'none'}")


def _cpu_col_stats(t: torch.Tensor):
    v = t.detach().cpu().view(-1).to(torch.float64)
    return float(v.min().item()), float(v.max().item()), float(v.mean().item())

_x_min, _x_max, _x_mean = _cpu_col_stats(x_all)
_y_min, _y_max, _y_mean = _cpu_col_stats(y_all)
_t_min, _t_max, _t_mean = _cpu_col_stats(t_all)
_p_min, _p_max, _p_mean = _cpu_col_stats(p_all)
_s_min, _s_max, _s_mean = _cpu_col_stats(Sco2_raw_all)
_s_train_min, _s_train_max, _s_train_mean = _cpu_col_stats(Sco2_train_all)
_s_flat = Sco2_train_all.view(-1).to(torch.float32)
_front_mask_data = (_s_flat >= float(CO2_FRONT_S_MIN)) & (_s_flat <= float(CO2_FRONT_S_MAX))
_plume_mask_data = _s_flat > float(DATA_PLUME_THRESHOLD)
_mobile_mask_data = _s_flat > float(Snr)
print(
    f"[DataRange] N={N_data} | x_tilde=[{_x_min:.3e},{_x_max:.3e}] mean={_x_mean:.3e} | "
    f"y_tilde=[{_y_min:.3e},{_y_max:.3e}] mean={_y_mean:.3e} | "
    f"t_tilde=[{_t_min:.3e},{_t_max:.3e}] mean={_t_mean:.3e}"
)
print(
    f"[DataFields] p_phys=[{_p_min:.6e},{_p_max:.6e}] mean={_p_mean:.6e} | "
    f"Sco2_raw_after_IC=[{_s_min:.6e},{_s_max:.6e}] mean={_s_mean:.6e} | "
    f"Sco2_training_target=[{_s_train_min:.6e},{_s_train_max:.6e}] mean={_s_train_mean:.6e}"
)
print(
    f"[DataMasks] front_count={int(_front_mask_data.sum().item())}/{N_data} "
    f"({100.0 * float(_front_mask_data.float().mean().item()):.2f}%) | "
    f"plume_count(S>{DATA_PLUME_THRESHOLD:.3f})={int(_plume_mask_data.sum().item())}/{N_data} "
    f"({100.0 * float(_plume_mask_data.float().mean().item()):.2f}%) | "
    f"mobile_CO2_count(S>Snr={Snr:.3f})={int(_mobile_mask_data.sum().item())}/{N_data} "
    f"({100.0 * float(_mobile_mask_data.float().mean().item()):.2f}%)"
)

_t_all_np = t_all.view(-1).cpu().numpy()
_time_unique_np, _time_inv_np = np.unique(_t_all_np, return_inverse=True)
TIME_BUCKETS = []
for k in range(len(_time_unique_np)):
    bucket_idx = np.nonzero(_time_inv_np == k)[0].astype(np.int64, copy=False)
    TIME_BUCKETS.append(torch.from_numpy(bucket_idx))
N_TIME_BUCKETS = len(TIME_BUCKETS)
_bucket_sizes_np = np.asarray([int(b.numel()) for b in TIME_BUCKETS], dtype=np.int64)
print(
    f"[DataSampling] stratified_by_time={USE_TIME_STRATIFIED_DATA} | N_time_buckets={N_TIME_BUCKETS} | "
    f"bucket_size[min/median/max]={int(_bucket_sizes_np.min())}/{float(np.median(_bucket_sizes_np)):.1f}/{int(_bucket_sizes_np.max())}"
)


def _estimate_fv_dt_tilde():
    """Infer a stable nondimensional FV time interval from the loaded data."""
    if len(_time_unique_np) >= 2:
        dts = np.diff(np.sort(_time_unique_np))
        dts = dts[dts > 0.0]
        if dts.size > 0:
            return float(np.median(dts))
    return float(FV_DT_TILDE_FALLBACK)


FV_DT_TILDE = _estimate_fv_dt_tilde()
T_DATA_MIN_TILDE = float(np.min(_time_unique_np))
T_DATA_MAX_TILDE = float(np.max(_time_unique_np))

# Retune the early-time injection constraint after the data time spacing is known.
INJ_BC_T_START_TILDE = max(
    float(T_DATA_MIN_TILDE),
    float(INJ_BC_START_DT_MULTIPLIER) * float(FV_DT_TILDE),
)

print(
    f"[FV] dt_tilde={FV_DT_TILDE:.3e} | h_tilde={FV_H_TILDE:.3e} | "
    f"water_weight={FV_W_WATER:.3e} | t_range=[{T_DATA_MIN_TILDE:.3e}, {T_DATA_MAX_TILDE:.3e}] | "
    f"front_band={USE_FV_FRONT_BAND_WEIGHT} center={FV_FRONT_CENTER:.3f} sigma={FV_FRONT_SIGMA:.3f} | "
    f"common_front_range=[{CO2_FRONT_S_MIN:.3f},{CO2_FRONT_S_MAX:.3f}]"
)
print(
    f"[EarlyTime] INJ_BC_T_START_TILDE={INJ_BC_T_START_TILDE:.3e}"
)


def _sample_data_indices_uniform_over_time(batch_size: int) -> torch.Tensor:
    chosen_bins = torch.randint(0, N_TIME_BUCKETS, (batch_size,), device="cpu", dtype=torch.int64)
    counts = torch.bincount(chosen_bins, minlength=N_TIME_BUCKETS)

    parts = []
    for k, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        bucket = TIME_BUCKETS[k]
        sel = torch.randint(0, bucket.numel(), (count,), device="cpu", dtype=torch.int64)
        parts.append(bucket.index_select(0, sel))

    idx = torch.cat(parts, dim=0)
    perm = torch.randperm(idx.numel(), device="cpu")
    return idx.index_select(0, perm)


def sample_data_batch(batch_size=None):
    if batch_size is None:
        batch_size = RUNTIME["DATA_BATCH"]

    if USE_TIME_STRATIFIED_DATA:
        idx = _sample_data_indices_uniform_over_time(batch_size)
    else:
        idx = torch.randint(0, N_data, (batch_size,), device="cpu", dtype=torch.int64)

    xd = x_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)
    yd = y_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)
    td = t_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)

    p_true = p_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)
    ptd = (p_true - p0) / P_ref

    Sco2_true = Sco2_train_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)

    return xd, yd, td, ptd, Sco2_true


def saturation_data_loss(S_pred: torch.Tensor, S_true: torch.Tensor):
    """Unified point-wise supervised saturation loss for M0-M7.

    M0-M3:
        L_S = global MSE.

    M4-M7:
        L_S = global MSE + lambda_front * front-band MSE + lambda_plume * plume-support MSE.

    M5-M7 add pair-wise front-gradient supervision separately, because that term
    constrains local front slope rather than point-wise saturation value.
    """
    err2 = (S_pred - S_true) ** 2
    loss_global = err2.mean()

    if USE_DATA_FRONT_LOSS:
        w_front = torch.exp(-((S_true - DATA_FRONT_CENTER) ** 2) / (2.0 * DATA_FRONT_SIGMA ** 2))
        loss_front = (w_front * err2).sum() / (w_front.sum() + 1.0e-12)
    else:
        loss_front = torch.zeros((), device=S_pred.device, dtype=S_pred.dtype)

    if USE_DATA_PLUME_LOSS:
        w_plume = (S_true > DATA_PLUME_THRESHOLD).to(dtype=S_pred.dtype)
        loss_plume = (w_plume * err2).sum() / (w_plume.sum() + 1.0e-12)
    else:
        loss_plume = torch.zeros((), device=S_pred.device, dtype=S_pred.dtype)

    loss_total = loss_global + float(W_DATA_S_FRONT_REL) * loss_front + float(W_DATA_S_PLUME_REL) * loss_plume
    return loss_total, loss_global, loss_front, loss_plume


# Precomputed CPU-side front weights and candidate mask for M5 pair-wise
# front-gradient sampling.  The mask removes nearly constant plume tails and
# high-saturation plateau/core regions; the Gaussian weight then concentrates
# selected pairs around the nominal transition band.
_SCO2_ALL_CLAMPED_FOR_PAIRGRAD = Sco2_train_all.view(-1).to(torch.float32)
PAIRGRAD_NODE_WEIGHT_ALL = torch.exp(
    -((_SCO2_ALL_CLAMPED_FOR_PAIRGRAD - float(PAIRGRAD_FRONT_CENTER)) ** 2)
    / (2.0 * float(PAIRGRAD_FRONT_SIGMA) ** 2)
) + float(PAIRGRAD_FRONT_WEIGHT_FLOOR)
PAIRGRAD_CANDIDATE_MASK_ALL = (
    (_SCO2_ALL_CLAMPED_FOR_PAIRGRAD >= float(PAIRGRAD_S_MIN))
    & (_SCO2_ALL_CLAMPED_FOR_PAIRGRAD <= float(PAIRGRAD_S_MAX))
)
_pairgrad_candidate_count = int(PAIRGRAD_CANDIDATE_MASK_ALL.sum().item())
_pairgrad_weight_min = float(PAIRGRAD_NODE_WEIGHT_ALL.min().item())
_pairgrad_weight_max = float(PAIRGRAD_NODE_WEIGHT_ALL.max().item())
_pairgrad_weight_mean = float(PAIRGRAD_NODE_WEIGHT_ALL.mean().item())
print(
    f"[PairGradData] candidate_S_range=[{PAIRGRAD_S_MIN:.3f},{PAIRGRAD_S_MAX:.3f}] | "
    f"candidate_count={_pairgrad_candidate_count}/{N_data} ({100.0 * _pairgrad_candidate_count / max(1, N_data):.2f}%) | "
    f"node_weight[min/mean/max]={_pairgrad_weight_min:.3e}/{_pairgrad_weight_mean:.3e}/{_pairgrad_weight_max:.3e}"
)


def _zero_pairgrad_loss(device=None, dtype=None):
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE
    z = torch.zeros((), device=device, dtype=dtype)
    return z, z, z


def _weighted_sample_bucket_indices(bucket: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Sample row indices from one time bucket with front-biased probabilities."""
    n_bucket = int(bucket.numel())
    if n_bucket <= 0:
        return bucket

    candidate_mask = PAIRGRAD_CANDIDATE_MASK_ALL.index_select(0, bucket).view(-1)
    candidate_pos = torch.nonzero(candidate_mask, as_tuple=False).view(-1)

    # Prefer transition-band candidates.  If a time level has too few such
    # points, fall back to the full bucket so that the loss remains defined.
    if int(candidate_pos.numel()) >= 2:
        pool_bucket = bucket.index_select(0, candidate_pos)
    else:
        pool_bucket = bucket

    n_pool = int(pool_bucket.numel())
    if n_pool <= 0:
        return bucket[:0]
    n_eff = min(int(n_sample), n_pool)

    weights = PAIRGRAD_NODE_WEIGHT_ALL.index_select(0, pool_bucket).view(-1)
    weights_sum = float(weights.sum().item())
    if not math.isfinite(weights_sum) or weights_sum <= 0.0:
        pos = torch.randint(0, n_pool, (n_eff,), device="cpu", dtype=torch.int64)
    else:
        pos = torch.multinomial(weights, n_eff, replacement=(n_eff > n_pool))
    return pool_bucket.index_select(0, pos)


def sample_pairwise_front_gradient_batch(
    n_time_slices=PAIRGRAD_N_TIME_SLICES,
    points_per_time=PAIRGRAD_POINTS_PER_TIME,
    max_pairs=PAIRGRAD_MAX_PAIRS,
):
    """Sample same-time local point pairs for the M5 front-gradient loss.

    Pair construction is deliberately data-based rather than autograd-based:
    for each sampled time slice, front-biased points are selected from the
    supervised dataset; each point is paired with its nearest valid same-time
    neighbor inside a prescribed nondimensional distance band.
    """
    if (not USE_DATA_PAIRWISE_FRONT_GRAD) or max_pairs <= 0:
        return None
    if N_TIME_BUCKETS <= 0:
        return None

    n_time_slices = max(1, int(n_time_slices))
    points_per_time = max(2, int(points_per_time))
    max_pairs = max(1, int(max_pairs))
    pairs_per_time = max(1, int(math.ceil(max_pairs / float(n_time_slices))))

    chosen_bins = torch.randint(0, N_TIME_BUCKETS, (n_time_slices,), device="cpu", dtype=torch.int64)

    xi_parts, yi_parts, ti_parts, si_parts = [], [], [], []
    xj_parts, yj_parts, tj_parts, sj_parts = [], [], [], []
    w_parts, d_parts = [], []
    total_pairs = 0

    for k in chosen_bins.tolist():
        bucket = TIME_BUCKETS[int(k)]
        if int(bucket.numel()) < 2:
            continue

        idx_cpu = _weighted_sample_bucket_indices(bucket, points_per_time)
        if int(idx_cpu.numel()) < 2:
            continue

        x = x_all.index_select(0, idx_cpu).to(DEVICE, dtype=DTYPE)
        y = y_all.index_select(0, idx_cpu).to(DEVICE, dtype=DTYPE)
        t = t_all.index_select(0, idx_cpu).to(DEVICE, dtype=DTYPE)
        s_true = Sco2_train_all.index_select(0, idx_cpu).to(DEVICE, dtype=DTYPE)

        xy = torch.cat([x, y], dim=1)
        dist = torch.cdist(xy, xy, p=2)
        m = int(dist.shape[0])
        if m < 2:
            continue

        inf = torch.tensor(float("inf"), device=DEVICE, dtype=DTYPE)
        dist.fill_diagonal_(float("inf"))
        valid_dist = (dist >= float(PAIRGRAD_MIN_DIST_TILDE)) & (dist <= float(PAIRGRAD_MAX_DIST_TILDE))
        dist = torch.where(valid_dist, dist, inf)
        nn_dist, j_local = torch.min(dist, dim=1)
        valid = torch.isfinite(nn_dist)
        if not bool(valid.any().item()):
            continue

        i_local = torch.arange(m, device=DEVICE, dtype=torch.int64)[valid]
        j_local = j_local[valid]
        dij = nn_dist[valid].view(-1, 1)

        s_mean = 0.5 * (s_true.index_select(0, i_local) + s_true.index_select(0, j_local))
        w_pair = torch.exp(
            -((s_mean - float(PAIRGRAD_FRONT_CENTER)) ** 2)
            / (2.0 * float(PAIRGRAD_FRONT_SIGMA) ** 2)
        ) + float(PAIRGRAD_FRONT_WEIGHT_FLOOR)

        n_valid = int(i_local.numel())
        if n_valid > pairs_per_time:
            probs = w_pair.detach().view(-1)
            probs_sum = float(probs.sum().item())
            if math.isfinite(probs_sum) and probs_sum > 0.0:
                keep = torch.multinomial(probs, pairs_per_time, replacement=False)
            else:
                keep = torch.randperm(n_valid, device=DEVICE)[:pairs_per_time]
            i_local = i_local.index_select(0, keep)
            j_local = j_local.index_select(0, keep)
            dij = dij.index_select(0, keep)
            w_pair = w_pair.index_select(0, keep)

        xi_parts.append(x.index_select(0, i_local))
        yi_parts.append(y.index_select(0, i_local))
        ti_parts.append(t.index_select(0, i_local))
        si_parts.append(s_true.index_select(0, i_local))

        xj_parts.append(x.index_select(0, j_local))
        yj_parts.append(y.index_select(0, j_local))
        tj_parts.append(t.index_select(0, j_local))
        sj_parts.append(s_true.index_select(0, j_local))

        w_parts.append(w_pair)
        d_parts.append(dij)
        total_pairs += int(i_local.numel())

        if total_pairs >= max_pairs:
            break

    if total_pairs <= 0:
        return None

    xi = torch.cat(xi_parts, dim=0)[:max_pairs]
    yi = torch.cat(yi_parts, dim=0)[:max_pairs]
    ti = torch.cat(ti_parts, dim=0)[:max_pairs]
    si = torch.cat(si_parts, dim=0)[:max_pairs]

    xj = torch.cat(xj_parts, dim=0)[:max_pairs]
    yj = torch.cat(yj_parts, dim=0)[:max_pairs]
    tj = torch.cat(tj_parts, dim=0)[:max_pairs]
    sj = torch.cat(sj_parts, dim=0)[:max_pairs]

    w = torch.cat(w_parts, dim=0)[:max_pairs]
    d = torch.cat(d_parts, dim=0)[:max_pairs]

    return xi, yi, ti, si, xj, yj, tj, sj, w, d


def pairwise_front_gradient_loss_from_batch(model, pair_batch):
    """Compute normalized pair-wise front-gradient loss from a fixed pair batch."""
    if (not USE_DATA_PAIRWISE_FRONT_GRAD) or pair_batch is None:
        return _zero_pairgrad_loss()

    xi, yi, ti, si_true, xj, yj, tj, sj_true, w_pair, dist = pair_batch
    if int(xi.numel()) <= 0:
        return _zero_pairgrad_loss()

    x_pair = torch.cat([xi, xj], dim=0)
    y_pair = torch.cat([yi, yj], dim=0)
    t_pair = torch.cat([ti, tj], dim=0)

    _, s_pair = model(x_pair, y_pair, t_pair)
    n = int(xi.shape[0])
    si_pred = s_pair[:n]
    sj_pred = s_pair[n:]

    denom = dist + float(PAIRGRAD_EPS_DIST)
    grad_pred = (si_pred - sj_pred) / denom
    grad_true = (si_true - sj_true) / denom

    err2 = ((grad_pred - grad_true) / float(PAIRGRAD_GRAD_SCALE)) ** 2
    loss = (w_pair * err2).sum() / (w_pair.sum() + 1.0e-12)
    n_pairs = torch.tensor(float(n), device=DEVICE, dtype=DTYPE)
    mean_dist = (w_pair * dist).sum() / (w_pair.sum() + 1.0e-12)
    return loss, n_pairs, mean_dist


def pairgrad_weight_rel_at(it: int) -> float:
    """Return the ramped relative multiplier for the M5 pair-gradient term."""
    if not USE_DATA_PAIRWISE_FRONT_GRAD:
        return 0.0
    if it <= int(PAIRGRAD_START):
        return 0.0
    if it <= int(PAIRGRAD_START + PAIRGRAD_RAMP_STEPS):
        return float(W_DATA_S_PAIRGRAD_REL) * float(it - PAIRGRAD_START) / float(PAIRGRAD_RAMP_STEPS)
    return float(W_DATA_S_PAIRGRAD_REL)


def pairwise_front_gradient_loss(model):
    pair_batch = sample_pairwise_front_gradient_batch()
    return pairwise_front_gradient_loss_from_batch(model, pair_batch)


def _outside_well_holes_xy_phys(x_phys: torch.Tensor, y_phys: torch.Tensor) -> torch.Tensor:
    dx1 = x_phys - _inj_cx
    dy1 = y_phys - _inj_cy
    dx2 = x_phys - _out_cx
    dy2 = y_phys - _out_cy
    in_inj = (dx1 * dx1 + dy1 * dy1) < (_R * _R)
    in_out = (dx2 * dx2 + dy2 * dy2) < (_R * _R)
    return ~(in_inj | in_out).squeeze(1)


def _rejection_sample_xy(N: int, oversample: float = 1.20):
    if N <= 0:
        raise ValueError("N must be positive")

    xs = []
    ys = []
    got = 0

    while got < N:
        M = int((N - got) * oversample) + 128
        x = torch.rand(M, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _L
        y = torch.rand(M, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _L
        mask = _outside_well_holes_xy_phys(x, y)
        if mask.any():
            x_ok = x[mask]
            y_ok = y[mask]
            xs.append(x_ok)
            ys.append(y_ok)
            got += x_ok.shape[0]

    x_all_phys = torch.cat(xs, dim=0)[:N]
    y_all_phys = torch.cat(ys, dim=0)[:N]
    return x_all_phys, y_all_phys


def _fv_cell_clearance_mask_phys(x_phys: torch.Tensor, y_phys: torch.Tensor, h_phys: float) -> torch.Tensor:
    """
    Keep local square FV cells inside the rectangular domain and away from well holes.
    The extra circular margin prevents a square control volume from intersecting either well.
    """
    half = 0.5 * float(h_phys)
    margin = 0.5 * math.sqrt(2.0) * float(h_phys)

    in_box = (
        (x_phys > half) & (x_phys < (_L - half)) &
        (y_phys > half) & (y_phys < (_L - half))
    )

    dx1 = x_phys - _inj_cx
    dy1 = y_phys - _inj_cy
    dx2 = x_phys - _out_cx
    dy2 = y_phys - _out_cy

    clear_inj = (dx1 * dx1 + dy1 * dy1) > ((_R + margin) ** 2)
    clear_out = (dx2 * dx2 + dy2 * dy2) > ((_R + margin) ** 2)

    return (in_box & clear_inj & clear_out).squeeze(1)


def sample_fv_cells(N, h_tilde=FV_H_TILDE, dt_tilde=FV_DT_TILDE):
    """
    Sample point-centered local square control volumes for M7.

    Coordinates are nondimensional. The time pair (t0, t1) is sampled strictly
    inside the data-supported time interval [T_DATA_MIN_TILDE, T_DATA_MAX_TILDE].
    This prevents the FV accumulation term from imposing long-time extrapolation
    constraints outside the supervised data window.
    """
    if N <= 0:
        raise ValueError("N must be positive")

    h_phys = float(h_tilde) * _L
    dt_tilde = float(dt_tilde)

    t_min = float(T_DATA_MIN_TILDE)
    t_max = float(T_DATA_MAX_TILDE)
    t_span = t_max - t_min

    if not (dt_tilde > 0.0):
        raise ValueError(f"FV dt_tilde must be positive, got {dt_tilde}")
    if not (t_span > dt_tilde):
        raise ValueError(
            f"FV dt_tilde={dt_tilde:.6e} is too large for data time range "
            f"[{t_min:.6e}, {t_max:.6e}] with span={t_span:.6e}."
        )

    xs = []
    ys = []
    got = 0

    while got < N:
        M = int((N - got) * 1.5) + 128
        x = torch.rand(M, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _L
        y = torch.rand(M, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _L
        mask = _fv_cell_clearance_mask_phys(x, y, h_phys)
        if mask.any():
            x_ok = x[mask]
            y_ok = y[mask]
            xs.append(x_ok)
            ys.append(y_ok)
            got += x_ok.shape[0]

    x_phys = torch.cat(xs, dim=0)[:N]
    y_phys = torch.cat(ys, dim=0)[:N]

    # Sample t1 in [t_min + dt, t_max], then set t0 = t1 - dt.
    t1 = (t_min + dt_tilde) + torch.rand(
        N, 1, device=DEVICE, dtype=torch.get_default_dtype()
    ) * (t_span - dt_tilde)
    t0 = t1 - dt_tilde

    return x_phys / _L, y_phys / _L, t0, t1


def fv_mass_loss(model, N_cells=FV_N_CELLS, h_tilde=FV_H_TILDE, dt_tilde=FV_DT_TILDE, eps=0.0):
    """
    M7 local finite-volume phase-mass-conservation loss.

    CO2 FV residual is active. Water-phase FV residual is weakly weighted by
    FV_W_WATER. Set FV_W_WATER=0.0 for a CO2-only FV ablation.
    """
    xc, yc, t0, t1 = sample_fv_cells(N_cells, h_tilde=h_tilde, dt_tilde=dt_tilde)

    h = torch.tensor(float(h_tilde), device=DEVICE, dtype=torch.get_default_dtype())
    area = h * h

    # Accumulation at local cell centers: t_n -> t_{n+1}
    p0_tilde_cv, S0 = model(xc, yc, t0)
    p1_tilde_cv, S1 = model(xc, yc, t1)

    p0_phys_cv = p0 + P_ref * p0_tilde_cv
    p1_phys_cv = p0 + P_ref * p1_tilde_cv

    rho_c0_tilde = rho_co2_model(p0_phys_cv) / rho_c_ref
    rho_c1_tilde = rho_co2_model(p1_phys_cv) / rho_c_ref

    rho_w0_tilde = torch.ones_like(rho_c0_tilde)
    rho_w1_tilde = torch.ones_like(rho_c1_tilde)

    Sw0 = 1.0 - S0
    Sw1 = 1.0 - S1

    acc_c = phi * A_time * area * ((rho_c1_tilde * S1 - rho_c0_tilde * S0) / float(dt_tilde))
    acc_w = phi * A_time * area * ((rho_w1_tilde * Sw1 - rho_w0_tilde * Sw0) / float(dt_tilde))

    # Face coordinates at t_{n+1}. The coordinates must require gradients because fluxes need grad(p).
    xe = (xc + 0.5 * h).clone().detach().requires_grad_(True)
    xw = (xc - 0.5 * h).clone().detach().requires_grad_(True)
    xn = xc.clone().detach().requires_grad_(True)
    xs = xc.clone().detach().requires_grad_(True)

    ye = yc.clone().detach().requires_grad_(True)
    yw = yc.clone().detach().requires_grad_(True)
    yn = (yc + 0.5 * h).clone().detach().requires_grad_(True)
    ys = (yc - 0.5 * h).clone().detach().requires_grad_(True)

    te = t1.clone().detach()
    tw = t1.clone().detach()
    tn = t1.clone().detach()
    ts = t1.clone().detach()

    Fwx_e, Fwy_e, Fcx_e, Fcy_e, *_ = phase_mass_fluxes(model, xe, ye, te, include_diffusion=False, eps=eps)
    Fwx_w, Fwy_w, Fcx_w, Fcy_w, *_ = phase_mass_fluxes(model, xw, yw, tw, include_diffusion=False, eps=eps)
    Fwx_n, Fwy_n, Fcx_n, Fcy_n, *_ = phase_mass_fluxes(model, xn, yn, tn, include_diffusion=False, eps=eps)
    Fwx_s, Fwy_s, Fcx_s, Fcy_s, *_ = phase_mass_fluxes(model, xs, ys, ts, include_diffusion=False, eps=eps)

    # Outward flux over a square cell in nondimensional coordinates.
    flux_c = h * (Fcx_e - Fcx_w + Fcy_n - Fcy_s)
    flux_w = h * (Fwx_e - Fwx_w + Fwy_n - Fwy_s)

    R_c = acc_c + flux_c
    R_w = acc_w + flux_w

    # Area normalization makes the loss scale comparable to the strong-form PDE residual.
    r_c_fv = R_c / (area + 1e-12)
    r_w_fv = R_w / (area + 1e-12)

    if USE_FV_FRONT_BAND_WEIGHT:
        # Evaluate the front indicator at t_{n+1}. The detached weight prevents
        # the model from reducing the FV objective by artificially moving S away
        # from the selected saturation band.
        front_w = torch.exp(-((S1.detach() - float(FV_FRONT_CENTER)) ** 2) / (2.0 * float(FV_FRONT_SIGMA) ** 2))
        if FV_FRONT_WEIGHT_FLOOR is not None and float(FV_FRONT_WEIGHT_FLOOR) > 0.0:
            floor = float(FV_FRONT_WEIGHT_FLOOR)
            front_w = floor + (1.0 - floor) * front_w
        denom = front_w.sum() + 1.0e-12
        loss_c = (front_w * (r_c_fv ** 2)).sum() / denom
        loss_w = (front_w * (r_w_fv ** 2)).sum() / denom
    else:
        loss_c = (r_c_fv ** 2).mean()
        loss_w = (r_w_fv ** 2).mean()

    loss_fv = loss_c + float(FV_W_WATER) * loss_w

    return loss_fv, loss_c, loss_w


# -----------------------------------------------------------------------------
# Collocation / BC / IC sampling
# -----------------------------------------------------------------------------
def _sample_time_tilde(N, t_min_tilde=None, t_max_tilde=None, device=None, dtype=None):
    """Sample nondimensional time only inside the data-supported interval."""
    device = DEVICE if device is None else device
    dtype = torch.get_default_dtype() if dtype is None else dtype

    data_lo = float(T_DATA_MIN_TILDE)
    data_hi = float(T_DATA_MAX_TILDE)
    lo = float(data_lo if t_min_tilde is None else t_min_tilde)
    hi = float(data_hi if t_max_tilde is None else t_max_tilde)

    lo = min(max(lo, data_lo), data_hi)
    hi = min(max(hi, data_lo), data_hi)

    if hi <= lo:
        if lo >= data_hi:
            return torch.full((N, 1), data_hi, device=device, dtype=dtype)
        hi = min(data_hi, lo + max(float(FV_DT_TILDE), 1.0e-8))
    if hi <= lo:
        return torch.full((N, 1), lo, device=device, dtype=dtype)
    return lo + torch.rand(N, 1, device=device, dtype=dtype) * (hi - lo)


def sample_interior(N):
    x, y = _rejection_sample_xy(N)
    tt = _sample_time_tilde(N).requires_grad_(True)
    xt = (x / _L).requires_grad_(True)
    yt = (y / _L).requires_grad_(True)
    return xt, yt, tt


def sample_initial(N):
    x, y = _rejection_sample_xy(N)
    t = torch.zeros(N, 1, device=DEVICE, dtype=torch.get_default_dtype())
    return x / _L, y / _L, t / _T


def sample_injection_arc(N):
    theta = _GEOM_EPS + torch.rand(N, 1, device=DEVICE, dtype=torch.get_default_dtype()) * (0.5 * math.pi - 2.0 * _GEOM_EPS)
    x = _inj_cx + _R * torch.cos(theta)
    y = _inj_cy + _R * torch.sin(theta)

    # Do not sample the injection saturation boundary exactly at the initial-time
    # intersection.  At t=0 the prescribed initial condition is Sco2=S_IC_CO2
    # (0.0 in the S0 configuration), whereas the injection boundary prescribes
    # Sco2=S_INJ_CO2 after injection starts.
    t0_tilde = max(float(T_DATA_MIN_TILDE), float(INJ_BC_T_START_TILDE))
    tt = _sample_time_tilde(N, t_min_tilde=t0_tilde).requires_grad_(True)

    xt = (x / _L).requires_grad_(True)
    yt = (y / _L).requires_grad_(True)

    # Direction from the injection well into the porous domain.  This is not the
    # outward normal of the computational domain, whose sign would be reversed.
    nx = torch.cos(theta)
    ny = torch.sin(theta)
    return xt, yt, tt, nx, ny


def sample_outlet_arc(N):
    theta = math.pi + _GEOM_EPS + torch.rand(N, 1, device=DEVICE, dtype=torch.get_default_dtype()) * (0.5 * math.pi - 2.0 * _GEOM_EPS)
    x = _out_cx + _R * torch.cos(theta)
    y = _out_cy + _R * torch.sin(theta)
    tt = _sample_time_tilde(N).requires_grad_(True)

    xt = (x / _L).requires_grad_(True)
    yt = (y / _L).requires_grad_(True)

    # Direction from porous domain into the outlet well.  Positive vn_out means
    # production/outflow; negative vn_out denotes local backflow from the outlet.
    nx = -torch.cos(theta)
    ny = -torch.sin(theta)
    return xt, yt, tt, nx, ny


def sample_outer_boundary(N_each=600):
    dtype = torch.get_default_dtype()

    y1 = (_R + _GEOM_EPS) + torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - 2.0 * _GEOM_EPS)
    x1 = torch.zeros_like(y1)
    n1 = torch.tensor([-1.0, 0.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    x2 = (_R + _GEOM_EPS) + torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - 2.0 * _GEOM_EPS)
    y2 = torch.zeros_like(x2)
    n2 = torch.tensor([0.0, -1.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    y3 = torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - _GEOM_EPS)
    x3 = torch.full_like(y3, _L)
    n3 = torch.tensor([1.0, 0.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    x4 = torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - _GEOM_EPS)
    y4 = torch.full_like(x4, _L)
    n4 = torch.tensor([0.0, 1.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    x = torch.cat([x1, x2, x3, x4], dim=0)
    y = torch.cat([y1, y2, y3, y4], dim=0)
    n = torch.cat([n1, n2, n3, n4], dim=0)

    tt = _sample_time_tilde(x.shape[0], dtype=dtype).requires_grad_(True)

    xt = (x / _L).requires_grad_(True)
    yt = (y / _L).requires_grad_(True)
    nx = n[:, 0:1]
    ny = n[:, 1:2]
    return xt, yt, tt, nx, ny


# -----------------------------------------------------------------------------
# RAR
# -----------------------------------------------------------------------------
@torch.no_grad()
def _pool_points(N_pool):
    x, y = _rejection_sample_xy(N_pool)
    tt = _sample_time_tilde(N_pool)
    return x / _L, y / _L, tt


def select_rar_points(model, eps, N_pool, N_candidate, N_sel_r, N_sel_g=None):
    """Periodic residual-biased resampling for M6-M7.

    N_pool is retained for call-signature compatibility with older scripts, but
    the M-series implementation samples exactly N_candidate points. This removes
    the previous wasteful two-step pool generation followed by random subselect.
    N_sel_g is retained for compatibility and ignored in the main M0-M7 path.
    """
    if not (USE_RAR and USE_RAR_RESIDUAL):
        raise RuntimeError("select_rar_points was called but residual-biased resampling is not enabled.")
    del N_pool, N_sel_g

    xt, yt, tt = _pool_points(int(N_candidate))
    xt = xt.clone().detach().requires_grad_(True)
    yt = yt.clone().detach().requires_grad_(True)
    tt = tt.clone().detach().requires_grad_(True)

    r_w, r_c, *_ = pde_residual(model, xt, yt, tt, eps=eps)
    score_r = (r_w.abs() + r_c.abs()).detach().squeeze()
    k_r = min(int(N_sel_r), score_r.numel())
    idx = torch.topk(score_r, k=k_r, largest=True).indices

    x_sel = xt[idx].clone().detach().requires_grad_(True)
    y_sel = yt[idx].clone().detach().requires_grad_(True)
    t_sel = tt[idx].clone().detach().requires_grad_(True)

    del xt, yt, tt, r_w, r_c, score_r
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return x_sel, y_sel, t_sel


def build_ckpt_config():
    return {
        "exp_name": EXP_NAME,
        "training_strategy": str(TRAINING_STRATEGY),
        "leave_out": str(LEAVE_OUT),
        "seed": int(SEED),
        "data_holdout": (dict(HOLDOUT_SPEC) if DATA_HOLDOUT else None),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_tag": str(CHECKPOINT_TAG),
        "model_type": (
            "two_cd" if USE_COARSE_DETAIL_S_BRANCH
            else ("plain_twonet" if (USE_TWONET and not USE_MSFOURIER) else ("two" if USE_TWONET else "single"))
        ),
        "width": int(RUNTIME["width"]),
        "dtype": str(FLOAT_DTYPE),
        "runtime_profile": str(RUNTIME["profile"]),
        "max_adam_iters": int(MAX_ADAM_ITERS),
        "enable_periodic_checkpoint": bool(ENABLE_PERIODIC_CHECKPOINT),
        "checkpoint_every": int(CHECKPOINT_EVERY),
        "architecture": {
            "use_twonet": bool(USE_TWONET),
            "use_msfourier": bool(USE_MSFOURIER),
            "msff_m_per_scale": int(MSFF_M_PER_SCALE),
            "msff_scales": tuple(float(v) for v in MSFF_SCALES),
            "msff_base_scale": float(MSFF_BASE_SCALE),
            "decouple_ps_input": bool(DECOUPLE_PS_INPUT),
            "use_coarse_detail_s_branch": bool(USE_COARSE_DETAIL_S_BRANCH),
            "use_plain_twonet": bool(USE_TWONET and (not USE_MSFOURIER) and (not USE_COARSE_DETAIL_S_BRANCH)),
            "m3_coarse_detail": {
                "coarse_m_per_scale": int(M3_COARSE_M_PER_SCALE),
                "coarse_scales": tuple(float(v) for v in M3_COARSE_SCALES),
                "coarse_base_scale": float(M3_COARSE_BASE_SCALE),
                "detail_m_per_scale": int(M3_DETAIL_M_PER_SCALE),
                "detail_scales": tuple(float(v) for v in M3_DETAIL_SCALES),
                "detail_base_scale": float(M3_DETAIL_BASE_SCALE),
                "coarse_depth": int(M3_COARSE_DEPTH),
                "detail_depth": int(M3_DETAIL_DEPTH),
                "detail_gain": float(M3_DETAIL_GAIN),
                "detail_tanh": bool(M3_DETAIL_TANH),
            },
        },
        "training_schedules": {
            "use_beta_curriculum": bool(USE_BETA_CURRICULUM),
            "beta_max": float(BETA_MAX),
            "beta_warm_steps": float(BETA_WARM_STEPS),
            "use_diffusion_decay": bool(USE_DIFFUSION_DECAY),
            "diff_eps0": float(DIFF_EPS0),
            "diff_eps1": float(DIFF_EPS1),
            "diff_decay_steps": float(DIFF_DECAY_STEPS),
        },
        "saturation_ic": {
            "s_ic_co2": float(S_IC_CO2),
            "s_inj_co2": float(S_INJ_CO2),
            "s_co2_output_min": float(S_EPS),
            "s_co2_output_max": float(S_CO2_MAX),
            "force_data_initial_s_to_s_ic": bool(FORCE_DATA_INITIAL_S_TO_S_IC),
            "force_data_initial_only_if_tmin_zero": bool(FORCE_DATA_INITIAL_ONLY_IF_TMIN_ZERO),
            "data_ic_time_atol_phys": float(DATA_IC_TIME_ATOL_PHYS),
            "data_time_min_phys": float(DATA_TIME_MIN_PHYS),
            "data_time_max_phys": float(DATA_TIME_MAX_PHYS),
            "use_data_max_time_as_t_ref": bool(USE_DATA_MAX_TIME_AS_T_REF),
            "t_ref_fallback": float(T_REF_FALLBACK),
            "inj_bc_t_start_tilde": float(INJ_BC_T_START_TILDE),
            "inj_bc_start_dt_multiplier": float(INJ_BC_START_DT_MULTIPLIER),
        },
        "m4_front_plume_loss": {
            "front_loss": bool(USE_DATA_FRONT_LOSS),
            "front_s_min": float(CO2_FRONT_S_MIN),
            "front_s_max": float(CO2_FRONT_S_MAX),
            "front_center": float(DATA_FRONT_CENTER),
            "front_sigma": float(DATA_FRONT_SIGMA),
            "front_weight_rel": float(W_DATA_S_FRONT_REL),
            "plume_loss": bool(USE_DATA_PLUME_LOSS),
            "plume_threshold": float(DATA_PLUME_THRESHOLD),
            "plume_weight_rel": float(W_DATA_S_PLUME_REL),
        },
        "m5_pairwise_front_gradient": {
            "enabled": bool(USE_DATA_PAIRWISE_FRONT_GRAD),
            "w_data_s_pairgrad_rel_max": float(W_DATA_S_PAIRGRAD_REL),
            "pairgrad_start": int(PAIRGRAD_START),
            "pairgrad_ramp_steps": int(PAIRGRAD_RAMP_STEPS),
            "front_center": float(PAIRGRAD_FRONT_CENTER),
            "front_sigma": float(PAIRGRAD_FRONT_SIGMA),
            "front_weight_floor": float(PAIRGRAD_FRONT_WEIGHT_FLOOR),
            "s_min": float(PAIRGRAD_S_MIN),
            "s_max": float(PAIRGRAD_S_MAX),
            "n_time_slices": int(PAIRGRAD_N_TIME_SLICES),
            "points_per_time": int(PAIRGRAD_POINTS_PER_TIME),
            "max_pairs": int(PAIRGRAD_MAX_PAIRS),
            "min_dist_tilde": float(PAIRGRAD_MIN_DIST_TILDE),
            "max_dist_tilde": float(PAIRGRAD_MAX_DIST_TILDE),
            "eps_dist": float(PAIRGRAD_EPS_DIST),
            "grad_scale": float(PAIRGRAD_GRAD_SCALE),
        },
        "m6_rar": {
            "enabled": bool(USE_RAR),
            "type": "periodic residual-biased resampling",
            "residual_score": "abs(r_w) + abs(r_c)",
            "rar_start": int(RAR_START),
            "rar_every": int(RAR_EVERY),
            "n_candidate": int(RUNTIME["N_CAND"]),
            "n_select_residual": int(RUNTIME["N_SEL_R"]),
            "select_ratio": float(RUNTIME["N_SEL_R"] / max(1, RUNTIME["N_CAND"])),
        },
        "m7_fv_mass": {
            "enabled": bool(USE_FV_MASS_LOSS),
            "w_fv_mass_max": float(W_FV_MASS_MAX),
            "fv_start": int(FV_START),
            "fv_ramp_steps": int(FV_RAMP_STEPS),
            "fv_n_cells": int(FV_N_CELLS),
            "fv_h_tilde": float(FV_H_TILDE),
            "fv_dt_tilde": float(FV_DT_TILDE),
            "fv_w_water": float(FV_W_WATER),
            "fv_type": ("CO2-only" if float(FV_W_WATER) == 0.0 else "CO2-dominant + weak-water"),
            "use_front_band_weight": bool(USE_FV_FRONT_BAND_WEIGHT),
            "front_center": float(FV_FRONT_CENTER),
            "front_sigma": float(FV_FRONT_SIGMA),
            "front_weight_floor": float(FV_FRONT_WEIGHT_FLOOR),
        },
        "physical_parameters": {
            "L_ref": float(L_ref),
            "T_ref": float(T_ref),
            "K": float(K),
            "phi": float(phi),
            "S_ic_co2": float(S_IC_CO2),
            "S_inj_co2": float(S_INJ_CO2),
            "mu_w": float(mu_w),
            "mu_c": float(mu_c),
            "mu_ref": float(mu_ref),
            "U_in": float(U_in),
            "U_ref": float(U_ref),
            "k_ref": float(k_ref),
            "p0": float(p0),
            "p_out": float(p_out),
            "P_ref": float(P_ref),
            "A_time": float(A_time),
            "Sw_irr": float(Sw_irr),
            "Snr": float(Snr),
            "S_CO2_MAX": float(S_CO2_MAX),
            "krw0": float(krw0),
            "krc0": float(krc0),
            "nw": float(nw),
            "nc": float(nc),
        },
        "loss_weights": {
            "w_data_p": float(W_DATA_P),
            "w_data_s": float(W_DATA_S),
            "w_data_s_pairgrad_rel_max": float(W_DATA_S_PAIRGRAD_REL),
            "pairgrad_start": int(PAIRGRAD_START),
            "pairgrad_ramp_steps": int(PAIRGRAD_RAMP_STEPS),
            "w_inj_flux": float(W_INJ_FLUX),
            "w_inj_sat": float(W_INJ_SAT),
            "w_out_p": float(W_OUT_P),
            "w_out_backflow": float(W_OUT_BACKFLOW),
            "w_noflow": float(W_NOFLOW),
            "w_ic_p": float(W_IC_P),
            "w_ic_s": float(W_IC_S),
        },
    }


def save_checkpoint(model, suffix: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tag = f"_{CHECKPOINT_TAG}" if CHECKPOINT_TAG else ""
    fname = f"model_{EXP_NAME}_{TRAINING_STRATEGY}_LOO-{LEAVE_OUT}_seed{SEED}_{RUNTIME['profile']}{tag}_{suffix}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": build_ckpt_config(),
    }
    torch.save(ckpt, path)
    print(f"[checkpoint] saved: {path}")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train():
    print_runtime_report()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    width = RUNTIME["width"]
    if USE_COARSE_DETAIL_S_BRANCH:
        model = TwoNetPINNCoarseDetail(width=width, decouple_ps_input=DECOUPLE_PS_INPUT).to(DEVICE)
    elif USE_TWONET:
        model = TwoNetPINN(width=width, use_msfourier=USE_MSFOURIER, decouple_ps_input=DECOUPLE_PS_INPUT).to(DEVICE)
    else:
        model = SingleNetPINN(width=width, use_msfourier=False).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Model] class={model.__class__.__name__} | total_params={n_params_total} | "
        f"trainable_params={n_params_trainable} | adam_lr=2.0e-4"
    )

    N_BASE = RUNTIME["N_BASE"]
    N_POOL = RUNTIME["N_POOL"]
    N_CAND = RUNTIME["N_CAND"]
    N_SEL_R = RUNTIME["N_SEL_R"]
    DATA_BATCH = RUNTIME["DATA_BATCH"]
    INJ_ARC_N = RUNTIME["INJ_ARC_N"]
    OUTLET_ARC_N = RUNTIME["OUTLET_ARC_N"]
    OUTER_BC_N = RUNTIME["OUTER_BC_N"]
    INIT_N = RUNTIME["INIT_N"]

    def eps_schedule(it):
        if not USE_DIFFUSION_DECAY:
            return EPS_CONST
        if it < DIFF_DECAY_STEPS:
            s = it / DIFF_DECAY_STEPS
            return DIFF_EPS0 * (DIFF_EPS1 / DIFF_EPS0) ** s
        return DIFF_EPS1

    def beta_schedule(it):
        if not USE_BETA_CURRICULUM:
            return BETA_CONST
        return min(BETA_MAX, 1.0 + (it / BETA_WARM_STEPS) * (BETA_MAX - 1.0))

    def w_pde_schedule(it):
        if it <= 2000:
            return 0.0
        if it <= 6000:
            return float((it - 2000) / 4000.0)
        return 1.0

    def w_fv_schedule(it):
        if not USE_FV_MASS_LOSS:
            return 0.0
        if it <= FV_START:
            return 0.0
        if it <= FV_START + FV_RAMP_STEPS:
            return float(W_FV_MASS_MAX * (it - FV_START) / FV_RAMP_STEPS)
        return float(W_FV_MASS_MAX)

    for it in range(1, MAX_ADAM_ITERS + 1):
        opt.zero_grad(set_to_none=True)

        beta = beta_schedule(it)
        eps = eps_schedule(it)
        model.set_beta(beta)

        w_pde = w_pde_schedule(it)
        w_fv = w_fv_schedule(it)
        w_pairgrad = pairgrad_weight_rel_at(it)

        if w_pde > 0.0:
            use_rar_now = USE_RAR and USE_RAR_RESIDUAL and (it >= RAR_START) and (it % RAR_EVERY == 0)
            if use_rar_now:
                xf, yf, tf = select_rar_points(
                    model, eps=eps,
                    N_pool=N_POOL, N_candidate=N_CAND,
                    N_sel_r=N_SEL_R, N_sel_g=None,
                )
                rar_flag = "RAR"
            else:
                xf, yf, tf = sample_interior(N_BASE)
                rar_flag = "rnd"

            r_w, r_c, *_ = pde_residual(model, xf, yf, tf, eps=eps)
            loss_pde = (r_w ** 2).mean() + (r_c ** 2).mean()
        else:
            loss_pde = torch.zeros((), device=DEVICE, dtype=DTYPE)
            rar_flag = "off"

        if w_fv > 0.0:
            loss_fv, loss_fv_c, loss_fv_w = fv_mass_loss(
                model,
                N_cells=FV_N_CELLS,
                h_tilde=FV_H_TILDE,
                dt_tilde=FV_DT_TILDE,
                eps=eps,
            )
        else:
            loss_fv = torch.zeros((), device=DEVICE, dtype=DTYPE)
            loss_fv_c = torch.zeros((), device=DEVICE, dtype=DTYPE)
            loss_fv_w = torch.zeros((), device=DEVICE, dtype=DTYPE)

        xd, yd, td, ptd_true, Sd_true = sample_data_batch(DATA_BATCH)
        ptd_pred, Sd_pred = model(xd, yd, td)
        loss_data_p = ((ptd_pred - ptd_true) ** 2).mean()
        loss_data_s, loss_data_s_global, loss_data_s_front, loss_data_s_plume = saturation_data_loss(Sd_pred, Sd_true)
        if w_pairgrad > 0.0:
            loss_data_s_pairgrad, pairgrad_n_pairs, pairgrad_mean_dist = pairwise_front_gradient_loss(model)
        else:
            loss_data_s_pairgrad, pairgrad_n_pairs, pairgrad_mean_dist = _zero_pairgrad_loss()

        xi, yi, ti, nxi, nyi = sample_injection_arc(INJ_ARC_N)
        _, S_i, vtx_i, vty_i = total_velocity(model, xi, yi, ti)
        vn_in = vtx_i * nxi + vty_i * nyi
        loss_inj_flux = ((vn_in - 1.0) ** 2).mean()
        loss_inj_sat = ((S_i - float(S_INJ_CO2)) ** 2).mean()

        xo, yo, to, nxo, nyo = sample_outlet_arc(OUTLET_ARC_N)
        p_t_o, _, vtx_o, vty_o = total_velocity(model, xo, yo, to)
        loss_out_p = ((p_t_o - p_out_tilde) ** 2).mean()
        vn_out = vtx_o * nxo + vty_o * nyo
        loss_out_backflow = torch.relu(-vn_out).pow(2).mean()

        xb, yb, tb, nxb, nyb = sample_outer_boundary(OUTER_BC_N)
        _, _, vtx_b, vty_b = total_velocity(model, xb, yb, tb)
        vn_b = vtx_b * nxb + vty_b * nyb
        loss_noflow = (vn_b ** 2).mean()

        x0, y0, t0 = sample_initial(INIT_N)
        loss_ic_p = pressure_ic_loss(model, x0, y0, t0)
        _, S_0 = model(x0, y0, t0)
        loss_ic_s = ((S_0 - float(S_IC_CO2)) ** 2).mean()

        loss = (
            w_pde * loss_pde
            + w_fv * loss_fv
            + W_DATA_P * loss_data_p
            + W_DATA_S * loss_data_s
            + W_DATA_S * w_pairgrad * loss_data_s_pairgrad
            + W_INJ_FLUX * loss_inj_flux
            + W_INJ_SAT * loss_inj_sat
            + W_OUT_P * loss_out_p
            + W_OUT_BACKFLOW * loss_out_backflow
            + W_NOFLOW * loss_noflow
            + W_IC_P * loss_ic_p
            + W_IC_S * loss_ic_s
        )

        loss.backward()
        opt.step()

        if it % 200 == 0:
            weighted_pde = float(w_pde) * loss_pde.detach()
            weighted_fv = float(w_fv) * loss_fv.detach()
            weighted_data_p = float(W_DATA_P) * loss_data_p.detach()
            weighted_data_s = float(W_DATA_S) * loss_data_s.detach()
            weighted_pairgrad = float(W_DATA_S) * float(w_pairgrad) * loss_data_s_pairgrad.detach()
            weighted_inj_flux = float(W_INJ_FLUX) * loss_inj_flux.detach()
            weighted_inj_sat = float(W_INJ_SAT) * loss_inj_sat.detach()
            weighted_out_p = float(W_OUT_P) * loss_out_p.detach()
            weighted_out_backflow = float(W_OUT_BACKFLOW) * loss_out_backflow.detach()
            weighted_noflow = float(W_NOFLOW) * loss_noflow.detach()
            weighted_ic_p = float(W_IC_P) * loss_ic_p.detach()
            weighted_ic_s = float(W_IC_S) * loss_ic_s.detach()
            weighted_bc = weighted_inj_flux + weighted_inj_sat + weighted_out_p + weighted_out_backflow + weighted_noflow
            weighted_ic = weighted_ic_p + weighted_ic_s
            weighted_data = weighted_data_p + weighted_data_s + weighted_pairgrad
            print(
                f"[{EXP_NAME}|{TRAINING_STRATEGY}] it={it:6d} [{rar_flag}] "
                f"wPDE={w_pde:4.2f} wFV={w_fv:.2e} wFG={w_pairgrad:.2e} beta={beta:5.2f} eps={eps:.2e} "
                f"loss={loss.item():.3e} | pde={loss_pde.item():.2e} "
                f"fv={loss_fv.item():.2e} fvC={loss_fv_c.item():.2e} fvW={loss_fv_w.item():.2e} | "
                f"dataP={loss_data_p.item():.2e} dataS={loss_data_s.item():.2e} "
                f"Sg={loss_data_s_global.item():.2e} Sf={loss_data_s_front.item():.2e} Sp={loss_data_s_plume.item():.2e} "
                f"Sfg={loss_data_s_pairgrad.item():.2e} nFG={pairgrad_n_pairs.item():.0f} dFG={pairgrad_mean_dist.item():.2e} | "
                f"injF={loss_inj_flux.item():.2e} injS={loss_inj_sat.item():.2e} outP={loss_out_p.item():.2e} "
                f"outBF={loss_out_backflow.item():.2e} noflow={loss_noflow.item():.2e} "
                f"icP={loss_ic_p.item():.2e} icS={loss_ic_s.item():.2e} | "
                f"{get_gpu_memory_report()}"
            )
            print(
                f"[WeightedTerms] it={it:6d} total={loss.item():.3e} | "
                f"pde={weighted_pde.item():.2e} fv={weighted_fv.item():.2e} | "
                f"data={weighted_data.item():.2e} (P={weighted_data_p.item():.2e}, S={weighted_data_s.item():.2e}, "
                f"Sfg={weighted_pairgrad.item():.2e}) | "
                f"bc={weighted_bc.item():.2e} (injF={weighted_inj_flux.item():.2e}, injS={weighted_inj_sat.item():.2e}, "
                f"outP={weighted_out_p.item():.2e}, outBF={weighted_out_backflow.item():.2e}, "
                f"noflow={weighted_noflow.item():.2e}) | "
                f"ic={weighted_ic.item():.2e} (icP={weighted_ic_p.item():.2e}, icS={weighted_ic_s.item():.2e})"
            )

        if ENABLE_PERIODIC_CHECKPOINT and CHECKPOINT_EVERY > 0 and (it % CHECKPOINT_EVERY == 0):
            save_checkpoint(model, f"it{it}")

        if DEVICE.type == "cuda" and it % 5000 == 0:
            torch.cuda.empty_cache()

    use_lbfgs_now = USE_LBFGS and RUNTIME["USE_LBFGS"]
    if use_lbfgs_now:
        print("\n=== Starting L-BFGS fine-tuning; FV is intentionally disabled in L-BFGS ===")
        beta = BETA_MAX if USE_BETA_CURRICULUM else BETA_CONST
        eps = DIFF_EPS1 if USE_DIFFUSION_DECAY else EPS_CONST
        model.set_beta(beta)

        xf, yf, tf = sample_interior(N_BASE)
        xd, yd, td, ptd_true, Sd_true = sample_data_batch(DATA_BATCH)
        xi, yi, ti, nxi, nyi = sample_injection_arc(INJ_ARC_N)
        xo, yo, to, nxo, nyo = sample_outlet_arc(OUTLET_ARC_N)
        xb, yb, tb, nxb, nyb = sample_outer_boundary(OUTER_BC_N)
        x0, y0, t0 = sample_initial(INIT_N)
        pairgrad_batch_lbfgs = sample_pairwise_front_gradient_batch()

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=RUNTIME["LBFGS_MAX_ITER"],
            history_size=RUNTIME["LBFGS_HISTORY_SIZE"],
            line_search_fn="strong_wolfe",
        )

        closure_calls = {"n": 0}

        def closure():
            closure_calls["n"] += 1
            lbfgs.zero_grad(set_to_none=True)

            r_w, r_c, *_ = pde_residual(model, xf, yf, tf, eps=eps)
            loss_pde = (r_w ** 2).mean() + (r_c ** 2).mean()

            ptd_pred, Sd_pred = model(xd, yd, td)
            loss_data_p = ((ptd_pred - ptd_true) ** 2).mean()
            loss_data_s, _, _, _ = saturation_data_loss(Sd_pred, Sd_true)
            loss_data_s_pairgrad, _, _ = pairwise_front_gradient_loss_from_batch(model, pairgrad_batch_lbfgs)

            _, S_i, vtx_i, vty_i = total_velocity(model, xi, yi, ti)
            vn_in = vtx_i * nxi + vty_i * nyi
            loss_inj_flux = ((vn_in - 1.0) ** 2).mean()
            loss_inj_sat = ((S_i - float(S_INJ_CO2)) ** 2).mean()

            p_t_o, _, vtx_o, vty_o = total_velocity(model, xo, yo, to)
            loss_out_p = ((p_t_o - p_out_tilde) ** 2).mean()
            vn_out = vtx_o * nxo + vty_o * nyo
            loss_out_backflow = torch.relu(-vn_out).pow(2).mean()

            _, _, vtx_b, vty_b = total_velocity(model, xb, yb, tb)
            vn_b = vtx_b * nxb + vty_b * nyb
            loss_noflow = (vn_b ** 2).mean()

            loss_ic_p = pressure_ic_loss(model, x0, y0, t0)
            _, S_0 = model(x0, y0, t0)
            loss_ic_s = ((S_0 - float(S_IC_CO2)) ** 2).mean()

            loss = (
                loss_pde
                + W_DATA_P * loss_data_p
                + W_DATA_S * loss_data_s
                + W_DATA_S * W_DATA_S_PAIRGRAD_REL * loss_data_s_pairgrad
                + W_INJ_FLUX * loss_inj_flux
                + W_INJ_SAT * loss_inj_sat
                + W_OUT_P * loss_out_p
                + W_OUT_BACKFLOW * loss_out_backflow
                + W_NOFLOW * loss_noflow
                + W_IC_P * loss_ic_p
                + W_IC_S * loss_ic_s
            )
            loss.backward()
            if closure_calls["n"] % 50 == 0:
                print(f"[LBFGS-{EXP_NAME}] call={closure_calls['n']:5d} loss={loss.item():.3e}")
            return loss

        lbfgs.step(closure)

    save_checkpoint(model, "final")
    return model


if __name__ == "__main__":
    configure_experiment(EXP_NAME)
    configure_training_strategy(TRAINING_STRATEGY)
    configure_leave_one_out(LEAVE_OUT)
    set_seed(SEED)

    print(
        "Running EXP_NAME = {} | TRAINING_STRATEGY = {} | LEAVE_OUT = {} "
        "(M0: SingleNet baseline; M1: TwoNet; "
        "M2: TwoNet+MSFF+p/S input decoupling; M3: M2+coarse/detail saturation; "
        "M4: M3+unified front/plume loss; "
        "M5: M4+pair-wise front-gradient loss; "
        "M6: M5+residual-biased resampling; "
        "M7: M6+delayed local FV phase-mass conservation, CO2-dominant+weak-water)".format(EXP_NAME, TRAINING_STRATEGY, LEAVE_OUT)
    )
    print(f"GPU_ID request = {GPU_ID}")
    print(f"P_ref={P_ref:.3e} Pa, p_out_tilde={p_out_tilde:.3e}, A_time={A_time:.3f}")
    train()


