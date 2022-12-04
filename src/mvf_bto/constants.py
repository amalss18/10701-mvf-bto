import numpy as np

# DATA NORMALIZATION
VOLTAGE_MIN = 1.9
VOLTAGE_MAX = 3.5
TEMPERATURE_MIN = 24
TEMPERATURE_MAX = 38
MAX_CYCLE = 2300

# PREPROCESSING
# Constant Current 4C Discharge current is approx 4A
# Used to split discharge step from full cycle in preprocessing
MAX_DISCHARGE_CURRENT = -3.98
MIN_DISCHARGE_CURRENT = -4.05

# default feature and target column labels
DEFAULT_FEATURES = ["T_norm", "Q_eval", "V_norm", "Cycle"]
DEFAULT_TARGETS = ["V_norm", "T_norm"]

# cells with corrupted temperature data
BLACKLISTED_CELL = ["b1c3", "b1c8", "b1c28"]

# charging current
MIN_CHARGE_CURRENT = 0.0

MAX_DISCHARGE_CAPACITY = 1.6
# INTERPOLATION
# Discharge curve is interpolated over these capacities
# More points in non-linear regions
REFERENCE_DISCHARGE_CAPACITIES = [
    0.0001,
    0.0125,
    0.025,
    0.05,
    0.075,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.98,
    0.985,
    0.99,
    0.995,
    0.998,
    1.0,
]

# REFERENCE CHARGE CAPACITIES: DIFFERENT FROM DISCHARGE
# THREE STEPS OF CHARGING REQUIRES NON-UNIFORM SPLIT OF INTERPOLATION POINTS
Qc_eval = np.linspace(0, 0.1, 21)
Qc_eval = np.append(Qc_eval, np.linspace(0.1, 0.8, 8))
Qc_eval = np.append(Qc_eval, np.linspace(0.8, 0.85, 11))
Qc_eval = np.append(Qc_eval, np.linspace(0.85, 0.98, 14))
Qc_eval = np.append(Qc_eval, np.linspace(0.98, 1.00, 2))
REFERENCE_CHARGE_CAPACITIES = Qc_eval
