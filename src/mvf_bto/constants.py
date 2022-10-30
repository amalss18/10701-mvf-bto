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

# INTERPOLATION
# Discharge curve is interpolated over these capacities
# More points in non-linear regions
REFERENCE_CAPACITIES = [
    0.00,
    0.025,
    0.075,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
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

# REFERENCE_CAPACITIES=np.linspace(0,1,31)
# REFERENCE_CAPACITIES=np.linspace(0,1,41)
