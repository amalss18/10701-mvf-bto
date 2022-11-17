import numpy as np
import pandas as pd
import random
import tqdm
from scipy.interpolate import interp1d
import sys
from mvf_bto.constants import (
    VOLTAGE_MIN,
    VOLTAGE_MAX,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    MAX_DISCHARGE_CURRENT,
    MIN_DISCHARGE_CURRENT,
    MIN_CHARGE_CURRENT,
    REFERENCE_DISCHARGE_CAPACITIES,
    REFERENCE_CHARGE_CAPACITIES,
    MAX_CYCLE
)

DEFAULT_FEATURES = ["T_norm", "Q_eval", "V_norm", "Cycle"]
DEFAULT_TARGETS = ["V_norm", "T_norm"]


def _split_sequences(sequences, n_steps_in, n_steps_out, n_outputs):
    X, y = list(), list()
    for i in range(len(sequences)):

        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, -n_outputs:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def _get_interpolated_normalized_discharge_data(cell_id, single_cell_data, q_eval):
    """
    Interpolates voltage, temperature and time over reference capacities
    (defined in `q_eval`).
    Stores time series from each cycle in a dataframe.
    """
    df_list, original_df_list = [], []

    # iterate over each cycle in data
    for cycle_key, time_series in tqdm.tqdm(single_cell_data.items()):
        cycle_num = int(cycle_key)

        if cycle_num < 1:
            continue
        if cycle_num >= MAX_CYCLE:
            continue
        df = pd.DataFrame(
            {
                "t": time_series["t"],
                "V": time_series["V"],
                "temp": time_series["T"],
                "I": time_series["I"],
                "Qd": time_series["Qd"]
            }
        )
        df['Cycle'] = cycle_num
        df['Cell'] = cell_id

        original_df_list.append(df)
        # drop duplicates to be able to interpolate over capacity
        df = df.drop_duplicates(subset="Qd")

        # get discharge part of curve only (current is negative during discharge)
        df = df[df.I < MAX_DISCHARGE_CURRENT]
        df = df[df.I > MIN_DISCHARGE_CURRENT]

        df["V_norm"] = (df.V - VOLTAGE_MIN) / (VOLTAGE_MAX - VOLTAGE_MIN)
        df["T_norm"] = (df.temp - TEMPERATURE_MIN) / (TEMPERATURE_MAX - TEMPERATURE_MIN)
        df["Qd"] = (df.Qd - df.Qd.min()) / (df.Qd.max() - df.Qd.min())
        interp_df = pd.DataFrame()

        # use capacity as reference to interpolate over
        interp_df["Q_eval"] = q_eval

        fV = interp1d(x=df.Qd.values, y=df.V_norm.values)
        interp_df["V_norm"] = fV(q_eval)
        # if data contains mislabeled points don't include in dataset
        # (since voltage should be monotonically decreasing)
        if (np.diff(interp_df.V_norm) > 0).any():
            continue

        fT = interp1d(x=df.Qd, y=df["T_norm"])
        interp_df["T_norm"] = fT(q_eval)
        interp_df["Cycle"] = [cycle_num / MAX_CYCLE for i in range(len(interp_df))]

        df_list.append(interp_df)
    return df_list, original_df_list
