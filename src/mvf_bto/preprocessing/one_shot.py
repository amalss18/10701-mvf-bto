import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import tqdm
from mvf_bto.constants import (
    VOLTAGE_MIN,
    VOLTAGE_MAX,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    MAX_DISCHARGE_CURRENT,
    MIN_DISCHARGE_CURRENT,
    MAX_DISCHARGE_CAPACITY,
    MAX_CYCLE,
)
from mvf_bto.preprocessing.utils import split_train_validation_test_sets
from scipy.interpolate import interp1d

REFERENCE_DISCHARGE_CAPACITIES = np.concatenate(
    (np.linspace(0, 0.06, 10), np.linspace(0.07, 0.85, 30), np.linspace(0.86, 1, 20)), axis=0)


def _get_interpolated_normalized_discharge_data(cell_id, single_cell_data):
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
                "Qd": time_series["Qd"],
            }
        )
        df["Cycle"] = cycle_num
        df["Cell"] = cell_id

        o_df = df.copy()

        # get discharge part of curve only (current is negative during discharge)
        df = df[df.I < MAX_DISCHARGE_CURRENT]
        df = df[df.I > MIN_DISCHARGE_CURRENT]
        df = df.drop_duplicates(subset="Qd")

        interp_df = pd.DataFrame()

        # use capacity as reference to interpolate over
        lower_threshold = df.Qd.min() + (df.Qd.max() - df.Qd.min()) * 0.06
        upper_threshold = df.Qd.min() + (df.Qd.max() - df.Qd.min()) * 0.9
        q_eval = np.concatenate(
            (
                np.linspace(df.Qd.min(), lower_threshold - 1e-5, 6),
                np.linspace(lower_threshold, upper_threshold - 1e-5, 30),
                np.linspace(upper_threshold, df.Qd.max(), 22),
            ),
            axis=0,
        )

        interp_df["Q_eval"] = q_eval

        fV = interp1d(x=df.Qd.values, y=df.V.values)

        interp_df["V_norm"] = fV(q_eval)
        # if data contains mislabeled points don't include in dataset
        # (since voltage should be monotonically decreasing)
        if (np.diff(interp_df.V_norm) > 0).any():
            continue

        fT = interp1d(x=df.Qd, y=df["temp"])
        interp_df["T_norm"] = fT(q_eval)
        interp_df["Cycle"] = [cycle_num / MAX_CYCLE for i in range(len(interp_df))]

        interp_df["V_norm"] = (interp_df.V_norm - VOLTAGE_MIN) / (
                VOLTAGE_MAX - VOLTAGE_MIN
        )
        interp_df["T_norm"] = (interp_df.T_norm - TEMPERATURE_MIN) / (
                TEMPERATURE_MAX - TEMPERATURE_MIN
        )
        interp_df["Q_eval"] = interp_df.Q_eval / MAX_DISCHARGE_CAPACITY
        df_list.append(interp_df)
        original_df_list.append(o_df)

    return df_list, original_df_list


def create_discharge_inputs(
        data,
        train_split,
        test_split,
        history_window=16,
):
    """
    Creates inputs to a 1D Convolutional forecasting model
    for voltage and/ or temperature forecasting.
    Parameters
    __________
    data: Dict[str, Dict]
        Nested dictionary with battery ID as top level keys.
        Format loaded using `load_data` function in `data_loading` module.)
    train_split: float
        Fraction of data to use for training.
    test_split: float
        Fraction of data to use for testing.
    history_widow: int
        Number of previous timesteps to use for prediction.
    forecast_horizon: int
        Number of timesteps in the future to predict.

    Returns
    _________
    datasets: Dict[str, np.ndarray]
        Dictionary with test, train and validation datasets.
         (Keys: X_train, X_test, X_val,
                y_train, y_test, y_val,
                batch_size.)
    """
    train_cells, val_cells, test_cells = split_train_validation_test_sets(
        data=data, train_split=train_split, test_split=test_split
    )

    train_df_list, train_odf_list, train_q_eval = [], [], []
    for cell_id in train_cells:
        single_cell_data = data[cell_id]['cycles']
        df_list, original_df_list = _get_interpolated_normalized_discharge_data(cell_id,
                                                                                single_cell_data, )
        train_df_list.extend(df_list)
        train_odf_list.extend(original_df_list)
        train_q_eval.extend([df.Q_eval.values for df in df_list])

    X_list, y_list = [], []
    for df in train_df_list:
        X = df[:history_window].values
        y = np.concatenate((df['V_norm'].values, df['T_norm'].values), axis=0)
        X_list.append(X)
        y_list.append(y)

    X_train = np.array(X_list)
    y_train = np.array(y_list)

    ## test set
    test_df_list, test_odf_list, test_q_eval = [], [], []
    for cell_id in test_cells:
        single_cell_data = data[cell_id]['cycles']
        df_list, original_df_list = _get_interpolated_normalized_discharge_data(cell_id, single_cell_data)
        test_df_list.extend(df_list)
        test_odf_list.extend(original_df_list)
        test_q_eval.extend([df.Q_eval.values for df in df_list])

    X_list, y_list = [], []
    for df in test_df_list:
        X = df[:history_window].values
        y = np.concatenate((df['V_norm'].values, df['T_norm'].values), axis=0)
        X_list.append(X)
        y_list.append(y)

    X_test = np.array(X_list)
    y_test = np.array(y_list)

    ## validation set
    val_df_list, val_odf_list, val_q_eval = [], [], []
    for cell_id in val_cells:
        single_cell_data = data[cell_id]['cycles']
        df_list, original_df_list = _get_interpolated_normalized_discharge_data(cell_id, single_cell_data,
                                                                                )
        val_df_list.extend(df_list)
        val_odf_list.extend(original_df_list)

    X_list, y_list = [], []
    for df in val_df_list:
        X = df[:history_window].values
        y = np.concatenate((df['V_norm'].values, df['T_norm'].values), axis=0)
        X_list.append(X)
        y_list.append(y)

    X_val = np.array(X_list)
    y_val = np.array(y_list)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "original_test": pd.concat(test_odf_list),
        "original_val": pd.concat(val_odf_list),
        "original_train": pd.concat(train_odf_list),
        "q_eval_test": test_q_eval,
        "q_eval_val": val_q_eval,
        "q_eval_train": train_q_eval,
    }
