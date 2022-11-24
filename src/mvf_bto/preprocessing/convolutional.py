import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
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
    DEFAULT_TARGETS,
    DEFAULT_FEATURES,
    REFERENCE_DISCHARGE_CAPACITIES,
)
from mvf_bto.preprocessing.utils import split_train_validation_test_sets

DEFAULT_FEATURES = ["T_norm", "Q_eval", "V_norm", "Cycle"]


def _split_sequences(sequences, n_steps_in, n_steps_out, n_outputs):
    """
    Helper function to split a multivariate sequence into samples.
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break

        seq_x, seq_y = (
            sequences.copy()[i:end_ix, :-n_outputs],
            sequences.copy()[end_ix:out_end_ix, -n_outputs:],
        )
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


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

        original_df_list.append(df)
        # drop duplicates to be able to interpolate over capacity

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
                np.linspace(df.Qd.min(), lower_threshold - 1e-5, 10),
                np.linspace(lower_threshold, upper_threshold - 1e-5, 30),
                np.linspace(upper_threshold, df.Qd.max(), 20),
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

    return df_list, original_df_list


def _dataframe_to_input_arrays(
    cycle_df, inputs_list, outputs_list, history_window, forecast_horizon
):
    """
    Shapes a dataframe containing the data for one cycle of a cell
    to the appropriate input and target format for the model.

    Parameters
    __________
    cycle_df: pd.DataFrame
        Data for one cycle.
    inputs_list: List[str]
        Columns labels corresponding to features.
    outputs_list: List[str]
        Columns labels corresponding to targets.
    history_widow: int
        Number of previous timesteps to use for prediction.
    forecast_horizon: int
        Number of timesteps in the future to predict.
    Returns
    _________
    X, y: Tuple[np.ndarray, np.ndarray]
        Feature and target arrays.
    """
    n_outputs = len(outputs_list)

    dataset_list = []
    # define input sequences
    for feature in inputs_list:
        in_seq = cycle_df[feature].values
        in_seq = in_seq.reshape((len(in_seq), 1))
        dataset_list.append(in_seq)

    for target in outputs_list:
        in_seq = cycle_df[target].values
        in_seq = in_seq.reshape((len(in_seq), 1))
        dataset_list.append(in_seq)

    # convert to [rows, columns] structure
    # by horizontally stacking columns
    dataset = np.hstack(tuple(dataset_list))
    # choose a number of time steps
    # convert into input/output
    X, y = _split_sequences(
        sequences=dataset,
        n_steps_in=history_window,
        n_steps_out=forecast_horizon,
        n_outputs=n_outputs,
    )
    # flatten output
    y = y.reshape((y.shape[0], n_outputs * forecast_horizon))
    return X, y


def create_discharge_inputs(
    data,
    train_split,
    test_split,
    input_columns=DEFAULT_FEATURES,
    output_columns=DEFAULT_TARGETS,
    history_window=4,
    forecast_horizon=1,
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
    input_columns: List[str]
        List of feature column labels.
    output_columns: List[str]
        List of target column labels.
    forecast_horizon: int
        Number of timesteps in the future to predict.
    q_eval: List[float]
        List of discharge
    Returns
    _________
    datasets: Dict[str, np.ndarray]
        Dictionary with test, train and validation datasets.
         (Keys: X_train, X_test, X_val,
                y_train, y_test, y_val,
                batch_size.)
    """
    train_cells, validation_cells, test_cells = split_train_validation_test_sets(
        data=data, train_split=train_split, test_split=test_split
    )

    train_df_list, train_odf_list = [], []
    for cell_id in train_cells:
        single_cell_data = data[cell_id]["cycles"]
        try:
            df_list, original_df_list = _get_interpolated_normalized_discharge_data(
                cell_id,
                single_cell_data,
            )
            train_df_list.extend(df_list)
            train_odf_list.extend(original_df_list)
        except ValueError:
            continue

    X_train, y_train = [], []
    for df in train_df_list:
        X_cycle, y_cycle = _dataframe_to_input_arrays(
            df.copy(),
            inputs_list=input_columns,
            outputs_list=output_columns,
            history_window=history_window,
            forecast_horizon=forecast_horizon,
        )
        X_train.append(X_cycle.copy())
        y_train.append(y_cycle.copy())

    test_df_list, test_odf_list = [], []
    for cell_id in test_cells:
        single_cell_data = data[cell_id]["cycles"]
        try:
            df_list, original_df_list = _get_interpolated_normalized_discharge_data(
                cell_id, single_cell_data
            )
            test_df_list.extend(df_list)
            test_odf_list.extend(original_df_list)
        except ValueError:
            continue

    X_test, y_test = [], []
    for df in test_df_list:

        X_cycle, y_cycle = _dataframe_to_input_arrays(
            df.copy(),
            inputs_list=input_columns,
            outputs_list=output_columns,
            history_window=history_window,
            forecast_horizon=forecast_horizon,
        )
        X_test.append(X_cycle.copy())
        y_test.append(y_cycle.copy())

    val_df_list, val_odf_list = [], []
    for cell_id in validation_cells:
        try:
            single_cell_data = data[cell_id]["cycles"]
            df_list, original_df_list = _get_interpolated_normalized_discharge_data(
                cell_id, single_cell_data
            )
            val_df_list.extend(df_list)
            val_odf_list.extend(original_df_list)

        except ValueError:
            continue

    X_val, y_val = [], []
    for df in val_df_list:
        X_cycle, y_cycle = _dataframe_to_input_arrays(
            df.copy(),
            inputs_list=input_columns,
            outputs_list=output_columns,
            history_window=history_window,
            forecast_horizon=forecast_horizon,
        )
        X_val.append(X_cycle.copy())
        y_val.append(y_cycle.copy())

    X_train = np.array(X_train)
    arrays_per_cycle = X_train.shape[1]
    X_train = X_train.reshape(
        X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3]
    )
    X_test = np.array(X_test)
    X_test = X_test.reshape(
        X_test.shape[0] * X_test.shape[1], X_test.shape[2], X_test.shape[3]
    )
    X_val = np.array(X_val)
    X_val = X_val.reshape(
        X_val.shape[0] * X_val.shape[1], X_val.shape[2], X_val.shape[3]
    )

    y_train = np.array(y_train)
    y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1], y_train.shape[2])
    y_test = np.array(y_test)
    y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2])
    y_val = np.array(y_val)
    y_val = y_val.reshape(y_val.shape[0] * y_val.shape[1], y_val.shape[2])

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
        "arrays_per_cycle": arrays_per_cycle,
        "n_features": X_train.shape[2],
    }
