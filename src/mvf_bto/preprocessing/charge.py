import numpy as np
import pandas as pd
import random
import tqdm
from scipy.interpolate import interp1d
import sys
import re
import math
import plotly.graph_objects as go
from mvf_bto.constants import (
    VOLTAGE_MIN,
    VOLTAGE_MAX,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    MIN_CHARGE_CURRENT,
    MAX_DISCHARGE_CURRENT,
    MIN_DISCHARGE_CURRENT,
    REFERENCE_CHARGE_CAPACITIES,
    MAX_CYCLE
)

DEFAULT_FEATURES = ["T_norm", "Q_eval", "V_norm", "Cycle", "C_rate1", "SOC1", "C_rate2"]
DEFAULT_TARGETS = ["V_norm", "T_norm"]
BLACKLISTED_CELL = ["b1c3", "b1c8"]


def create_charge_inputs(
        data,
        train_split,
        test_split,
        input_columns=DEFAULT_FEATURES,
        output_columns=DEFAULT_TARGETS,
        history_window=4,
        q_eval=None,
        forecast_horizon=1,
):
    # TODO: add multi-timestep forecast horizon
    """
    Creates inputs to LSTM model for voltage and/ or temperature forecasting.
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
    q_eval: List[float]
        List of normalized capcities
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
    cell_ids = list(data.keys())
    # random.shuffle(cell_ids)

    n_train = int(train_split * len(cell_ids))
    n_test = int(test_split * len(cell_ids))
    assert train_split + test_split <= 1

    train_cells, test_cells, validation_cells = (
        cell_ids[:n_train],
        cell_ids[n_train: n_train + n_test],
        cell_ids[n_train + n_test:],
    )

    X_train_list, X_test_list, X_val_list = [], [], []
    y_train_list, y_test_list, y_val_list = [], [], []

    if(q_eval.any()==None):
        soc = float(re.findall(r"\d*\.*\d+",data[cell_id]["charge_policy"])[1])/100.0
        q_eval = np.linspace(0,0.1,11)
        q_eval = np.append(q_eval, np.linspace(0.2,soc,int(math.floor((soc-0.2)/0.1+1))))
        q_eval = np.append(q_eval, np.linspace(soc+0.005,soc+0.05,10))
        q_eval = np.append(q_eval, np.linspace(soc+0.06,1.0-0.02,math.floor(1.0-0.02-soc-0.06)/0.01+1))
        q_eval = np.append(q_eval, np.linspace(0.99,1.00,2))
        # q_eval=np.linspace(0,1,201)

    original_train_dfs = []
    fig_list=[]
    for cell_id in train_cells:
        X_cell_list, y_cell_list, original_df_list, fig = _get_single_cell_inputs(
            cell_id=cell_id,
            single_cell_data=data[cell_id]["cycles"],
            policy=data[cell_id]["charge_policy"],
            input_columns=input_columns,
            output_columns=output_columns,
            history_window=history_window,
            q_eval=q_eval,
            forecast_horizon=forecast_horizon
        )
        X_train_list.extend(X_cell_list)
        y_train_list.extend(y_cell_list)
        original_train_dfs.extend(original_df_list)
        fig_list.append(fig)

    original_test_dfs = []
    for cell_id in test_cells:
        X_cell_list, y_cell_list, original_df_list, fig = _get_single_cell_inputs(
            cell_id=cell_id,
            single_cell_data=data[cell_id]["cycles"],
            policy=data[cell_id]["charge_policy"],
            input_columns=input_columns,
            output_columns=output_columns,
            history_window=history_window,
            q_eval=q_eval,
            forecast_horizon=forecast_horizon,
        )
        X_test_list.extend(X_cell_list)
        y_test_list.extend(y_cell_list)
        original_test_dfs.extend(original_df_list)
        fig_list.append(fig)

    original_val_dfs = []
    if len(validation_cells):
        for cell_id in validation_cells:
            X_cell_list, y_cell_list, original_df_list, fig = _get_single_cell_inputs(
                cell_id=cell_id,
                single_cell_data=data[cell_id]["cycles"],
                policy=data[cell_id]["charge_policy"],
                input_columns=input_columns,
                output_columns=output_columns,
                history_window=history_window,
                q_eval=q_eval,
                forecast_horizon=forecast_horizon,
            )
            X_val_list.extend(X_cell_list)
            y_val_list.extend(y_cell_list)
            original_val_dfs.extend(original_df_list)
            fig_list.append(fig)

    batch_size = X_train_list[0].shape[0]
    X_train = np.array(X_train_list)
    X_test = np.array(X_test_list)
    X_val = np.array(X_val_list)

    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    y_val = np.array(y_val_list)

    X_train = X_train.reshape(
        X_train.shape[0] * batch_size, X_train[0].shape[1], X_train.shape[-1]
    )
    X_test = X_test.reshape(
        X_test.shape[0] * batch_size, X_test[0].shape[1], X_test.shape[-1]
    )
    X_val = X_val.reshape(
        X_val.shape[0] * batch_size, X_val[0].shape[1], X_val.shape[-1]
    )

    y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1], y_val.shape[2], y_train.shape[-1])
    y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1], y_val.shape[2], y_test.shape[-1])
    y_val = y_val.reshape(y_val.shape[0] * y_val.shape[1], y_val.shape[2], y_val.shape[-1])

    return {
               "X_train": X_train,
               "X_test": X_test,
               "X_val": X_val,
               "y_train": y_train,
               "y_test": y_test,
               "y_val": y_val,
               "original_test": pd.concat(original_test_dfs),
               "original_val": pd.concat(original_val_dfs),
               "original_train": pd.concat(original_train_dfs),
               "batch_size": batch_size,
               "fig_list": fig_list,
               "q_eval": q_eval
           }


def _get_interpolated_normalized_charge_data(cell_id, single_cell_data, policy, q_eval):
    """
    Interpolates voltage, temperature and time over reference capacities
    (defined in `q_eval`).
    ('q_eval' needs modifying between charge and discharge)
    Stores time series from each cycle in a dataframe.
    """
    df_list, original_df_list = [], []

    # iterate over each cycle in data
    fig=go.Figure()
    for cycle_key, time_series in tqdm.tqdm(single_cell_data.items()):
        cycle_num = int(cycle_key)

        if cycle_num < 1:
            continue
        # if cycle_num >= MAX_CYCLE:
        #     continue
        df = pd.DataFrame(
            {
                "t": time_series["t"],
                "V": time_series["V"],
                "temp": time_series["T"],
                "I": time_series["I"],
                "Qc": time_series["Qc"],
            }
        )
        df['Cycle'] = cycle_num
        df['Cell'] = cell_id
        df['C_rate1'] = float(re.findall(r"\d*\.*\d+",policy)[0])
        df['SOC1'] = float(re.findall(r"\d*\.*\d+",policy)[1])/100.0
        df['C_rate2'] = float(re.findall(r"\d*\.*\d+",policy)[2])

        # get discharge part of curve only (current is negative during discharge)
        # everything before discharge is charge
        # df = df[df.I > MIN_CHARGE_CURRENT]
        discharge_df = df[df.I < MAX_DISCHARGE_CURRENT].copy()
        discharge_df = discharge_df[discharge_df.I > MIN_DISCHARGE_CURRENT]
        df= df[df.t < discharge_df.t.values[0]]
        if (np.diff(df.t.values) >30).any():
            continue

        # original_df_list.append(df)
        temp_df=df.copy()

        # drop duplicates to be able to interpolate over capacity
        df = df.drop_duplicates(subset="Qc")

        df["V_norm"] = (df.V - VOLTAGE_MIN) / (VOLTAGE_MAX - VOLTAGE_MIN)
        df["T_norm"] = (df.temp - TEMPERATURE_MIN) / (TEMPERATURE_MAX - TEMPERATURE_MIN)
        df["Qc"] = (df.Qc - df.Qc.min()) / (df.Qc.max() - df.Qc.min())

        interp_df = pd.DataFrame()

        # use capacity as reference to interpolate over
        interp_df["Q_eval"] = q_eval

        fV = interp1d(x=df.Qc.values, y=df.V_norm.values)
        interp_df["V_norm"] = fV(q_eval)

        # TODO: add comments about logic why these are anomalies
        # if (np.diff(interp_df.V_norm[np.where(q_eval == 0.1)[0][0]:np.where(q_eval == df['SOC1'].values[0])[0][0]]) < -1e-5).any():
        #     continue
        # if len(np.where(abs(interp_df.V_norm - 3.6) < 1e-3)) > 3:
        #     continue

        fT = interp1d(x=df.Qc, y=df["T_norm"])
        interp_df["T_norm"] = fT(q_eval)

        # judge if datapoints are usable with temperature, which should not fluctuate during charging
        # range (0.2,0.8)
        # if (np.diff(interp_df.T_norm[np.where(q_eval == 0.1)[0][0]:np.where(q_eval == df["SOC1"].values[0])[0][0]]) < 0).any():
        #     fig.add_trace(go.Scatter(x=interp_df.Q_eval, y=interp_df.T_norm))
        #     continue


        interp_df["Cycle"] = [cycle_num / MAX_CYCLE for i in range(len(interp_df))]
        interp_df["C_rate1"] = [float(re.findall(r"\d*\.*\d+",policy)[0]) for i in range(len(interp_df))]
        interp_df["SOC1"] = [float(re.findall(r"\d*\.*\d+",policy)[1])/100.0 for i in range(len(interp_df))]
        interp_df["C_rate2"] = [float(re.findall(r"\d*\.*\d+",policy)[2]) for i in range(len(interp_df))]

        df_list.append(interp_df)
        original_df_list.append(temp_df)        
    return df_list, original_df_list,fig


def _split_sequences(sequences, n_steps, n_outputs, nf_steps):
    """
    Helper function to split a multivariate sequence into samples.
    """
    X, y = list(), list()
    for i in range(len(sequences) - n_steps - nf_steps):
        # find the end of this pattern
        end_ix = i + n_steps
        # gather input and output parts of the pattern
        seq_x, seq_y = (
            sequences[i: end_ix - 1, :-n_outputs],
            [sequences[end_ix - 1 + j, -n_outputs:] for j in np.arange(nf_steps)],
        )
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def _get_single_cell_inputs(
        cell_id, single_cell_data, policy, input_columns, output_columns, history_window, q_eval, forecast_horizon
):
    """
    Helper function to preprocess time series inputs for a single cell (battery).
    """
    X_list, y_list = [], []
    df_list, original_df_list,fig = _get_interpolated_normalized_charge_data(cell_id, single_cell_data, policy, q_eval=q_eval)
    for df in df_list:
        sequence_list = []

        # define input sequences
        for column in input_columns:
            in_seq = df[column].values
            in_seq = in_seq.reshape((len(in_seq), 1))
            sequence_list.append(in_seq)

        # define output sequences
        for column in output_columns:
            out_seq = df[column].values
            out_seq = out_seq.reshape((len(out_seq), 1))
            sequence_list.append(out_seq)

        # convert to [rows, columns] structure
        # horizontally stack columns
        dataset = np.hstack(tuple(sequence_list))

        # convert into input/output
        X_cycle, y_cycle = _split_sequences(
            dataset, history_window, n_outputs=len(output_columns), nf_steps=forecast_horizon
        )
        X_list.append(X_cycle)
        y_list.append(y_cycle)

    return X_list, y_list, original_df_list, fig
