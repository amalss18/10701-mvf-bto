import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import tqdm
import re

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
    MAX_CHARGE_CAPACITY,
)
from mvf_bto.preprocessing.utils import split_train_validation_test_sets

DEFAULT_FEATURES = ["T_norm", "Q_eval", "V_norm", "Cycle", "C_rate1", 'C_rate2']
double_V_drop = ['b1c8']

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


def _get_interpolated_normalized_charge_data(cell_id, single_cell_data, policy, q_eval):
    """
    Interpolates voltage, temperature and time over reference capacities
    (defined in `q_eval`).
    Stores time series from each cycle in a dataframe.
    policy: str
        charge policy (e.g. 3.6C-40%-3.6C)
    """
    df_list, original_df_list = [], []
    if (not (q_eval is None)):
        flag=True
    else:
        flag=False
    # iterate over each cycle in data
    for cycle_key, time_series in tqdm.tqdm(single_cell_data.items()):
        cycle_num = int(cycle_key)
        # make sure q_eval is none if it is specified as none when called
        # flag=True -> q_eval not none
        if (flag==False and not (q_eval is None)):
            q_eval=None

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
                "Qc": time_series["Qc"],
            }
        )
        df["Cycle"] = cycle_num
        df["Cell"] = cell_id
        df['C_rate1'] = float(re.findall(r"\d*\.*\d+",policy)[0])
        # df['SOC1'] = float(re.findall(r"\d*\.*\d+",policy)[1])/100.0
        df['C_rate2'] = float(re.findall(r"\d*\.*\d+",policy)[2])

        original_df_list.append(df)
        # drop duplicates to be able to interpolate over capacity
        # get discharge part of curve only (current is negative during discharge)
        # everything before discharge is charge
        # df = df[df.I > MIN_CHARGE_CURRENT]
        discharge_df = df[df.I < MAX_DISCHARGE_CURRENT].copy()
        discharge_df = discharge_df[discharge_df.I > MIN_DISCHARGE_CURRENT]
        df= df[df.t < discharge_df.t.values[0]]
        # if (np.diff(df.t.values) >30).any():
        #     continue

        temp_df=df.copy()

        # drop duplicates to be able to interpolate over capacity
        df = df.drop_duplicates(subset="Qc")

        interp_df = pd.DataFrame()
        if(q_eval is None):
            # soc=[]
            max_interp_num=45
            # Q_where_V_drop=[]
            V_drop_index=np.where(np.diff(df["V"].values)<-1e-4)[0]
            # if(cycle_num==612 and cell_id=='b1c5'):
            #     print(len(V_drop_index))
            #     print(V_drop_index)
            # V_drop_index=np.delete(V_drop_index,-1)
            # for charge step with only one voltage drop
            if (not (cell_id in double_V_drop)):
                q_eval = np.concatenate(
                    (
                        np.linspace(df.Qc.min(),0.1,10),
                        np.linspace(0.2,df['Qc'].values[V_drop_index[0]-1],8),
                        np.linspace(df['Qc'].values[V_drop_index[0]-1], df['Qc'].values[V_drop_index[0]+2],3),
                        np.linspace(df['Qc'].values[V_drop_index[0]+4], max(df['Qc'].values)-0.01,13),
                    ),
                    axis=0,
                )
                q_eval = np.append(q_eval, np.linspace(df.Qc.max()-0.01,df.Qc.max(),max_interp_num-len(q_eval)))
                # q_eval = np.append(q_eval, np.array(df['Qc'].values[V_drop_index[np.where(np.diff(V_drop_index)<10)[0]]]))
                # q_eval = np.append(q_eval, np.linspace(df['Qc'].values[V_drop_index[np.where(np.diff(V_drop_index)<10)[0][-1]+1]],df['Qc'].values[V_drop_index[-1]],10))
                # q_eval = np.append(q_eval, np.linspace(df['Qc'].values[V_drop_index[-1]+1],max(df['Qc'].values)-0.01,10))
            # for charge step with 2 voltage drop
            elif (cell_id in double_V_drop):
                q_eval = np.concatenate(
                    (
                        np.linspace(df.Qc.min(),0.1,10),
                        np.linspace(0.2,df['Qc'].values[V_drop_index[0]-1],4),
                        np.linspace(df['Qc'].values[V_drop_index[0]-1], df['Qc'].values[V_drop_index[0]+2],3),
                        np.linspace(df['Qc'].values[V_drop_index[0]+4], df['Qc'].values[V_drop_index[np.where(np.diff(V_drop_index)>10)[0][1]-1]],9),
                        np.linspace(df['Qc'].values[V_drop_index[np.where(np.diff(V_drop_index)>10)[0][1]]],df['Qc'].values[V_drop_index[np.where(np.diff(V_drop_index)>10)[0][1]]+2],3),
                        np.linspace(df['Qc'].values[V_drop_index[np.where(np.diff(V_drop_index)>10)[0][1]]+4],max(df['Qc'].values)-0.01,13),
                    ),
                    axis=0
                )
                q_eval = np.append(q_eval, np.linspace(df.Qc.max()-0.01,df.Qc.max(),max_interp_num-len(q_eval)))
                # q_eval = np.append(q_eval, np.array(df['Qc'].values[V_drop_index[np.where(np.diff(V_drop_index)<10)]]))
                # q_eval = np.append(q_eval, np.linspace(df['Qc'].values[V_drop_index[-1]+1],max(df['Qc'].values)-0.01,10))
        else:
            raise RuntimeError('q_eval not none')
        if(len(q_eval)>max_interp_num):
            raise RuntimeError('check q_eval length (should be less than 50)')
        interp_df["Q_eval"] = q_eval
        fV = interp1d(x=df.Qc.values, y=df.V.values)  
        interp_df["V_norm"] = fV(q_eval) 
        fT = interp1d(x=df.Qc, y=df["temp"], fill_value='extrapolate')
        interp_df["T_norm"] = fT(q_eval)
        # judge if datapoints are usable with temperature, which should not fluctuate during charging
        # range (0.2,first_voltage_drop)
        if (np.diff(interp_df.T_norm[np.where(q_eval == 0.1)[0][0]:np.where(q_eval == df['Qc'].values[V_drop_index[0]-1])[0][0]]) < 0).any():
            continue
        interp_df["Cycle"] = [cycle_num / MAX_CYCLE for i in range(len(interp_df))]
        interp_df["V_norm"] = (interp_df.V_norm - VOLTAGE_MIN) / (
            VOLTAGE_MAX - VOLTAGE_MIN
        )
        interp_df["T_norm"] = (interp_df.T_norm - TEMPERATURE_MIN) / (
            TEMPERATURE_MAX - TEMPERATURE_MIN
        )
        interp_df["Q_eval"] = interp_df.Q_eval / MAX_CHARGE_CAPACITY
        interp_df["C_rate1"] = [float(re.findall(r"\d*\.*\d+",policy)[0]) for i in range(len(interp_df))]
        # interp_df["SOC1"] = [float(re.findall(r"\d*\.*\d+",policy)[1])/100.0 for i in range(len(interp_df))]
        interp_df["C_rate2"] = [float(re.findall(r"\d*\.*\d+",policy)[2]) for i in range(len(interp_df))]
        interp_df["Cell"] = [cell_id for i in range(len(interp_df))]

        df_list.append(interp_df)
        original_df_list.append(temp_df)

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


def create_charge_inputs(
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
    print(train_cells,validation_cells,test_cells)

    train_df_list, train_odf_list = [], []
    for cell_id in train_cells:
        single_cell_data = data[cell_id]["cycles"]
        policy = data[cell_id]['charge_policy']
        # print(cell_id, policy)
        try:
            df_list, original_df_list = _get_interpolated_normalized_charge_data(
                cell_id,
                single_cell_data,
                policy=policy,
                q_eval=None
            )
            train_df_list.extend(df_list)
            train_odf_list.extend(original_df_list)
        except ValueError:
            print(f'no train cell{cell_id}!')
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
        policy = data[cell_id]['charge_policy']
        # print(cell_id, policy)
        try:
            df_list, original_df_list = _get_interpolated_normalized_charge_data(
                cell_id, single_cell_data,
                policy=policy,
                q_eval=None
            )
            test_df_list.extend(df_list)
            test_odf_list.extend(original_df_list)
        except ValueError:
            print(f'no test cell{cell_id}!')
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
            policy = data[cell_id]['charge_policy']
            # print(cell_id, policy)
            df_list, original_df_list = _get_interpolated_normalized_charge_data(
                cell_id, single_cell_data,
                policy=policy,
                q_eval=None
            )
            val_df_list.extend(df_list)
            val_odf_list.extend(original_df_list)

        except ValueError:
            print(f'no validation cell{cell_id}!')
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
        "train_dfs": pd.concat(train_df_list),
        "test_dfs": pd.concat(test_df_list),
        "val_dfs": pd.concat(val_df_list),
        "arrays_per_cycle": arrays_per_cycle,
        "n_features": X_train.shape[2],
    }
