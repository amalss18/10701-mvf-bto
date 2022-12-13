import h5py
import tqdm
import numpy as np


def load_data(file_path, num_cells=None, batch_id = 1):
    """
    Loads battery cycling data from a .mat file.
    Parameters
    __________
    file_path: str
        Absolute path to data file.
        (download here: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)
    num_cells: Optional[int]
        Number of cells to load data for.
        Defaults to all cells in file.

    Returns
    _______
    bat_dict: dict
        Nested dictionary with battery ID as top level keys.
        Within each battery level dictionary:
        `cycle_life`: Total number of cycles completed by cell.
        `charge_policy`: Charging protocol.
        `summary`: Aggregate features calculated from time series.
        `cycles`: Time series battery cycling data
                  (with cycle number str as key).
    """
    f = h5py.File(file_path)
    batch = f["batch"]
    bat_dict = {}
    if num_cells is None:
        num_cells = batch["summary"].shape[0]
    for i in tqdm.tqdm(range(num_cells)):
        cl = f[batch["cycle_life"][i, 0]][()]
        policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch["summary"][i, 0]]["IR"][0, :].tolist())
        summary_QC = np.hstack(f[batch["summary"][i, 0]]["QCharge"][0, :].tolist())
        summary_QD = np.hstack(f[batch["summary"][i, 0]]["QDischarge"][0, :].tolist())
        summary_TA = np.hstack(f[batch["summary"][i, 0]]["Tavg"][0, :].tolist())
        summary_TM = np.hstack(f[batch["summary"][i, 0]]["Tmin"][0, :].tolist())
        summary_TX = np.hstack(f[batch["summary"][i, 0]]["Tmax"][0, :].tolist())
        summary_CT = np.hstack(f[batch["summary"][i, 0]]["chargetime"][0, :].tolist())
        summary_CY = np.hstack(f[batch["summary"][i, 0]]["cycle"][0, :].tolist())
        summary = {
            "IR": summary_IR,
            "QC": summary_QC,
            "QD": summary_QD,
            "Tavg": summary_TA,
            "Tmin": summary_TM,
            "Tmax": summary_TX,
            "chargetime": summary_CT,
            "cycle": summary_CY,
        }
        cycles = f[batch["cycles"][i, 0]]
        cycle_dict = {}
        for j in range(cycles["I"].shape[0]):
            I = np.hstack((f[cycles["I"][j, 0]][()]))
            Qc = np.hstack((f[cycles["Qc"][j, 0]][()]))
            Qd = np.hstack((f[cycles["Qd"][j, 0]][()]))
            Qdlin = np.hstack((f[cycles["Qdlin"][j, 0]][()]))
            T = np.hstack((f[cycles["T"][j, 0]][()]))
            Tdlin = np.hstack((f[cycles["Tdlin"][j, 0]][()]))
            V = np.hstack((f[cycles["V"][j, 0]][()]))
            dQdV = np.hstack((f[cycles["discharge_dQdV"][j, 0]][()]))
            t = np.hstack((f[cycles["t"][j, 0]][()]))
            cd = {
                "I": I,
                "Qc": Qc,
                "Qd": Qd,
                "Qdlin": Qdlin,
                "T": T,
                "Tdlin": Tdlin,
                "V": V,
                "dQdV": dQdV,
                "t": t,
            }
            cycle_dict[str(j)] = cd

        cell_dict = {
            "cycle_life": cl,
            "charge_policy": policy,
            "summary": summary,
            "cycles": cycle_dict,
        }
        key = f"b{batch_id}c{i}"
        bat_dict[key] = cell_dict
    return bat_dict
