import random
from mvf_bto.constants import BLACKLISTED_CELL

BLACKLISTED_CELL=['b1c3','b1c8','b1c28']
def split_train_validation_test_sets(data, train_split, test_split):
    loaded_cell_ids = list(data.keys())
    cell_ids = []

    for cell_id in loaded_cell_ids:
        if cell_id not in BLACKLISTED_CELL:
            cell_ids.append(cell_id)

        else:
            print(f" Data for cell {cell_id} is corrupted. Skipping cell.")

    random.shuffle(cell_ids)

    n_train = int(train_split * len(cell_ids))
    n_test = int(test_split * len(cell_ids))
    assert train_split + test_split <= 1

    train_cells, test_cells, validation_cells = (
        cell_ids[:n_train],
        cell_ids[n_train : n_train + n_test],
        cell_ids[n_train + n_test :],
    )

    if len(cell_ids) == 2:
        train_cells = [
            cell_ids[0],
        ]
        test_cells = [
            cell_ids[1],
        ]

    if len(cell_ids) == 3:
        train_cells = [
            cell_ids[0],
        ]
        test_cells = [
            cell_ids[1],
        ]
        validation_cells = [
            cell_ids[2],
        ]

    return train_cells, validation_cells, test_cells
