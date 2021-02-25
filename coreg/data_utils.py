import os

import pandas as pd
import numpy as np


# def load_data(data_dir):
#     """
#     Loads data from data directory.
#     """
#     X = np.load(os.path.join(data_dir, 'X.npy'))
#     y = np.load(os.path.join(data_dir, 'y.npy'))
#     return X, y.reshape(-1, 1)

def load_data(data_dir):
    labeled = read_helper(os.path.join(data_dir, "Training_wells.csv"))
    unlabeled = read_helper(os.path.join(data_dir, "Empty_part.csv"), test=True)
    data = pd.concat([labeled, unlabeled])
    X = data[["x", "y"]].values
    y = data["target"].values
    return X, y.reshape(-1, 1)

def read_helper(path: str, *, test: bool = False) -> pd.DataFrame:
    columns = "id x y target".split()
    if test:
        columns = columns[:-1]
    df = pd.read_csv(path)
    df.columns = columns
    return df
