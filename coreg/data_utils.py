import pandas as pd


def load_data(labeled_path, unlabeled_path):
    labeled = read_helper(labeled_path)
    unlabeled = read_helper(unlabeled_path, test=True)
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
