import os
import pandas as pd


def load_dataset(dataset_dir):
    """
    Loads ic_train_X.csv and ic_train_Y.csv, merges them,
    prioritizing columns from X when duplicates exist.
    """

    path_x = os.path.join(dataset_dir, "ic_train_X.csv")
    path_y = os.path.join(dataset_dir, "ic_train_Y.csv")

    df_x = pd.read_csv(path_x)
    df_y = pd.read_csv(path_y)

    # Validation: same number of rows (since alignment is by order)
    if len(df_x) != len(df_y):
        raise ValueError(f"Row mismatch in {dataset_dir}: X={len(df_x)} Y={len(df_y)}")

    # Remove duplicated columns from Y
    overlap_cols = set(df_x.columns).intersection(set(df_y.columns))
    df_y = df_y.drop(columns=list(overlap_cols))

    df = pd.concat([df_x, df_y], axis=1)

    return df