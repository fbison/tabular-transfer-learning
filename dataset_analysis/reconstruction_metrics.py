import numpy as np


def reconstruction_error(df_gt, df_imp):
    """
    Assumes same order and same columns.
    """

    if df_gt.shape != df_imp.shape:
        raise ValueError("Shape mismatch between GT and imputed")

    diff = df_gt.values - df_imp.values

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }