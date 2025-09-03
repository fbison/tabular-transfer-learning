from deep_tabular.utils.ic_tools import impute_and_save, split_downstream_dataset

def input_missing_columns(method):
    path_src = "data/ic_downstream1/ic_train_X.csv"
    path_dir = "data/ic_upstream2"
    impute_and_save(path_dir, path_src, "exp_100_1", method=method, seed=42)
    
    path_dir = "data/ic_upstream3"
    impute_and_save(path_dir, path_src, "exp_100_1", method=method, seed=42)

    path_dir = "data/ic_upstream4"
    impute_and_save(path_dir, path_src, "exp_100_1", method=method, seed=42)

    path_src = "data/ic_upstream2/ic_train_X.csv"
    path_dir = "data/ic_downstream1"
    impute_and_save(path_dir, path_src, "exp_100_2", method=method, seed=42)

    path_src = "data/ic_upstream3/ic_train_X.csv"
    impute_and_save(path_dir, path_src, "exp_100_3", method=method, seed=42)
    
    path_src = "data/ic_upstream4/ic_train_X.csv"
    impute_and_save(path_dir, path_src, "exp_100_4", method=method, seed=42)

def order_columns_in_csv(path_csv):
    import pandas as pd
    df = pd.read_csv(path_csv)
    cols = df.columns.tolist()
    cols_sorted = sorted(cols)
    df = df[cols_sorted]
    df.to_csv(path_csv, index=False)

def order_x_csv_in_directory(path_dir):
    import os
    for filename in os.listdir(path_dir):
        if "_y" in filename:
            continue
        if filename.endswith(".csv"):
            path_csv = os.path.join(path_dir, filename)
            order_columns_in_csv(path_csv)

def remove_target_from_csv(path_csv):
    import pandas as pd
    df = pd.read_csv(path_csv)
    if 'pIC50' in df.columns:
        df = df.drop(columns=['pIC50'])
        df.to_csv(path_csv, index=False)
def remove_target_from_directory(path_dir):
    import os
    for filename in os.listdir(path_dir):
        if "_y" in filename:
            continue
        if filename.endswith(".csv"):
            path_csv = os.path.join(path_dir, filename)
            remove_target_from_csv(path_csv)

def input_missing_columns_downstream_samples(method):
    sample_train_sizes = [5, 10, 20, 50, 75]
    for sample_size in sample_train_sizes:
        path_dir = f"data/ic_downstream1_Sample{sample_size}"

        path_src = "data/ic_upstream2/ic_train_X.csv"
        impute_and_save(path_dir, path_src, "exp_100_2", method=method, seed=42)

        path_src = "data/ic_upstream3/ic_train_X.csv"
        impute_and_save(path_dir, path_src, "exp_100_3", method=method, seed=42)
        
        path_src = "data/ic_upstream4/ic_train_X.csv"
        impute_and_save(path_dir, path_src, "exp_100_4", method=method, seed=42)

def main():
    input_missing_columns_downstream_samples(method="mean")
    input_missing_columns_downstream_samples(method="gaussian")

if __name__ == "__main__":
    main()