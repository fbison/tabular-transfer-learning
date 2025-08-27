from deep_tabular.utils.ic_tools import imputar_colunas_faltantes

def input_missing_columns_with_gausian():
    path_src = "data/ic_downstream1/exp_100_1.csv"
    path_dir = "data/ic_upstream2"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)
    
    path_dir = "data/ic_upstream3"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)

    path_dir = "data/ic_upstream4"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)

    path_src = "data/ic_upstream2/exp_100_2.csv"
    path_dir = "data/ic_downstream1"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)

    path_src = "data/ic_upstream3/exp_100_3.csv"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)
    
    path_src = "data/ic_upstream4/exp_100_4.csv"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)

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
def main():
    remove_target_from_directory("data/ic_downstream1_ImputacaoEstatistica_exp_100_2")
    remove_target_from_directory("data/ic_downstream1_ImputacaoEstatistica_exp_100_3")
    remove_target_from_directory("data/ic_downstream1_ImputacaoEstatistica_exp_100_4")
    remove_target_from_directory("data/ic_upstream2_ImputacaoEstatistica_exp_100_1")
    remove_target_from_directory("data/ic_upstream3_ImputacaoEstatistica_exp_100_1")
    remove_target_from_directory("data/ic_upstream4_ImputacaoEstatistica_exp_100_1")

if __name__ == "__main__":
    main()