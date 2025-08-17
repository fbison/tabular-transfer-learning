from deep_tabular.utils.ic_tools import imputar_colunas_faltantes


def main():
    # Example usage of imputar_colunas_faltantes
    path_src = "data/ic_downstream1/exp_100_1.csv"
    #path_dir = "data/ic_upstream2"
    #imputar_colunas_faltantes(path_dir, path_src, seed=42)
    
    #path_dir = "data/ic_upstream3"
    #imputar_colunas_faltantes(path_dir, path_src, seed=42)

    #path_dir = "data/ic_upstream4"
    #imputar_colunas_faltantes(path_dir, path_src, seed=42)

    path_src = "data/ic_upstream2/exp_100_2.csv"
    path_dir = "data/ic_downstream1"
    #imputar_colunas_faltantes(path_dir, path_src, seed=42)

    path_src = "data/ic_upstream3/exp_100_3.csv"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)
    
    path_src = "data/ic_upstream4/exp_100_4.csv"
    imputar_colunas_faltantes(path_dir, path_src, seed=42)

    print("Missing columns imputed successfully.")

if __name__ == "__main__":
    main()