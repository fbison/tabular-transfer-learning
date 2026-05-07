import os
import pandas as pd
from dataset_analysis.blocks import analyze_dataset
from dataset_analysis.loaders import load_dataset
from dataset_analysis.validators import validate_no_nan
from dataset_analysis.dimensionality_metrics import compute_pca_metrics, effective_rank
from dataset_analysis.alignment_metrics import correlation_distance, domain_classifier_accuracy, mmd_rbf
from dataset_analysis.reconstruction_metrics import reconstruction_error
from dataset_analysis.plots import plot_pca_curve
from dataset_analysis.pairing import parse_key, is_valid_pair

def check_column_order(df1, df2, name1="df1", name2="df2"):
    """
    Verifica se dois DataFrames possuem as mesmas colunas na mesma ordem.
    Se não, imprime os erros encontrados.
    """

    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    # Verifica se têm as mesmas colunas (ignorando ordem)
    if set(cols1) != set(cols2):
        missing_in_df2 = set(cols1) - set(cols2)
        missing_in_df1 = set(cols2) - set(cols1)

        print("❌ DataFrames não possuem as mesmas colunas.")
        if missing_in_df2:
            print(f"Colunas em {name1} e não em {name2}: {missing_in_df2}")
        if missing_in_df1:
            print(f"Colunas em {name2} e não em {name1}: {missing_in_df1}")
        return False

    # Verifica ordem
    if cols1 != cols2:
        print("⚠️ Mesmas colunas, mas ordem diferente.")
        for i, (c1, c2) in enumerate(zip(cols1, cols2)):
            if c1 != c2:
                print(f"Primeira diferença na posição {i}: {name1}='{c1}' vs {name2}='{c2}'")
                break
        return False

    print("✅ DataFrames possuem mesmas colunas na mesma ordem.")
    return True

def run_analysis(dataset_paths, output_path, original_paths=[]):
    os.makedirs(output_path, exist_ok=True)

    results_dim = []
    results_recon = []
    results_align = []

    datasets = {}

    if len(original_paths) > 0:
        dataset_paths_to_pca = {**dataset_paths, **original_paths}
    else:
        dataset_paths_to_pca = dataset_paths
    # --- Load all datasets ---
    for name, path in dataset_paths_to_pca.items():
        df = load_dataset(path)

        validate_no_nan(df, name)

        datasets[name] = df

        # --- Dimensionality ---
        pca_metrics = compute_pca_metrics(df)
        r_eff = effective_rank(df)

        results_dim.append({
            "dataset": name,
            "n_features": pca_metrics["n_features"],
            "n_components_90": pca_metrics["n_components_90"],
            "effective_rank": r_eff
        })

        plot_pca_curve(
            pca_metrics["cumulative_variance"],
            os.path.join(output_path, f"pca_{name}")
        )

    pd.DataFrame(results_dim).to_csv(os.path.join(output_path, "dimensionality.csv"), index=False) 
    # --- Reconstruction (GT vs imputed) ---
    # Assumes naming convention contains "gt"
    gt_datasets = {k: v for k, v in datasets.items() if "gt" in k.lower()}

    for gt_name, df_gt in gt_datasets.items():
        for name, df_imp in datasets.items():
            if name == gt_name:
                continue

            try:
                metrics = reconstruction_error(df_gt, df_imp)

                results_recon.append({
                    "gt": gt_name,
                    "imp": name,
                    **metrics
                })
            except:
                continue

    # --- Pairwise alignment ---
    keys = list(datasets.keys())

    meta_map = {k: parse_key(k) for k in datasets.keys()}
    processed = set()

    for a in datasets:
        for b in datasets:
            if a == b:
                continue
            pair_id = tuple(sorted([a, b]))
            if pair_id in processed:
                continue
            processed.add(pair_id)

            meta_a = meta_map[a]
            meta_b = meta_map[b]

            if not is_valid_pair(meta_a, meta_b):
                continue

            df_a = datasets[a]
            df_b = datasets[b]
            check_column_order(df_a, df_b, name1=a, name2=b)
            try:
                corr_dist = correlation_distance(df_a, df_b)
                domain_acc = domain_classifier_accuracy(df_a, df_b)

                mmd = mmd_rbf(df_a, df_b)
                results_align.append({
                    "dataset_a": a,
                    "dataset_b": b,
                    "id": meta_a["id"],
                    "imputation": meta_a["imputation"],
                    "corr_distance": corr_dist,
                    "domain_acc": domain_acc,
                    "mmd_rbf": mmd
                })
            except Exception as e:
                print(f"ERROR in pair {a} vs {b}: {e}")
                continue

    # --- Save results ---
    pd.DataFrame(results_recon).to_csv(os.path.join(output_path, "reconstruction.csv"), index=False)
    pd.DataFrame(results_align).to_csv(os.path.join(output_path, "alignment.csv"), index=False)


if __name__ == "__main__":
    base_path = rf"C:\usp\tabular-transfer-learning\data"
    dataset_paths = {
        # --- Downstream ---
        "downstream_2_gaussian": rf"{base_path}\ic_downstream1_Sample75_Imputation_Gaussian_exp_100_2",
        "downstream_3_gaussian": rf"{base_path}\ic_downstream1_Sample75_Imputation_Gaussian_exp_100_3",
        "downstream_4_gaussian": rf"{base_path}\ic_downstream1_Sample75_Imputation_Gaussian_exp_100_4",

        "downstream_2_gt": rf"{base_path}\ic_downstream1_Sample75_Imputation_Ground_Truth_exp_100_2",
        "downstream_3_gt": rf"{base_path}\ic_downstream1_Sample75_Imputation_Ground_Truth_exp_100_3",
        "downstream_4_gt": rf"{base_path}\ic_downstream1_Sample75_Imputation_Ground_Truth_exp_100_4",

        "downstream_2_mean": rf"{base_path}\ic_downstream1_Sample75_Imputation_Mean_exp_100_2",
        "downstream_3_mean": rf"{base_path}\ic_downstream1_Sample75_Imputation_Mean_exp_100_3",
        "downstream_4_mean": rf"{base_path}\ic_downstream1_Sample75_Imputation_Mean_exp_100_4",

        "downstream_2_pseudo": rf"{base_path}\ic_downstream1_Sample75_Imputation_pseudo_features_exp_100_2",
        "downstream_3_pseudo": rf"{base_path}\ic_downstream1_Sample75_Imputation_pseudo_features_exp_100_3",
        "downstream_4_pseudo": rf"{base_path}\ic_downstream1_Sample75_Imputation_pseudo_features_exp_100_4",

        # --- Upstream ---
        "upstream_2_gaussian": rf"{base_path}\ic_upstream2_Imputation_Gaussian_exp_100_1",
        "upstream_3_gaussian": rf"{base_path}\ic_upstream3_Imputation_Gaussian_exp_100_1",
        "upstream_4_gaussian": rf"{base_path}\ic_upstream4_Imputation_Gaussian_exp_100_1",

        "upstream_2_gt": rf"{base_path}\ic_upstream2_Imputation_Ground_Truth_exp_100_1",
        "upstream_3_gt": rf"{base_path}\ic_upstream3_Imputation_Ground_Truth_exp_100_1",
        "upstream_4_gt": rf"{base_path}\ic_upstream4_Imputation_Ground_Truth_exp_100_1",

        "upstream_2_mean": rf"{base_path}\ic_upstream2_Imputation_Mean_exp_100_1",
        "upstream_3_mean": rf"{base_path}\ic_upstream3_Imputation_Mean_exp_100_1",
        "upstream_4_mean": rf"{base_path}\ic_upstream4_Imputation_Mean_exp_100_1",

        "upstream_2_pseudo": rf"{base_path}\ic_upstream2_Imputation_pseudo_features_exp_100_1",
        "upstream_3_pseudo": rf"{base_path}\ic_upstream3_Imputation_pseudo_features_exp_100_1",
        "upstream_4_pseudo": rf"{base_path}\ic_upstream4_Imputation_pseudo_features_exp_100_1",
    }
    output_path = "./dataset_analysis/outputs"
    original_paths = {
        "original_2": rf"{base_path}\ic_upstream2",
        "original_3": rf"{base_path}\ic_upstream3",
        "original_4": rf"{base_path}\ic_upstream4",
        "original_downstream_2": rf"{base_path}\ic_downstream1_Sample75",
    }
    analyze_dataset(original_paths, dataset_paths, output_path)
    #run_analysis(dataset_paths, output_path)