import hydra
import torch
import json
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import numpy as np
import os

BASE_PATH = rf"C:\usp\tabular-transfer-learning"
GROUND_TRUTH_PATH = "data\high-dimension\cd_moleculas_544_833.csv"

def get_path(path: str) -> str:
    return os.path.join(BASE_PATH, path)
def evaluate_imputation(original_dir: str, imputed_dir: str, should_be_ground_truth: bool = False) -> dict:
    # -----------------------------
    # Load datasets
    # -----------------------------
    df_gt = pd.read_csv(get_path(GROUND_TRUTH_PATH), sep="|")

    original_path = os.path.join(get_path(original_dir), "ic_test_X.csv")
    df_orig = pd.read_csv(original_path, sep=",")

    imputed_files = ["ic_test_X.csv", "ic_train_X.csv", "ic_val_X.csv"]
    df_imp_list = []

    for f in imputed_files:
        path = os.path.join(get_path(imputed_dir), f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing imputed file: {path}")
        df_imp_list.append(pd.read_csv(path, sep=","))

    df_imp = pd.concat(df_imp_list, ignore_index=True)

    # -----------------------------
    # Feature separation
    # -----------------------------
    original_features = list(df_orig.columns)
    imputed_features = [col for col in df_imp.columns if col not in original_features]

    if len(imputed_features) == 0:
        raise ValueError("No imputed features found.")

    # -----------------------------
    # Column validation
    # -----------------------------
    missing_in_gt = [col for col in original_features if col not in df_gt.columns]
    if missing_in_gt:
        raise ValueError(f"Original features missing in GT: {missing_in_gt[:10]}")

    missing_imputed_in_gt = [col for col in imputed_features if col not in df_gt.columns]
    if missing_imputed_in_gt:
        raise ValueError(f"Imputed features missing in GT: {missing_imputed_in_gt[:10]}")

    # Align column order explicitly
    df_gt = df_gt[original_features + imputed_features]
    df_imp = df_imp[original_features + imputed_features]

    # Debug info
    print("Feature counts:")
    print(f"Original: {len(original_features)}")
    print(f"Imputed: {len(imputed_features)}")
    print(f"GT total: {len(df_gt.columns)}")

    # -----------------------------
    # Prepare matching
    # -----------------------------
    df_gt_grouped = df_gt.groupby(original_features, dropna=False)

    errors = []
    unmatched = []
    duplicates = []
    debug_differences = []

    # -----------------------------
    # Matching + error computation
    # -----------------------------
    for idx, row in df_imp.iterrows():
        key = tuple(row[original_features])

        try:
            matches = df_gt_grouped.get_group(key)
        except KeyError:
            unmatched.append(idx)
            continue

        if len(matches) > 1:
            duplicates.append(idx)

        gt_row = matches.iloc[0]

        imp_vals = row[imputed_features]
        gt_vals = gt_row[imputed_features]

        diff = imp_vals.values - gt_vals.values
        errors.append(diff)

        # -----------------------------
        # Strict debug mode
        # -----------------------------
        if should_be_ground_truth:
            unequal_mask = ~(np.isclose(imp_vals.values, gt_vals.values, atol=1e-12))

            if np.any(unequal_mask):
                diff_cols = np.array(imputed_features)[unequal_mask]

                debug_entry = {
                    "row_index": int(idx),
                    "num_different_features": int(np.sum(unequal_mask)),
                    "different_columns": diff_cols.tolist(),
                    "imputed_values": imp_vals[diff_cols].to_dict(),
                    "ground_truth_values": gt_vals[diff_cols].to_dict(),
                    "abs_diff": np.abs(diff[unequal_mask]).tolist(),
                    "original_features": row[original_features].to_dict(),
                }

                debug_differences.append(debug_entry)

                # Print first few mismatches immediately
                if len(debug_differences) <= 3:
                    print("\n🚨 Mismatch detected:")
                    print(json.dumps(debug_entry, indent=2))

    if len(errors) == 0:
        raise ValueError("No matched molecules found.")

    errors = np.array(errors)

    # -----------------------------
    # Metrics
    # -----------------------------
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    gt_values = df_gt[imputed_features].values
    std = np.std(gt_values)

    nrmse = np.nan if std == 0 else rmse / std

    # -----------------------------
    # Save debug log if needed
    # -----------------------------
    if should_be_ground_truth and debug_differences:
        debug_path = f"debug_differences_{os.path.basename(imputed_dir)}.json"

        with open(debug_path, "w") as f:
            json.dump(debug_differences, f, indent=2)

        print(f"\n⚠️ Found {len(debug_differences)} mismatched molecules.")
        print(f"Saved detailed log to: {debug_path}")

    # -----------------------------
    # Return results
    # -----------------------------
    stats = {
        "original_dir": original_dir,
        "imputed_dir": imputed_dir,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "NRMSE": float(nrmse),
        "num_total_imputed": len(df_imp),
        "num_matched": len(errors),
        "num_unmatched": len(unmatched),
        "num_duplicates": len(duplicates),
        "unmatched_indices": unmatched,
        "duplicate_indices": duplicates,
        "num_mismatched_rows": len(debug_differences),
    }

    print("\nFinal stats:")
    print(json.dumps(stats, indent=4))
    print("\n\n")

    return stats

@hydra.main(config_path=None)
def main(cfg: DictConfig):

    stats = []
    original_dir = f"data\ic_downstream1_Sample75"
    imputation_evaluated= "pseudo_features"#"Ground_Truth"
    isRealValues = imputation_evaluated == "Ground_Truth"
    imputed_downstream_dir = [
        f"data\ic_downstream1_Sample75_Imputation_{imputation_evaluated}_exp_100_2",
                              f"data\ic_downstream1_Sample75_Imputation_{imputation_evaluated}_exp_100_3",
                              f"data\ic_downstream1_Sample75_Imputation_{imputation_evaluated}_exp_100_4"
                              ]
    
    stats = [
        evaluate_imputation(original_dir, imputed_downstream_dir[0], isRealValues),
        evaluate_imputation(original_dir, imputed_downstream_dir[1], isRealValues),
        evaluate_imputation(original_dir, imputed_downstream_dir[2], isRealValues)
    ]
        
    for i in range(3):
        i= i+2
        original = f"data\ic_upstream{i}"
        imputed = f"data\ic_upstream{i}_Imputation_{imputation_evaluated}_exp_100_{(i) if imputation_evaluated == 'pseudo_features' else 1}"
        stats_upstream = evaluate_imputation(
            original_dir=original,
            imputed_dir=imputed,
            should_be_ground_truth=isRealValues
        )
        stats.append(stats_upstream)
    
    
    
    with open("eval.csv", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved results")


if __name__ == "__main__":
    main()