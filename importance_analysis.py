import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

def compute_mi_importance(X, y, normalize=True):
    """Retorna MI normalizado via MinMax."""
    mi = mutual_info_regression(X, y, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    if normalize:
        mi = MinMaxScaler().fit_transform(mi.reshape(-1,1)).flatten()
    return mi

def feature_stability_pipeline(
    csv_path,
    target_name,
    n_subsets=30,
    subset_sizes=(0.5, 0.7, 0.9),  # percentuais do dataset
    random_state=42,
    output_csv="feature_stability_mi.csv",
    normalize=True
):

    # -----------------------------
    # 1. Ler dataset
    # -----------------------------
    df = pd.read_csv(csv_path, sep="|")
    df = df.dropna()  # opcional
        # converter target para numérico (caso esteja como string)
    df[target_name] = pd.to_numeric(df[target_name], errors="coerce")

    # remover linhas onde o target não é numérico
    df = df.dropna(subset=[target_name])

    # remover TODAS colunas não numéricas (exceto target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_name not in numeric_cols:
        raise ValueError("Target não é numérico!")

    df = df[numeric_cols]
    
    y = df[target_name]
    X = df.drop(columns=[target_name]).select_dtypes(include=[np.number])
    print(f"Número de features numéricas consideradas: {X.shape[1]}")

    features = X.columns.tolist()

    # -----------------------------
    # 2. MI global
    # -----------------------------
    global_mi = compute_mi_importance(X, y, normalize)

    # ordem de importância
    sorted_idx = np.argsort(global_mi)[::-1]
    sorted_features = [features[i] for i in sorted_idx]

    # -----------------------------
    # 3. Preparar estruturas para registrar estabilidade
    # -----------------------------
    mi_values_map = {feat: [] for feat in features}

    # -----------------------------
    # 4. Rodar subsets aleatórios
    # -----------------------------
    rng = np.random.RandomState(random_state)
    total_sizes = len(subset_sizes) * n_subsets
    current_run = 0
    for size in subset_sizes:

        # Determinar quantas vezes rodar para este tamanho
        runs = 1 if size == 1 else n_subsets
        n_samples = len(df) if size == 1 else int(len(df) * size)

        for _ in range(runs):
            current_run += 1
            # Criar subset (caso size == 1, é só o df inteiro sem custo extra)
            if size == 1:
                df_sub = df
            else:
                df_sub = resample(
                    df,
                    replace=False,
                    n_samples=n_samples,
                    random_state=rng
                )

            X_sub = df_sub.drop(columns=[target_name])
            y_sub = df_sub[target_name]

            # MI do subset
            mi_sub = compute_mi_importance(X_sub, y_sub)

            # Registrar valores
            for feat, val in zip(features, mi_sub):
                mi_values_map[feat].append(val)
        print(f"Run {current_run}/{total_sizes} completed for subset size {size} ({n_samples} samples).")

    # -----------------------------
    # 5. Construir dataframe final
    # -----------------------------
    rows = []
    for feat in sorted_features:
        vals = mi_values_map[feat]
        rows.append([
            feat,
            global_mi[features.index(feat)],
            np.mean(vals),
            np.median(vals),
            np.var(vals)
        ])

    result_df = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "importance_global",
            "importance_mean",
            "importance_median",
            "importance_variance"
        ]
    )

    # salvar
    result_df.to_csv(output_csv, index=False)
    print(f"Arquivo gerado: {output_csv}")

    return result_df

# ---------------------------------------------------------
# Weighted Jaccard
# ---------------------------------------------------------
def weighted_jaccard(vec_a, vec_b):
    min_sum = (pd.concat([vec_a, vec_b], axis=1).min(axis=1)).sum()
    max_sum = (pd.concat([vec_a, vec_b], axis=1).max(axis=1)).sum()
    if max_sum == 0:
        return 0.0
    return min_sum / max_sum

# ---------------------------------------------------------
# Constrói vetores ponderados
# ---------------------------------------------------------
def build_weighted_vector(global_imp, features_selected):
    features_selected = list(features_selected)  # <-- FIX
    vec = pd.Series(0.0, index=global_imp.index)
    if len(features_selected) > 0:
        vec.loc[features_selected] = global_imp.loc[features_selected]
    return vec


# ---------------------------------------------------------
# Calcula estatísticas pedidas
# ---------------------------------------------------------
def compare_sets(global_imp, upstream, downstream):

    upstream = set(upstream)
    downstream = set(downstream)

    # Vetores e Jaccard
    vec_up = build_weighted_vector(global_imp, upstream)
    vec_down = build_weighted_vector(global_imp, downstream)
    wj = weighted_jaccard(vec_up, vec_down)

    # Partições
    only_up = upstream - downstream
    only_down = downstream - upstream
    inter = upstream & downstream

    def stats(features):
        if len(features) == 0:
            return 0.0, 0, 0.0
        values = global_imp.loc[list(features)]
        return values.sum(), len(values), values.mean()

    sum_up, cnt_up, mean_up = stats(upstream)
    sum_down, cnt_down, mean_down = stats(downstream)
    sum_ou, cnt_ou, mean_ou = stats(only_up)
    sum_od, cnt_od, mean_od = stats(only_down)
    sum_i, cnt_i, mean_i = stats(inter)

    # ---- Novas métricas ----
    transfer_ratio_up = sum_i / sum_up if sum_up > 0 else 0.0
    transfer_ratio_down = sum_i / sum_down if sum_down > 0 else 0.0
    dispersion_index = 1 - (sum_i / (sum_ou + sum_od)) if (sum_ou + sum_od) > 0 else 0.0
    efficiency_index = mean_i / mean_up if mean_up > 0 else 0.0

    return {
        "sum_importance_upstream": sum_up,
        "count_upstream": cnt_up,
        "mean_importance_upstream": mean_up,
        "sum_importance_downstream": sum_down,
        "count_downstream": cnt_down,
        "mean_importance_downstream": mean_down,
        "weighted_jaccard": wj,
        "sum_importance_only_upstream": sum_ou,
        "count_only_upstream": cnt_ou,
        "mean_importance_only_upstream": mean_ou,
        "sum_importance_only_downstream": sum_od,
        "count_only_downstream": cnt_od,
        "mean_importance_only_downstream": mean_od,
        "sum_importance_intersection": sum_i,
        "count_intersection": cnt_i,
        "mean_importance_intersection": mean_i,
        "transfer_ratio_upstream": transfer_ratio_up,
        "transfer_ratio_downstream": transfer_ratio_down,
        "dispersion_index": dispersion_index,
        "efficiency_index": efficiency_index
    }

# ---------------------------------------------------------
# Função principal
# ---------------------------------------------------------
def compare_all(csv_importances_path, upstream_lists, downstream_list, output_csv_path):
    df = pd.read_csv(csv_importances_path)
    df = df.set_index("feature")

    global_imp = df["importance_global"]
    global_set = set(global_imp.index)

    results = []
    for i, upstream in enumerate(upstream_lists, start=2):

        # 1) Upstream x Downstream (já existente)
        stats_ud = compare_sets(global_imp, upstream, downstream_list)
        stats_ud["scenario"] = "upstream_vs_downstream"
        stats_ud["upstream_id"] = f"upstream_{i}"
        stats_ud["downstream_id"] = "downstream"
        results.append(stats_ud)

        # 2) Upstream x Global
        stats_ug = compare_sets(global_imp, upstream, global_set)
        stats_ug["scenario"] = "upstream_vs_global"
        stats_ug["upstream_id"] = f"upstream_{i}"
        stats_ug["downstream_id"] = "global"
        results.append(stats_ug)

        # 3) (Upstream ∪ Downstream) x Global
        union_set = set(upstream) | set(downstream_list)
        stats_union_g = compare_sets(global_imp, union_set, global_set)
        stats_union_g["scenario"] = "union_vs_global"
        stats_union_g["upstream_id"] = f"upstream_{i}"
        stats_union_g["downstream_id"] = "global"
        results.append(stats_union_g)

    out = pd.DataFrame(results)
    out.to_csv(output_csv_path, index=False)
    print(f"Resultado salvo em {output_csv_path}")


import os
if __name__ == "__main__":
    path = rf"C:\usp\tabular-transfer-learning\data"
    output_importances_csv = os.path.join(path, rf"high-dimension\mi_feature_stability_bruto.csv")
    df_result = feature_stability_pipeline(
        csv_path=os.path.join(path, rf"high-dimension\cd_moleculas_544_833.csv"),
        target_name="pIC50",
        n_subsets=30,
        subset_sizes=(0.2, 0.4, 0.6, 0.8, 1.0),
        output_csv=output_importances_csv,
        normalize=False
    )

    downstream_features = ["SpDiam_A","AATS5d","AATS7s","AATS8s","AATS1i","AATS2i","AATS3i","AATS4i","AATS6i","ATSC1dv","ATSC8dv","ATSC1d","ATSC7d","AATSC0v","MATS6s","MATS7s","MATS8s","GATS2c","GATS3c","GATS8c","GATS1dv","GATS3dv","GATS4dv","GATS6dv","GATS7dv","GATS8dv","GATS1d","GATS2d","GATS5d","GATS2s","GATS3s","GATS4s","GATS6s","GATS1v","GATS2v","GATS5p","GATS6p","GATS7p","GATS1i","GATS2i","GATS3i","GATS4i","GATS5i","GATS7i","GATS8i","RNCG","RPCG","Xc-3dv","Xc-5dv","Xc-6dv","AXp-0d","SdssC","SsNH2","SdO","SssO","SssS","SaaS","SddssS","MAXaaCH","AETA_alpha","AETA_beta_ns_d","ETA_dAlpha_B","ETA_epsilon_5","ETA_dEpsilon_D","IC1","CIC1","CIC2","ZMIC2","PEOE_VSA1","PEOE_VSA4","PEOE_VSA6","PEOE_VSA9","SMR_VSA1","SMR_VSA3","SMR_VSA4","SMR_VSA9","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA10","EState_VSA4","EState_VSA5","VSA_EState8","VSA_EState9","AMID_C","TopoPSA(NO)","GGI6","GGI7","JGI4","RNCS","Mor02m","Mor03m","Mor13m","Mor26m","Mor30m","Mor31m","MOMI-Z","MLOGP","ESOL_Solubility_(mg/ml)","Ali_Log_S"]
    upstream2_features = ["SpMax_A","VE1_A","AATS8dv","AATS8s","AATS2i","ATSC1dv","ATSC8d","ATSC0p","ATSC0i","MATS1c","MATS2s","MATS3s","MATS6s","MATS7s","MATS8s","GATS4c","GATS1dv","GATS5dv","GATS7dv","GATS6d","GATS7d","GATS2s","GATS3s","GATS2v","GATS3v","GATS1p","GATS6p","GATS3i","GATS6i","GATS8i","BCUTc-1h","BCUTd-1l","BCUTs-1h","RPCG","Xch-5d","Xch-7d","Xc-5d","Xc-5dv","Xc-6dv","AXp-1d","SdssC","SaasC","SaaaC","SssssC","SsNH2","SssNH","SsOH","SssO","SdS","SddssS","MAXaaCH","AETA_beta_s","AETA_eta_L","AETA_eta_F","ETA_epsilon_5","IC1","IC2","CIC2","ZMIC1","PEOE_VSA1","PEOE_VSA2","PEOE_VSA9","SlogP_VSA1","SlogP_VSA2","SlogP_VSA10","EState_VSA1","EState_VSA2","EState_VSA3","EState_VSA6","EState_VSA9","VSA_EState3","VSA_EState7","VSA_EState8","MDEC-33","TopoPSA(NO)","GGI3","GGI5","GGI6","GGI7","GGI8","GGI9","JGI2","JGI5","FPSA3","RPCS","Mor02m","Mor03m","Mor06m","Mor08m","Mor11m","Mor13m","Mor16m","Mor23m","XLOGP3","Silicos-IT_Log_P","ESOL_Log_S","ESOL_Solubility_(mg/ml)","Ali_Log_S","Ali_Solubility_(mg/ml)","Silicos-IT_Solubility_(mg/ml)"]
    upstream3_features= ["SpDiam_A","AATS3d","AATS2s","AATS3v","AATS1i","AATS2i","AATS3i","AATS4i","ATSC8dv","ATSC1d","ATSC8d","ATSC0i","AATSC0c","MATS1c","MATS2s","MATS7s","GATS1c","GATS2c","GATS5c","GATS7c","GATS8c","GATS2dv","GATS5dv","GATS7dv","GATS8dv","GATS1d","GATS3d","GATS5d","GATS6d","GATS7d","GATS8d","GATS2s","GATS6s","GATS6v","GATS3p","GATS5p","GATS1i","GATS6i","BCUTs-1h","BCUTs-1l","BCUTi-1h","BCUTi-1l","RPCG","Xch-6d","Xc-5d","Xc-3dv","Xc-4dv","AXp-1d","SsCH3","SdCH2","SdsCH","SsssCH","SaaNH","SdsN","SsOH","SssO","SdS","SaaS","SddssS","MINaasC","AETA_alpha","AETA_beta_ns_d","ETA_epsilon_5","ETA_dEpsilon_C","IC2","ZMIC2","PEOE_VSA1","PEOE_VSA2","PEOE_VSA3","PEOE_VSA6","PEOE_VSA7","PEOE_VSA11","SMR_VSA5","SMR_VSA6","SlogP_VSA2","SlogP_VSA3","EState_VSA1","EState_VSA2","EState_VSA3","EState_VSA6","VSA_EState7","AMID_C","TopoPSA(NO)","GGI5","GGI7","GGI8","GGI9","JGI3","TSRW10","TASA","Mor02m","Mor10m","Mor11m","Mor12m","Mor16m","Mor21m","Mor22m","Mor23m","iLOGP","Silicos-IT_Log_P"]
    upstream4_features=["SpMax_A","VE1_A","AATS7d","AATS6s","AATS8s","AATS3p","AATS1i","AATS3i","AATS4i","AATS6i","ATSC1d","AATSC0c","MATS1c","MATS5s","MATS6s","MATS7s","GATS1c","GATS6c","GATS7c","GATS1dv","GATS3dv","GATS5dv","GATS6dv","GATS3d","GATS4d","GATS6d","GATS1s","GATS2s","GATS8s","GATS2v","GATS3v","GATS7v","GATS4p","GATS8i","BCUTc-1l","BCUTd-1l","BCUTs-1h","BCUTs-1l","RNCG","RPCG","Xc-5dv","Xc-6dv","SsCH3","SdCH2","SdsCH","SsssCH","SaaaC","SssssC","SssNH","SsssN","SsOH","SssO","SdS","SsCl","MAXaasC","MINaaCH","ETA_shape_y","AETA_beta_s","AETA_eta_L","ETA_dEpsilon_D","fMF","ZMIC2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA10","SMR_VSA1","SMR_VSA9","SlogP_VSA1","SlogP_VSA2","SlogP_VSA4","SlogP_VSA5","SlogP_VSA10","EState_VSA1","VSA_EState1","VSA_EState8","VSA_EState9","MDEC-22","MDEC-23","TopoPSA(NO)","GGI3","GGI4","GGI6","GGI8","GGI9","GGI10","JGI2","FNSA1","RASA","Mor02m","Mor08m","Mor10m","Mor12m","Mor23m","Mor24m","MOMI-Z","Silicos-IT_Log_P","ESOL_Solubility_(mg/ml)"]


    compare_all(
        csv_importances_path=output_importances_csv,
        upstream_lists=[upstream2_features, upstream3_features, upstream4_features],
        downstream_list=downstream_features,
        output_csv_path=os.path.join(path, "comparacao_output_bruto.csv")
    )

