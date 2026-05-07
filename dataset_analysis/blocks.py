import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from dataset_analysis.loaders import load_dataset


# =========================================================
# ----------------- METRIC UTILITIES ----------------------
# =========================================================

def effective_rank(X):
    """
    Measures intrinsic dimensionality using PCA spectrum entropy.
    Why it matters:
        - Captures how many dimensions are effectively used
        - Lower values = more redundancy / collinearity
    """
    pca = PCA().fit(X)
    s = pca.explained_variance_ratio_
    s = s[s > 0]
    return np.exp(-np.sum(s * np.log(s)))


def rv_coefficient(X, Y):
    """
    Measures similarity between two feature blocks (subspace similarity).
    Why it matters:
        - Detects linear redundancy between feature groups
        - 1 = identical subspace, 0 = independent
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    Sxy = X.T @ Y
    Sxx = X.T @ X
    Syy = Y.T @ Y

    num = np.trace(Sxy @ Sxy.T)
    den = np.sqrt(np.trace(Sxx @ Sxx.T) * np.trace(Syy @ Syy.T))
    return num / (den + 1e-12)


def cca_score(X, Y, n_comp=1):
    """
    Measures maximum linear correlation between two feature spaces.
    Why it matters:
        - Captures shared latent structure between blocks
        - High values indicate shared representation space
    """
    n_comp = min(n_comp, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_comp, max_iter=5000)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    return np.corrcoef(X_c.T, Y_c.T)[0, 1]


def distance_correlation(X, y):
    """
    Measures nonlinear dependency between features and target.
    Why it matters:
        - Detects arbitrary (nonlinear) relationships
        - 0 = independent, 1 = fully dependent
    """
    X = X - X.mean(axis=0)

    def _dcov(a, b):
        A = squareform(pdist(a.reshape(-1, 1)))
        B = squareform(pdist(b.reshape(-1, 1)))
        A -= A.mean(axis=0)[None, :] + A.mean(axis=1)[:, None] - A.mean()
        B -= B.mean(axis=0)[None, :] + B.mean(axis=1)[:, None] - B.mean()
        return np.mean(A * B)

    dcov_xy = _dcov(X.mean(axis=1), y)
    dcov_xx = _dcov(X.mean(axis=1), X.mean(axis=1))
    dcov_yy = _dcov(y, y)

    return dcov_xy / (np.sqrt(dcov_xx * dcov_yy) + 1e-12)

def distance_correlation_fixed(X, y):
    # Em vez de média, usamos o PC1 para representar o "eixo de maior informação" do bloco
    if X.shape[1] > 1:
        pca = PCA(n_components=1)
        X_rep = pca.fit_transform(X).ravel()
    else:
        X_rep = X.ravel()
        
    # Cálculo simplificado de correlação de distância (ou use correlação de Spearman para não-linear)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(X_rep, y)
    return abs(corr)


def mi_score(X, y):
    """
    Measures mutual information between features and target.
    Why it matters:
        - Captures nonlinear predictive signal
        - Higher = more informative representation
    """
    return mutual_info_regression(X, y).mean()

def mi_pca_weighted(X, y):
    if isinstance(X, np.ndarray) and X.shape[1] == 1:
        return mutual_info_regression(X, y)[0]
    
    pca_full = PCA().fit(X)
    s = pca_full.explained_variance_ratio_
    s_pos = s[s > 0]
    eff_r = max(1, int(np.ceil(np.exp(-np.sum(s_pos * np.log(s_pos))))))
    n_comp = min(eff_r, X.shape[1], X.shape[0] - 1)
    
    pca = PCA(n_components=n_comp)
    X_proj = pca.fit_transform(X)
    mi_per_pc = mutual_info_regression(X_proj, y, random_state=42)
    weights = pca.explained_variance_ratio_[:n_comp]
    weights = weights / weights.sum()  # renormaliza para n_comp componentes
    return float(np.sum(mi_per_pc * weights))

from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import numpy as np

def mi_pca_score(X, y, n_components=None):
    """
    MI between the PCA projection of X (up to effective rank) and y.
    Comparable across blocks of different sizes.
    """
    if n_components is None:
        # use effective rank as cutoff
        pca_full = PCA().fit(X)
        s = pca_full.explained_variance_ratio_
        s_pos = s[s > 0]
        eff_r = int(np.ceil(np.exp(-np.sum(s_pos * np.log(s_pos)))))
        n_components = min(eff_r, X.shape[1], X.shape[0] - 1)
    
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)
    
    # MI of each PC weighted by explained variance
    mi_per_pc = mutual_info_regression(X_proj, y)
    weights = pca.explained_variance_ratio_
    return np.sum(mi_per_pc * weights)  # variance-weighted sum
# =========================================================
# ----------------- BLOCK DECOMPOSITION -------------------
# =========================================================

def split_blocks_corrected(df_aligned, df_original, df_other_original):
    """
    df_aligned: O dataset final (ex: downstream com 166 colunas)
    df_original: O dataset original do domínio atual (ex: downstream original 101 colunas)
    df_other_original: O dataset original do OUTRO domínio (ex: upstream original)
    """
    all_cols = set(df_aligned.drop(columns=["pIC50"], errors='ignore').columns)
    orig_cols = set(df_original.drop(columns=["pIC50"], errors='ignore').columns)
    other_cols = set(df_other_original.drop(columns=["pIC50"], errors='ignore').columns)

    # 1. Intercessão: O que estava nos dois originais
    intersection = list(orig_cols.intersection(other_cols))
    
    # 2. Exclusivas: O que estava no original do domínio, mas não na intercessão
    exclusive = list(orig_cols - set(intersection))
    
    # 3. Imputadas: O que tem no alinhado que não existia no original
    imputed = list(all_cols - orig_cols)

    return intersection, exclusive, imputed

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def mean_vif(X):
    """
    Calcula o VIF médio de um bloco de features.
    VIF > 5-10 indica alta multicolinearidade.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Remove colunas constantes (evita divisao por zero)
    X = X[:, ~np.all(X == X[0, :], axis=0)]
    
    if X.shape[1] <= 1:
        return 1.0
    
    vifs = []
    # Usando amostragem se o dataset for muito grande para acelerar
    indices = np.arange(X.shape[1])
    for i in indices:
        try:
            # variance_inflation_factor espera uma constante (intercepto)
            # mas como seus dados já estão normalizados pelo StandardScaler, 
            # o cálculo direto é válido.
            vif = variance_inflation_factor(X, i)
            vifs.append(vif)
        except:
            vifs.append(1e6) # Valor de "infinito" para colinearidade perfeita
            
    return np.mean(vifs)

# =========================================================
# NOVA FUNÇÃO DE EFICIÊNCIA DE INFORMAÇÃO
# =========================================================
def calculate_efficiency(mi, rank):
    return mi / (rank + 1e-12)


# =========================================================
# ----------------- MAIN ANALYSIS -------------------------
# =========================================================

def analyze_dataset(original_paths, dataset_paths, output_path):
    results = []

    # load base datasets
    base_upstreams = {
        k: load_dataset(v) for k, v in original_paths.items()
        if "original_" in k and "downstream" not in k
    }

    base_downstream = load_dataset(original_paths["original_downstream_2"])
    for name, path in dataset_paths.items():

        df = load_dataset(path)

        y = df["pIC50"].values
        X = df.drop(columns=["pIC50"])

        # =====================================================
        # decide base reference
        # =====================================================
        if "downstream" in name:
            base = base_downstream
            # Pega qualquer upstream original para achar a intercessão real
            other_base = list(base_upstreams.values())[0] 
        else:
            key = [k for k in base_upstreams if k.endswith(name.split("_")[1])]
            base = base_upstreams[key[0]]
            other_base = base_downstream

        X_base = base.drop(columns=["pIC50"])

        # =====================================================
        # NORMALIZATION (CRÍTICO: antes de qualquer split)
        # =====================================================
        scaler = StandardScaler()

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns
        )

        X_base_scaled = pd.DataFrame(
            scaler.fit_transform(X_base),
            columns=X_base.columns
        )

        # =====================================================
        # BLOCK DECOMPOSITION (em cima do espaço normalizado)
        # =====================================================
        intersection, exclusive, imputed = split_blocks_corrected(X_scaled, X_base_scaled, other_base)

        X_int = X_scaled[intersection] if len(intersection) > 0 else np.zeros((len(df), 1))
        X_exc = X_scaled[exclusive] if len(exclusive) > 0 else np.zeros((len(df), 1))
        X_imp = X_scaled[imputed] if len(imputed) > 0 else np.zeros((len(df), 1))

        # fallback seguro (evita erro em blocos vazios)
        X_all = X_scaled.values

        # =====================================================
        # STRUCTURAL METRICS (X only)
        # =====================================================
        originalX = np.hstack([X_int.values, X_exc.values])

        eff_rank = effective_rank(X_all)
        eff_rank_int = effective_rank(X_int.values)
        X_combined = np.hstack([X_int.values, X_imp.values])
        eff_original = effective_rank(originalX)
        #eff_rank_int_imp = effective_rank(X_combined)
        eff_rank_exc = effective_rank(X_exc.values)
        eff_rank_imp = effective_rank(X_imp.values)
        delta_effective_rank__after_imputation = eff_rank - eff_original

        #rv_int_imp = rv_coefficient(X_int.values, X_imp.values)
        #rv_int_exc = rv_coefficient(X_int.values, X_exc.values)

        #cca_int_imp = cca_score(X_int.values, X_imp.values)
        #cca_int_exc = cca_score(X_int.values, X_exc.values)

        # =====================================================
        # TARGET-RELATED METRICS (X-Y)
        # =====================================================

        mi_total = mi_score(X_all, y)
        mi_imp = mi_score(X_imp.values, y)
        #mi_int_exc = mi_score(originalX, y)
        mi_int = mi_score(X_int.values, y)
        mi_exc = mi_score(X_exc.values, y)
        #delta_mi = mi_int_exc - mi_total 


        #mi_total_pca = mi_pca_score(X_all, y)
        #mi_imp_pca = mi_pca_score(X_imp.values, y)
        #mi_int_exc_pca = mi_pca_score(originalX, y)
        #mi_int_pca = mi_pca_score(X_int.values, y)
        #mi_exc_pca = mi_pca_score(X_exc.values, y)
        #delta_mi_pca = mi_int_exc_pca - mi_total_pca

        
        #mi_total_pca_weighted = mi_pca_weighted(X_all, y)
        #mi_imp_pca_weighted = mi_pca_weighted(X_imp.values, y)
        #mi_int_exc_pca_weighted = mi_pca_weighted(originalX, y)
        #mi_int_pca_weighted = mi_pca_weighted(X_int.values, y)
        #mi_exc_pca_weighted = mi_pca_weighted(X_exc.values, y)
        #delta_mi_pca_weighted = mi_int_exc_pca_weighted - mi_total_pca_weighted 



        #dcor_total = distance_correlation(X_all, y)

        #dcor_total_fixed = distance_correlation_fixed(X_all, y)
        #dcor_imp = distance_correlation(X_imp.values, y)

        #cca_xy = cca_score(X_all, y.reshape(-1, 1))
        #cca_int_y = cca_score(X_int.values, y.reshape(-1, 1))
        #cca_exc_y = cca_score(X_exc.values, y.reshape(-1, 1))
        #cca_imp_y = cca_score(X_imp.values, y.reshape(-1, 1))
        #
        ## 1. VIF dos Blocos Isolados
        #vif_imp = mean_vif(X_imp)
        #vif_int = mean_vif(X_int)
        #vif_exc = mean_vif(X_exc)

        ## 2. VIF da "Redundância Teórica" (Intercessão + Imputadas)
        ## Isso prova se as Pseudo-features são colineares com a origem delas
        #X_int_imp = np.hstack([X_int.values, X_imp.values])
        #vif_combined = mean_vif(X_int_imp)

        ## 3. Eficiência de Informação (IE)
        ie_total = calculate_efficiency(mi_total, eff_rank)
        ie_imp = calculate_efficiency(mi_imp, eff_rank_imp)
        ie_int = calculate_efficiency(mi_int, eff_rank_int)
        ie_exc = calculate_efficiency(mi_exc, eff_rank_exc)
        # =====================================================
        # PACK RESULT
        # =====================================================

        results.append({
            "dataset": name,
            "n_features": X_all.shape[1],
            "n_intersection": X_int.shape[1],
            "n_exclusive": X_exc.shape[1],
            "n_imputed": X_imp.shape[1],
            # structure
            "effective_rank": eff_rank,
            "delta_effective_rank__after_imputation": delta_effective_rank__after_imputation,
            "eff_rank_intersection": eff_rank_int,
            "eff_rank_exclusive": eff_rank_exc,
            "eff_rank_imputed": eff_rank_imp,

            # efficiency
            #"ie_total": ie_total,
            #"ie_imputed": ie_imp,
            #"ie_intersection": ie_int,
            #"ie_exclusive": ie_exc,
            #"vif_imputed": vif_imp,
            #"vif_intersection": vif_int,
            #"vif_exclusive": vif_exc,
            #"vif_combined": vif_combined,
            # redundancy
            #"rv_intersection_imputed": rv_int_imp,
            #"rv_intersection_exclusive": rv_int_exc,
            #"cca_intersection_imputed": cca_int_imp,
            #"cca_intersection_exclusive": cca_int_exc,

            # target relation
            "mi_total": mi_total,
            "mi_imputed": mi_imp,
            #"mi_intersection": mi_int,
            #"mi_exclusive": mi_exc,
            #"mi_intersection_exclusive": mi_int_exc,
            #"delta_mi": delta_mi,

            #"mi_total_pca": mi_total_pca,
            #"mi_imputed_pca": mi_imp_pca,
            #"mi_intersection_pca": mi_int_pca,
            #"mi_exclusive_pca": mi_exc_pca,
            #"mi_intersection_exclusive_pca": mi_int_exc_pca,
            #"delta_mi_pca": delta_mi_pca,

            #"mi_total_pca_weighted": mi_total_pca_weighted,
            #"mi_imputed_pca_weighted": mi_imp_pca_weighted,
            #"mi_intersection_pca_weighted": mi_int_pca_weighted,
            #"mi_exclusive_pca_weighted": mi_exc_pca_weighted,
            #"mi_intersection_exclusive_pca_weighted": mi_int_exc_pca_weighted,
            #"delta_mi_pca_weighted": delta_mi_pca_weighted,

            #"dcor_total": dcor_total,
            #"dcor_total_fixed": dcor_total_fixed,
            #"dcor_imputed": dcor_imp,
            #"cca_XY": cca_xy,
            #"cca_intersection_Y": cca_int_y,
            #"cca_exclusive_Y": cca_exc_y,
            #"cca_imputed_Y": cca_imp_y,
        })
    df_out = pd.DataFrame(results)

    # append-safe saving
    file_path = os.path.join(output_path, "analysis_blocks.csv")
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(file_path):
        df_out.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_out.to_csv(file_path, index=False)

    return df_out