import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_metrics(df, variance_threshold=0.9):
    X = df.values

    pca = PCA()
    X_scaled = StandardScaler().fit_transform(X)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    n_components_90 = np.argmax(cumulative >= variance_threshold) + 1

    return {
        "n_features": X_scaled.shape[1],
        "n_components_90": n_components_90,
        "explained_variance": explained,
        "cumulative_variance": cumulative
    }


def effective_rank(df):
    X = df.values
    X_scaled = StandardScaler().fit_transform(X)
    # covariance eigenvalues
    cov = np.cov(X_scaled, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)

    eigvals = eigvals[eigvals > 0]
    p = eigvals / eigvals.sum()

    entropy = -np.sum(p * np.log(p))
    r_eff = np.exp(entropy)

    return r_eff