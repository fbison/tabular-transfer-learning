import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

def safe_corrcoef(df):
    df = df.loc[:, df.std() > 1e-8]  # remove constants ONLY HERE

    if df.shape[1] == 0:
        return None  # or np.nan

    return np.corrcoef(df.values, rowvar=False)
def correlation_distance(df_a, df_b):
    std_a = df_a.std()
    std_b = df_b.std()

    # keep only features that are non-constant in BOTH
    mask = (std_a > 1e-8) & (std_b > 1e-8)

    df_a = df_a.loc[:, mask]
    df_b = df_b.loc[:, mask]

    if df_a.shape[1] == 0:
        return np.nan

    Ra = np.corrcoef(df_a.values, rowvar=False)
    Rb = np.corrcoef(df_b.values, rowvar=False)

    return np.linalg.norm(Ra - Rb, ord="fro")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel


def median_heuristic(X):
    dists = pairwise_distances(X, X, metric="euclidean")
    return 1.0 / (np.median(dists) ** 2 + 1e-8)


def mmd_rbf(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    XY = np.vstack([X, Y])

    # scale FIRST (important)
    scaler = StandardScaler().fit(XY)
    X = scaler.transform(X)
    Y = scaler.transform(Y)
    XY = scaler.transform(XY)

    gamma = median_heuristic(XY)

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def domain_classifier_accuracy(df_a, df_b):
    df_b_shuffled = shuffle(df_b, random_state=42)
    df_a_shuffled = shuffle(df_a, random_state=0)
    X = np.vstack([df_a_shuffled.values, df_b_shuffled.values])
    y = np.array([0] * len(df_a_shuffled) + [1] * len(df_b_shuffled))

    clf = make_pipeline(
        VarianceThreshold(threshold=1e-8),  # remove constants
        StandardScaler(),
        LogisticRegression(max_iter=6000)
    )

    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    return scores.mean()