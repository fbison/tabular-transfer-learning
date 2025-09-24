import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import plotly.express as px
import plotly.io as pio
import pickle

# ---------------------------
# Helpers: parsing and mapping
# ---------------------------
def parse_sample_from_dataset_name(name):
    # tries to extract the number after "_Sample"
    m = re.search(r"_Sample(\d+)", name)
    return int(m.group(1)) if m else None

def parse_imputation_from_dataset_name(name):
    # split after _Imputation_
    m = re.search(r"_Imputation_([^_]+)", name)
    return m.group(1) if m else "unknown"

def map_strategy_from_runid(run_id):
    # adapt mapping to cover variants seen in your run_id
    rid = run_id.lower()
    if "fromscratch" in rid or "from_scratch" in rid or "_from_scratch_" in rid:
        return "FS"
    # patterns: mlpHeadFalse_freezeFalse  -> LH-E2E
    if "mlpheadfalse_freezefalse" in rid or "mlpheadfalse_freezefalse" in rid:
        return "LH-E2E"
    if "mlpheadtrue_freezefalse" in rid or "mlpheadtrue_freezefalse" in rid:
        return "MLP-E2E"
    if "mlpheadfalse_freezetrue" in rid or "mlpheadfalse_freezetrue" in rid:
        return "LH"
    if "mlpheadtrue_freezetrue" in rid or "mlpheadtrue_freezetrue" in rid:
        return "MLP"
    # try other guesses
    if "mlphead" in rid and "freezetrue" in rid:
        return "MLP"
    if "mlphead" in rid and "freezefalse" in rid:
        return "MLP-E2E"
    return "UNKNOWN"

# ---------------------------
# Load JSONL preserving seed-level RMSE
# ---------------------------
def load_results(jsonl_path, prefer="test"):
    """
    prefer: 'test'|'val'|'train' - which stats to use in priority
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            cfg = j.get("config", {})
            stats = j.get("stats", {})

            # run id might be in config or stats
            run_id = cfg.get("run_id") or stats.get("run_id") or ""

            dataset_name = cfg.get("dataset", {}).get("name", "") or stats.get("dataset", "")
            sample = parse_sample_from_dataset_name(dataset_name)
            imputation = parse_imputation_from_dataset_name(dataset_name)

            strategy = map_strategy_from_runid(run_id)

            # choose rmse: prefer test, then val, then train
            rmse = None
            if prefer == "test":
                rmse = stats.get("test_stats", {}).get("rmse")
                if rmse is None:
                    rmse = stats.get("val_stats", {}).get("rmse")
                if rmse is None:
                    rmse = stats.get("train_stats", {}).get("rmse")
            elif prefer == "val":
                rmse = stats.get("val_stats", {}).get("rmse") or stats.get("test_stats", {}).get("rmse") or stats.get("train_stats", {}).get("rmse")
            else:
                rmse = stats.get("train_stats", {}).get("rmse") or stats.get("val_stats", {}).get("rmse") or stats.get("test_stats", {}).get("rmse")
            rmse_train = stats.get("train_stats", {}).get("rmse")
            rmse_test  = stats.get("test_stats", {}).get("rmse")

            # If RMSE still None, skip
            if rmse is None:
                continue

            rows.append({
                "sample": sample,
                "strategy": strategy,
                "imputation": imputation,
                "rmse": float(rmse),
                "rmse_train": float(rmse_train) if rmse_train is not None else None,
                "rmse_test": float(rmse_test) if rmse_test is not None else None,
                "run_id": run_id
            })
    df = pd.DataFrame(rows)
    # drop rows that failed to parse sample or strategy (optional)
    df = df.dropna(subset=["sample"])
    df['sample'] = df['sample'].astype(int)
    return df

# ---------------------------
# Ranking function for one sample
# ---------------------------
def compute_ranks_for_sample(df_sample, alpha=0.05, min_seeds=2, verbose=False):
    """
    df_sample: subset of df with a single sample value; must have columns strategy, imputation, rmse
    Returns dict: {(strategy, imputation): rank}
    """
    # list of configs
    configs_df = df_sample[['strategy','imputation']].drop_duplicates().reset_index(drop=True)
    keys = [tuple(x) for x in configs_df.values.tolist()]

    # seed counts per config
    counts = {k: df_sample[(df_sample['strategy']==k[0]) & (df_sample['imputation']==k[1])].shape[0] for k in keys}
    if verbose:
        print("Sample:", df_sample['sample'].iloc[0], "config seed counts:", counts)

    # if any config has too few seeds, fall back to mean-based dense ranking
    if any(v < min_seeds for v in counts.values()):
        if verbose:
            print("Not enough seeds for at least one config (min_seeds={}): falling back to mean-based ranking.".format(min_seeds))
        means = {k: df_sample[(df_sample['strategy']==k[0]) & (df_sample['imputation']==k[1])]['rmse'].mean() for k in keys}
        # dense ranking by mean (smaller rmse -> better rank 1)
        sorted_keys = sorted(keys, key=lambda k: means[k])
        ranks = {}
        current_rank = 1
        prev = None
        for k in sorted_keys:
            if prev is None or not np.isclose(means[k], prev):
                ranks[k] = current_rank
                prev = means[k]
                current_rank += 1
            else:
                ranks[k] = current_rank - 1
        return ranks

    # Otherwise perform iterative Mann-Whitney grouping as described
    ranks = {}
    assigned = set()
    current_rank = 1
    while len(assigned) < len(keys):
        unassigned = [k for k in keys if k not in assigned]
        # choose the best (lowest mean rmse) among unassigned as reference
        best = min(unassigned, key=lambda k: df_sample[(df_sample['strategy']==k[0]) & (df_sample['imputation']==k[1])]['rmse'].mean())

        same_rank = []
        for k in unassigned:
            a = df_sample[(df_sample['strategy']==best[0]) & (df_sample['imputation']==best[1])]['rmse'].values
            b = df_sample[(df_sample['strategy']==k[0]) & (df_sample['imputation']==k[1])]['rmse'].values
            try:
                stat, p = mannwhitneyu(a, b, alternative="less")
            except Exception as e:
                # if mannwhitney fails (e.g., all-values-equal edge cases), treat as non-significant
                p = 1.0
            if verbose:
                print(f"Compare best {best} vs {k} -> p={p:.4f}")
            if p >= alpha:
                same_rank.append(k)

        for k in same_rank:
            ranks[k] = current_rank
            assigned.add(k)
        current_rank += 1

    return ranks

# ---------------------------
# Build heatmap-ready DataFrame
# ---------------------------
def build_rank_table(df, alpha=0.05, min_seeds=2, verbose=False):
    results = []
    for sample, dfg in df.groupby('sample'):
        ranks = compute_ranks_for_sample(dfg, alpha=alpha, min_seeds=min_seeds, verbose=verbose)
        for (strategy, imputation), rank in ranks.items():
            results.append({"sample": sample, "strategy": strategy, "imputation": imputation, "rank": rank})
    rank_df = pd.DataFrame(results)
    return rank_df

def plot_and_save_heatmap(rank_df, out_dir, strategies_order=None, imputations_order=None):
    os.makedirs(out_dir, exist_ok=True)
    
    # Strategy and imputation ordering
    if strategies_order is None:
        strategies_order = ['FS', 'LH-E2E', 'MLP-E2E', 'LH', 'MLP']
    if imputations_order is None:
        imputations_order = sorted(rank_df['imputation'].unique())
    
    n_imputations = len(imputations_order)

    # Determine global min/max rank for consistent color scale
    vmin = rank_df["rank"].min()
    vmax = rank_df["rank"].max()

    # Equal width for each imputation subplot
    fig, axes = plt.subplots(
        1, n_imputations,
        figsize=(2 * len(strategies_order) * n_imputations, 4),
        sharey=True,
        gridspec_kw={"width_ratios": [1] * n_imputations}
    )

    if n_imputations == 1:
        axes = [axes]

    for ax, imp in zip(axes, imputations_order):
        df_imp = rank_df[rank_df['imputation'] == imp].pivot(
            index="sample", columns="strategy", values="rank"
        )
        df_imp = df_imp[strategies_order]

        sns.heatmap(
            df_imp,
            ax=ax,
            annot=True,
            fmt="d",
            cmap="RdYlBu_r",
            vmin=vmin, vmax=vmax,   # keep same color scale
            cbar=ax == axes[-1],
            cbar_kws={"label": "Rank"},
            annot_kws={"fontsize": 10},
            linewidths=0.5,
            linecolor="white"
        )

        # Titles and labels
        ax.set_title(imp, fontsize=14, fontname="Times New Roman")
        ax.set_xlabel("")
        ax.set_ylabel("Num Samples", fontsize=12, fontname="Times New Roman")

        # Clean x-axis labels
        new_labels = [lab.replace(f"{imp}-", "") for lab in df_imp.columns]
        ax.set_xticklabels(
            new_labels, rotation=45, ha="right", fontsize=10, fontname="Times New Roman"
        )

        ax.set_yticklabels(
            df_imp.index, rotation=0, fontsize=10, fontname="Times New Roman"
        )

        # Force same aspect for all
        ax.set_aspect("equal")

    plt.tight_layout()
    path_complete = os.path.join(out_dir, "heatmap")
    plt.savefig((path_complete + ".png"), dpi=300, bbox_inches="tight")
    plt.savefig((path_complete + ".pdf"), bbox_inches="tight")
    plt.savefig((path_complete + ".svg"), bbox_inches="tight")

    with open((path_complete + ".fig.pickle"), "wb") as f:
        pickle.dump(fig, f)
    plt.savefig("heatmap.eps", format="eps", bbox_inches="tight")

    plt.show()

def plot_BoxPlots_overfitting(df, out_dir, strategies_order=None, imputations_order=None):
    """
    Gera boxplots do overfitting gap (test_rmse - train_rmse) para cada estratégia e imputação.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "rmse_train" not in df.columns or "rmse_test" not in df.columns:
        raise ValueError("O DataFrame precisa conter as colunas rmse_train e rmse_test.")

    df = df.copy()
    df["overfit_gap"] = df["rmse_test"] - df["rmse_train"]

    if strategies_order is None:
        strategies_order = ['FS', 'LH-E2E', 'MLP-E2E', 'LH', 'MLP']
    if imputations_order is None:
        imputations_order = sorted(df['imputation'].unique())

    n_imputations = len(imputations_order)
    fig, axes = plt.subplots(
        1, n_imputations,
        figsize=(5 * n_imputations, 5),
        sharey=True
    )
    if n_imputations == 1:
        axes = [axes]

    for ax, imp in zip(axes, imputations_order):
        df_imp = df[df["imputation"] == imp]

        sns.boxplot(
            data=df_imp,
            x="strategy",
            y="overfit_gap",
            order=strategies_order,
            ax=ax,
            palette="Set2"
        )
        sns.stripplot(
            data=df_imp,
            x="strategy",
            y="overfit_gap",
            order=strategies_order,
            ax=ax,
            color="black",
            alpha=0.5,
            jitter=True,
            dodge=True
        )

        ax.set_title(f"Imputation: {imp}", fontsize=14, fontname="Times New Roman")
        ax.set_xlabel("Strategy", fontsize=12, fontname="Times New Roman")
        ax.set_ylabel("Overfitting gap (Test RMSE - Train RMSE)", fontsize=12, fontname="Times New Roman")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path_complete = os.path.join(out_dir, "boxplot_overfitting")
    plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(path_complete + ".pdf", bbox_inches="tight")
    plt.savefig(path_complete + ".svg", bbox_inches="tight")
    with open(path_complete + ".fig.pickle", "wb") as f:
        pickle.dump(fig, f)

    plt.show()

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # path to folder containing results.jsonl
    path = r"C:\usp\tabular-transfer-learning\outputs\transfer-learning-from-upstream\ft-transformer\ic_upstream3"
    jsonl = os.path.join(path, "results.jsonl")

    df = load_results(jsonl, prefer="test")

    # rank_df = build_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)
    # plot_and_save_heatmap(rank_df, out_dir=path)
    plot_BoxPlots_overfitting(df, out_dir=path)
