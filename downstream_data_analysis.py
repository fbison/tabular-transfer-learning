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

            # If RMSE still None, skip
            if rmse is None:
                continue

            rows.append({
                "sample": sample,
                "strategy": strategy,
                "imputation": imputation,
                "rmse": float(rmse),
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
    
    # Escolha de ordem
    if strategies_order is None:
        strategies_order = ['FS', 'LH-E2E', 'MLP-E2E', 'LH', 'MLP']
    if imputations_order is None:
        imputations_order = sorted(rank_df['imputation'].unique())
    
    # Criar pivot table com MultiIndex (strategy, imputation)
    pivot = rank_df.pivot_table(index='sample', columns=['imputation', 'strategy'], values='rank')
    pivot = pivot.reindex(index=sorted(pivot.index), columns=pd.MultiIndex.from_product([imputations_order, strategies_order]))
    
    # ---------------- Heatmap com Seaborn ----------------
    plt.figure(figsize=(max(8, len(strategies_order)*len(imputations_order)*0.8), max(4, len(pivot.index)*0.5)))
    
    # Plotando heatmap
    sns.heatmap(
        pivot, 
        annot=True, fmt=".0f", linewidths=0.5, linecolor='gray', 
        cbar_kws={"label": "Rank"}, cmap="RdYlBu_r",
        vmin=1, vmax=int(np.nanmax(rank_df['rank'])) if not rank_df['rank'].isna().all() else None
    )
        
    # Adicionar linhas verticais de separação entre imputações
    for i in range(1, len(imputations_order)):
        plt.axvline(i * len(strategies_order), color='black', lw=1.2)
        
    # Personalizar fonte e rótulos
    plt.ylabel("Num Samples", fontsize=12, fontname="Times New Roman")
    plt.xlabel("", fontsize=12, fontname="Times New Roman")
    
    # Ajuste das colunas: mostrar apenas estratégia
    plt.xticks(rotation=45, ha='right', fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=12, fontname="Times New Roman")
    
    # Títulos dos blocos de imputação
    for i, imputation in enumerate(imputations_order):
        col_start = i * len(strategies_order)
        col_end = (i+1) * len(strategies_order) - 1
        plt.text((col_start+col_end)/2 + 0.5, -0.8, imputation.capitalize(), ha='center', va='bottom', fontsize=12, fontname="Times New Roman", fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar PNG
    png_path = os.path.join(out_dir, "heatmap.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print("Saved PNG:", png_path)
    
    # ---------------- Heatmap Interativo com Plotly ----------------
    pivot_for_plotly = pivot.copy()
    pivot_for_plotly.columns = [c[1] for c in pivot_for_plotly.columns]  # apenas estratégia no x
    fig = px.imshow(
        pivot_for_plotly.values,
        labels=dict(x="Strategy", y="Num Samples", color="Rank"),
        x=pivot_for_plotly.columns,
        y=pivot_for_plotly.index.astype(str),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        origin='lower'
    )
    # Adiciona títulos para blocos de imputação no Plotly
    for i, imputation in enumerate(imputations_order):
        col_start = i * len(strategies_order)
        col_end = (i+1) * len(strategies_order) - 1
        fig.add_annotation(
            x=(col_start + col_end)/2, y=-0.5, text=imputation.capitalize(),
            showarrow=False, font=dict(family="Times New Roman", size=12, color="black")
        )
    
    html_path = os.path.join(out_dir, "heatmap.html")
    pio.write_html(fig, file=html_path, auto_open=False)
    print("Saved interactive HTML:", html_path)
    
    plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # path to folder containing results.jsonl
    path = r"C:\usp\tabular-transfer-learning\outputs\transfer-learning-from-upstream\ic_upstream3"
    jsonl = os.path.join(path, "results.jsonl")

    df = load_results(jsonl, prefer="test")
    print("Loaded rows:", len(df))
    # Quick check
    counts = df.groupby(['sample','strategy','imputation']).size().reset_index(name='n_seeds')
    print("\nCounts per config (sample, strategy, imputation):\n", counts.to_string(index=False))

    rank_df = build_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)
    print("\nRank table (first rows):\n", rank_df.head())
    plot_and_save_heatmap(rank_df, out_dir=path)
