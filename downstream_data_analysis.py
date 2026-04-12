import os
import re
import json
from turtle import color
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statistics import mode, StatisticsError
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.io as pio
import pickle
import re
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np
from result_analysis import anova_analysis, analysis_gain_by_transfer_learning, distribution_of_score_by_transfer_learning

# Set global style (Times New Roman)
plt.rcParams["font.family"] = "Times New Roman"
sns.set_style("whitegrid")

NOT_USED = "Not Used"
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
    if 'real' in name.lower():
        return 'RealValue'
    if 'pseudo' in name.lower():
        return 'PseudoFeature'
    return m.group(1) if m else NOT_USED

def parse_upstream_from_model_path_name(name: str):
    # pega o trecho que começa com mlp- e termina antes de /model_best.pth
    if not name:
        return None
    match = re.search(r'mlp-(.+?)/model_best\.pth$', name)
    return match.group(1) if match else None

def extract_upstream(config_name):
    """
    Extrai o número de 'upstream' de uma string de configuração.
    Retorna None se não houver upstream na string. (Sem TL)
    """
    match = re.search(r'upstream(\d+)', config_name)
    return (match.group(1)) if match else None


def map_strategy_from_runid(run_id):
    # adapt mapping to cover variants seen in your run_id
    _run_id = run_id.lower()
    if "fromscratch" in _run_id or "from_scratch" in _run_id or "_from_scratch_" in _run_id:
        return "FS"
    # patterns: mlpHeadFalse_freezeFalse  -> LH-E2E
    if "mlpheadfalse_freezefalse" in _run_id:
        return "LH-E2E"
    if "mlpheadtrue_freezefalse" in _run_id:
        return "MLP-E2E"
    if "mlpheadfalse_freezetrue" in _run_id:
        return "LH"
    if "mlpheadtrue_freezetrue" in _run_id:
        return "MLP"
    # try other guesses
    if "mlphead" in _run_id and "freezetrue" in _run_id:
        return "MLP"
    if "mlphead" in _run_id and "freezefalse" in _run_id:
        return "MLP-E2E"
    return "UNKNOWN"

def extract_seed_from_runid(run_id):
    match = re.search(r'seed(\d+)', run_id)
    return int(match.group(1)) if match else None

def extract_hyp_params_source_from_runid(run_id, upstream):
    if 'fromScratch' not in run_id:
        return upstream
    
    match = re.search(r'hypParamsFrom-(.+?)_fromScratch', run_id)
    return match.group(1) if match else None
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
            rmse_val   = stats.get("val_stats", {}).get("rmse")
            all_train_stats = stats.get("all_train_stats", [])

            # If RMSE still None, skip
            if rmse is None:
                continue
            upstream = extract_upstream(cfg.get("run_id", ""))
            rows.append({
                "sample": sample,
                "strategy": strategy,
                "imputation": imputation,
                "upstream": upstream,
                "rmse": float(rmse),
                "rmse_train": float(rmse_train) if rmse_train is not None else None,
                "rmse_test": float(rmse_test) if rmse_test is not None else None,
                "run_id": run_id,
                "all_train_stats": all_train_stats, 
                "seed": extract_seed_from_runid(run_id),
                "hyp_source": extract_hyp_params_source_from_runid(run_id, upstream),
            })
    df = pd.DataFrame(rows)
    # drop rows that failed to parse sample or strategy (optional)
    df = df.dropna(subset=["sample"])
    df['sample'] = df['sample'].astype(int)
    return df

# ---------------------------
# Ranking function for one sample
# ---------------------------
def compute_ranks_for_sample(df_sample, alpha=0.05, min_seeds=2, group_field=None, verbose=False):
    """
    df_sample: subset of df with a single sample value.
    group_field: optional column name to subdivide rankings (e.g., 'upstream'). 
                 If None, behaves like the original function.
    Returns dict: {(strategy, imputation): mean_rank_across_groups} if group_field provided,
                  else {(strategy, imputation): rank}.
    """
    # função auxiliar para calcular o ranking de um subconjunto
    def _rank_subset(sub_df):
        configs_df = sub_df[['strategy', 'imputation']].drop_duplicates().reset_index(drop=True)
        keys = [tuple(x) for x in configs_df.values.tolist()]

        counts = {
            k: sub_df[(sub_df['strategy'] == k[0]) & (sub_df['imputation'] == k[1])].shape[0]
            for k in keys
        }

        if any(v < min_seeds for v in counts.values()):
            means = {
                k: sub_df[(sub_df['strategy'] == k[0]) & (sub_df['imputation'] == k[1])]['rmse'].mean()
                for k in keys
            }
            sorted_keys = sorted(keys, key=lambda k: means[k])
            ranks, current_rank, prev = {}, 1, None
            for k in sorted_keys:
                if prev is None or not np.isclose(means[k], prev):
                    ranks[k] = current_rank
                    prev = means[k]
                    current_rank += 1
                else:
                    ranks[k] = current_rank - 1
            return ranks

        ranks, assigned, current_rank = {}, set(), 1
        while len(assigned) < len(keys):
            unassigned = [k for k in keys if k not in assigned]
            best = min(
                unassigned,
                key=lambda k: sub_df[(sub_df['strategy'] == k[0]) & (sub_df['imputation'] == k[1])]['rmse'].mean(),
            )

            same_rank = []
            for k in unassigned:
                a = sub_df[(sub_df['strategy'] == best[0]) & (sub_df['imputation'] == best[1])]['rmse'].values
                b = sub_df[(sub_df['strategy'] == k[0]) & (sub_df['imputation'] == k[1])]['rmse'].values
                try:
                    _, p = mannwhitneyu(a, b, alternative="less")
                except Exception:
                    p = 1.0
                if p >= alpha:
                    same_rank.append(k)

            for k in same_rank:
                ranks[k] = current_rank
                assigned.add(k)
            current_rank += 1
        return ranks

    # sem campo adicional → ranking direto
    if not group_field:
        return _rank_subset(df_sample)

    # com campo adicional → calcula ranking dentro de cada grupo e tira média dos ranks
    control_rows = df_sample[df_sample[group_field].isna()]
    grouped_ranks = []
    for val, sub_df in df_sample.groupby(group_field):
        # adiciona as linhas None ao grupo atual
        sub_with_control = pd.concat([sub_df, control_rows], ignore_index=True)
        sub_ranks = _rank_subset(sub_with_control)
        grouped_ranks.append(sub_ranks)

    # média dos ranks para cada config
    all_keys = set(k for ranks in grouped_ranks for k in ranks.keys())
    mean_ranks = {
        k: np.mean([ranks.get(k, np.nan) for ranks in grouped_ranks if k in ranks])
        for k in all_keys
    }
    return mean_ranks


# ---------------------------
# Build heatmap-ready DataFrame
# ---------------------------
def build_rank_table(df, alpha=0.05, min_seeds=2, group_field=None, verbose=False):
    """
    Constrói um DataFrame com os ranks médios por (strategy, imputation)
    e opcionalmente agrupados por um campo adicional.
    """
    results = []
    for sample, dfg in df.groupby('sample'):
        ranks = compute_ranks_for_sample(
            dfg, alpha=alpha, min_seeds=min_seeds, group_field=group_field, verbose=verbose
        )
        for (strategy, imputation), rank in ranks.items():
            results.append({
                "sample": sample,
                "strategy": strategy,
                "imputation": imputation,
                "rank": rank,
            })
    return pd.DataFrame(results)

def sum_strategies(imputations_order: List[str]) -> int:
    strategies = []
    for imp in imputations_order:
        strategies.append(strategies_order_per_imputation(imp))
    return len(strategies)

def plot_and_save_heatmap(
        rank_df,
        out_dir,
        name= "",
        strategies_order=None,
        imputations_order=None):
    os.makedirs(out_dir, exist_ok=True)
    
    
    if imputations_order is None:
        imputations_order = sorted(rank_df['imputation'].unique())
    
    n_imputations = len(imputations_order)

    # Determine global min/max rank for consistent color scale
    vmin = rank_df["rank"].min()
    vmax = rank_df["rank"].max()

    # Proportional width for each imputation subplot
    n_imputations = len(imputations_order)

    width_ratios = [
        len(strategies_order_per_imputation(imp))
        for imp in imputations_order
    ]
    cbar_compensation = 0.9  # ajuste fino (0.4–0.8 costuma funcionar bem)
    width_ratios[-1] += cbar_compensation

    fig, axes = plt.subplots(
        1, n_imputations,
        figsize=(2 * sum_strategies(imputations_order), 4),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios}
    )
    fig.subplots_adjust(right=0.92)

    if n_imputations == 1:
        axes = [axes]

    for ax, imp in zip(axes, imputations_order):
        strategies_order = strategies_order_per_imputation(imp)
        df_imp = rank_df[rank_df['imputation'] == imp].pivot(
            index="sample", columns="strategy", values="rank"
        )
        df_imp = df_imp[strategies_order]

        sns.heatmap(
            df_imp,
            ax=ax,
            annot=True,
            fmt="0.2f",
            cmap="RdYlBu_r",
            vmin=vmin, vmax=vmax,   # keep same color scale
            cbar=ax == axes[-1],
            cbar_kws={"label": "Rank", "shrink": 0.75, "anchor": (0.0, 0.5)},
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
    path_complete = os.path.join(out_dir, f"heatmap-{name}")
    plt.savefig((path_complete + ".png"), dpi=300, bbox_inches="tight")
    #plt.savefig((path_complete + ".pdf"), bbox_inches="tight")
    #plt.savefig((path_complete + ".svg"), bbox_inches="tight")

    with open((path_complete + ".fig.pickle"), "wb") as f:
        pickle.dump(fig, f)
    #plt.savefig("heatmap.eps", format="eps", bbox_inches="tight")

    plt.close()

def strategies_order_per_imputation(imputation):
    if imputation == NOT_USED:
        return ['FS']
    else:
        return ['FS','LH-E2E', 'MLP-E2E', 'LH', 'MLP']
    
def plot_BoxPlots_overfitting(df, out_dir, strategies_order=None, imputations_order=None):
    """
    Gera boxplots do overfitting gap (test_rmse - train_rmse) para cada estratégia e imputação.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "rmse_train" not in df.columns or "rmse_test" not in df.columns:
        raise ValueError("O DataFrame precisa conter as colunas rmse_train e rmse_test.")

    df = df.copy()
    df["overfit_gap"] = df["rmse_test"] - df["rmse_train"]

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
        strategies_order = strategies_order_per_imputation(imp)

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
    ##plt.savefig(path_complete + ".pdf", bbox_inches="tight")
    ###plt.savefig(path_complete + ".svg", bbox_inches="tight")
    with open(path_complete + ".fig.pickle", "wb") as f:
        pickle.dump(fig, f)

    plt.close()

def summarize_results(df: pd.DataFrame, out_dir: str, group_cols=None, filename="summary.csv"):
    """
    Gera estatísticas agregadas por configuração.

    Args:
        df (pd.DataFrame): DataFrame com colunas ["sample", "strategy", "imputation", "rmse", ...]
        out_dir (str): Caminho onde salvar o CSV.
        group_cols (list[str], opcional): Colunas para agrupar. Default = ["sample", "strategy", "imputation"].
        filename (str): Nome do CSV de saída.
    """
    if group_cols is None:
        group_cols = ["sample", "strategy", "imputation"]

    summaries = []

    grouped = df.groupby(group_cols)
    for keys, group in grouped:
        rmses = group["rmse"].tolist()
        try:
            rmse_mode = mode(rmses)
        except StatisticsError:
            rmse_mode = None  # se não houver moda definida

        summaries.append({
            **dict(zip(group_cols, keys)),
            "best": min(rmses),
            "worst": max(rmses),
            "mean": group["rmse"].mean(),
            "std": group["rmse"].std(),
            "mode": rmse_mode,
            "median": group["rmse"].median(),
            "count": len(rmses),
        })

    summary_df = pd.DataFrame(summaries)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    summary_df.to_csv(out_path, index=False)

    return summary_df

def analyze_training_curves(df: pd.DataFrame, out_dir: str):
    # paleta de cores por upstream
    upstreams = df["upstream"].unique()
    palette = dict(zip(upstreams, sns.color_palette("tab20", len(upstreams))))

    # armazenar médias para o gráfico comparativo
    mean_curves = []

    # função auxiliar: preencher épocas pós-stopping com o último valor
    def pad_with_last_value(values, target_len):
        if len(values) < target_len:
            return np.concatenate([values, np.full(target_len - len(values), values[-1])])
        return np.array(values[:target_len])

    for (strategy, imputation), group in df.groupby(["strategy", "imputation"]):
        plt.figure(figsize=(8, 5))

        # plot curvas individuais
        unique_upstreams = df["upstream"].unique()
        palette = dict(zip(unique_upstreams, sns.color_palette("hls", len(unique_upstreams))))

        for _, row in group.iterrows():
            if not row["all_train_stats"]:
                continue
            epochs = [e["epoch"] for e in row["all_train_stats"]]
            rmses = [e["train_stats"]["rmse"] for e in row["all_train_stats"]]
            color = palette[row["upstream"]]
            plt.plot(epochs, rmses, alpha=0.3, lw=1, color=color, label=row["upstream"])

        # remover labels duplicados da legenda
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Upstreams")

        plt.title(f"Curvas de Aprendizado — {strategy} / {imputation}")
        plt.xlabel("Época")
        plt.ylabel("RMSE de Treino")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_dir_complete = os.path.join(out_dir, "training_curves")
        os.makedirs(out_dir_complete, exist_ok=True)
        path_complete = os.path.join(out_dir_complete, f"CurvaAprendizado {strategy} - {imputation}")
        plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
        ##plt.savefig(path_complete + ".pdf", bbox_inches="tight")
        ##plt.savefig(path_complete + ".svg", bbox_inches="tight")
        plt.close()

        # === curva média ===
        max_epochs = max(max([e["epoch"] for e in r["all_train_stats"]]) for _, r in group.iterrows())
        all_rmse = np.full((len(group), max_epochs + 1), np.nan)

        for i, (_, row) in enumerate(group.iterrows()):
            epochs = [e["epoch"] for e in row["all_train_stats"]]
            rmses = [e["train_stats"]["rmse"] for e in row["all_train_stats"]]
            rmses_padded = pad_with_last_value(rmses, max_epochs + 1)
            all_rmse[i, :] = rmses_padded

        # calcular estatísticas considerando IC 95%
        mean_rmse = np.nanmean(all_rmse, axis=0)
        median_rmse = np.nanmedian(all_rmse, axis=0)
        sem = stats.sem(all_rmse, axis=0, nan_policy="omit")
        ci95 = 1.96 * sem  # intervalo de confiança de 95%
        std_rmse = np.nanstd(all_rmse, axis=0)

        mean_curves.append({
            "strategy": strategy,
            "imputation": imputation,
            "mean_rmse": mean_rmse,
            "median_rmse": median_rmse,
            "std_rmse": std_rmse
        })

        # === detecção de plateau ===
        diffs = np.abs(np.gradient(mean_rmse))
        plateau_epoch = np.argmax(diffs < 1e-4)  # época onde o gradiente da perda "achata"
        plt.figure(figsize=(8, 5))
        plt.plot(mean_rmse, label="RMSE médio")
        plt.plot(median_rmse, ls="--", color="gray", lw=1.5, label="Mediana (tracejada)")
        plt.fill_between(range(len(mean_rmse)),
                         mean_rmse - ci95,
                         mean_rmse + ci95,
                         alpha=0.2,
                         color="blue",
                         label="IC 95%")
        plt.axvline(plateau_epoch, color="red", ls="--", label=f"Plateau ~ Época {plateau_epoch}")
        plt.title(f"Análise de Plateau — {strategy} - {imputation}")
        plt.xlabel("Época")
        plt.ylabel("RMSE")
        plt.legend(title="Legenda")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_dir_complete = os.path.join(out_dir, "plateau")
        os.makedirs(out_dir_complete, exist_ok=True)
        path_complete = os.path.join(out_dir_complete, f"Análise de Plateau- {strategy} - {imputation}")
        plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
        ##plt.savefig(path_complete + ".pdf", bbox_inches="tight")
        ##plt.savefig(path_complete + ".svg", bbox_inches="tight")
        plt.close()

        print(f"→ {strategy}/{imputation}: plateau detectado próximo da época {plateau_epoch}, "
              f"RMSE médio final = {mean_rmse[-1]:.4f}")

    # === gráfico comparativo final com as curvas médias de todas as combinações ===
    plot_configs = [
        {"max_epochs": None, "suffix": "", "title_suffix": ""},
        {"max_epochs": 100, "suffix": "_EpocasIniciais100", "title_suffix": " (100 Épocas Iniciais)"},
        {"max_epochs": 20, "suffix": "_EpocasIniciais20", "title_suffix": " (20 Épocas Iniciais)"},
        {"max_epochs": 10, "suffix": "_EpocasIniciais10", "title_suffix": " (10 Épocas Iniciais)"}
    ]
    cmap = plt.cm.get_cmap("tab20", len(mean_curves))
    for cfg in plot_configs:

        plt.figure(figsize=(10, 6))
        legend_lines = []

        for i, c in enumerate(mean_curves):
            label = f"{c['strategy']} / {c['imputation']}"
            if cfg["max_epochs"] is None:
                x = range(len(c["mean_rmse"]))
                mean = c["mean_rmse"]
                median = c["median_rmse"]
                std = c["std_rmse"]
            else:
                x = range(cfg["max_epochs"])
                mean = c["mean_rmse"][:cfg["max_epochs"]]
                median = c["median_rmse"][:cfg["max_epochs"]]
                std = c["std_rmse"][:cfg["max_epochs"]]

            color = cmap(i)
            plt.plot(x, mean, lw=2, label=label, color=color)
            legend_lines.append((color, label))


            plt.plot(x, median, linestyle="--", color=color, lw=1.5)

            plt.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15
            )

        # === título e legendas ===
        plt.title("Convergência Média por Estratégia e Imputação" + cfg["title_suffix"])
        plt.xlabel("Época")
        plt.ylabel("RMSE médio de Treino")
        plt.grid(True, alpha=0.3)

        # === legenda composta ===
        # Apenas uma vez, explicamos o que significam as texturas

        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Linha contínua: Média das execuções'),
            Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Linha tracejada: Mediana das execuções'),
            Patch(facecolor='gray', alpha=0.15, label='Área sombreada: ±1 desvio padrão'),
            Line2D([], [], color='none', label='──────────────────────────────'),
            Line2D([0], [0], color='none', label='Cores: representam cada combinação'),
        ]

        plt.legend(
            handles=legend_elements + [
                Line2D([0], [0], color=color, lw=2, label=label)
                for color, label in legend_lines
            ],
            title="Texturas e cores:",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        plt.tight_layout()

        # === salvar ===
        path_complete = os.path.join(out_dir, f"ConvergenciaMedia_Todas{cfg['suffix']}")
        plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
        plt.close()

def build_statistical_mean_rank_table(df, alpha=0.05, min_seeds=2, verbose=False):
    """
    Gera um DataFrame de rank médio por (upstream, strategy, imputation)
    com base em comparações estatísticas (Mann–Whitney U-test) entre estratégias,
    controlando para variações de sample.
    
    - ignora linhas com upstream=None (ex: FS)
    - usa teste de significância em cada sample antes de agregar ranks médios
    """
    df = df.copy()
    df = df[df["upstream"].notna()]  # ignorar FS

    results = []

    # loop por sample
    for sample, dfg_sample in df.groupby("sample"):
        # loop por imputação
        for imp, dfg_imp in dfg_sample.groupby("imputation"):
            # loop por upstream
            for up, dfg_up in dfg_imp.groupby("upstream"):
                # obtém todas as estratégias testadas
                strategies = dfg_up["strategy"].unique()

                # calcula médias de RMSE por strategy (caso haja repetições)
                mean_rmse = dfg_up.groupby("strategy")["rmse"].mean()

                # inicializa estrutura de ranking
                ranks = {s: None for s in strategies}
                assigned = set()
                current_rank = 1

                # ranking com Mann–Whitney
                while len(assigned) < len(strategies):
                    # entre os ainda não ranqueados, pegue o de menor média de RMSE
                    remaining = [s for s in strategies if s not in assigned]
                    best = min(remaining, key=lambda s: mean_rmse[s])

                    # encontra estratégias estatisticamente equivalentes
                    same_rank = []
                    for s in remaining:
                        a = dfg_up[dfg_up["strategy"] == best]["rmse"].values
                        b = dfg_up[dfg_up["strategy"] == s]["rmse"].values

                        if len(a) < min_seeds or len(b) < min_seeds:
                            continue

                        try:
                            _, p = mannwhitneyu(a, b, alternative="less")
                        except Exception:
                            p = 1.0

                        if p >= alpha:  # diferença não significativa
                            same_rank.append(s)

                    for s in same_rank:
                        ranks[s] = current_rank
                        assigned.add(s)

                    current_rank += 1

                # salva resultados desse sample / upstream / imputação
                for s, r in ranks.items():
                    results.append({
                        "sample": sample,
                        "upstream": up,
                        "imputation": imp,
                        "strategy": s,
                        "rank": r
                    })

    rank_df = pd.DataFrame(results)

    # agora média de ranks por upstream (controlando samples)
    mean_rank_df = (
        rank_df.groupby(["upstream", "strategy", "imputation"])["rank"]
        .mean()
        .reset_index()
    )

    if verbose:
        print(f"Gerado mean_rank_df com {len(mean_rank_df)} combinações únicas.")
    return mean_rank_df

def build_upstream_rank_by_strategy(df, alpha=0.05, min_seeds=2, verbose=False):
    """
    Calcula o rank relativo entre upstreams para cada estratégia,
    controlando o efeito de sample e imputação.
    """
    results = []
    for imp, df_imp in df.groupby("imputation"):
        for strat, df_strat in df_imp.groupby("strategy"):
            for sample, df_sample in df_strat.groupby("sample"):
                df_sample = df_sample[df_sample["upstream"].notna()]
                upstreams = df_sample["upstream"].unique()
                if len(upstreams) < 2:
                    continue

                mean_rmse = {
                    u: df_sample[df_sample["upstream"] == u]["rmse"].mean()
                    for u in upstreams
                }

                ranks, assigned, current_rank = {}, set(), 1
                while len(assigned) < len(upstreams):
                    unassigned = [u for u in upstreams if u not in assigned]
                    best = min(unassigned, key=lambda u: mean_rmse[u])
                    same_rank = []
                    for u in unassigned:
                        a = df_sample[df_sample["upstream"] == best]["rmse"].values
                        b = df_sample[df_sample["upstream"] == u]["rmse"].values
                        try:
                            _, p = mannwhitneyu(a, b, alternative="less")
                        except Exception:
                            p = 1.0
                        if p >= alpha:
                            same_rank.append(u)
                    for u in same_rank:
                        ranks[u] = current_rank
                        assigned.add(u)
                    current_rank += 1

                for u, r in ranks.items():
                    results.append({
                        "strategy": strat,
                        "imputation": imp,
                        "upstream": u,
                        "sample": sample,   # 🔥 ESSENCIAL
                        "rank": r,
                    })

    df_rank = pd.DataFrame(results)

    # média (como você já fazia)
    mean_rank_df = (
        df_rank.groupby(["strategy", "upstream", "imputation"])["rank"]
        .mean()
        .reset_index()
    )

    return mean_rank_df, df_rank
def build_upstream_rank_by_imputation(df, alpha=0.05, min_seeds=2, verbose=False):
    """
    Calcula o rank relativo entre upstreams para cada imputação,
    controlando o efeito de sample e estratégia.
    """
    results = []
    for imp, df_imp in df.groupby("imputation"):
        for strat, df_strat in df_imp.groupby("strategy"):
            for sample, df_sample in df_strat.groupby("sample"):
                df_sample = df_sample[df_sample["upstream"].notna()]
                upstreams = df_sample["upstream"].unique()
                if len(upstreams) < 2:
                    continue

                mean_rmse = {
                    u: df_sample[df_sample["upstream"] == u]["rmse"].mean()
                    for u in upstreams
                }

                ranks, assigned, current_rank = {}, set(), 1
                while len(assigned) < len(upstreams):
                    unassigned = [u for u in upstreams if u not in assigned]
                    best = min(unassigned, key=lambda u: mean_rmse[u])
                    same_rank = []
                    for u in unassigned:
                        a = df_sample[df_sample["upstream"] == best]["rmse"].values
                        b = df_sample[df_sample["upstream"] == u]["rmse"].values
                        try:
                            _, p = mannwhitneyu(a, b, alternative="less")
                        except Exception:
                            p = 1.0
                        if p >= alpha:
                            same_rank.append(u)
                    for u in same_rank:
                        ranks[u] = current_rank
                        assigned.add(u)
                    current_rank += 1

                for u, r in ranks.items():
                    results.append({
                        "imputation": imp,
                        "upstream": u,
                        "strategy": strat,
                        "sample": sample,
                        "rank": r,
                    })

    df_rank = pd.DataFrame(results)

    # média global colapsando estratégias e samples
    mean_rank_df = (
        df_rank.groupby(["imputation", "upstream"])["rank"]
        .mean()
        .reset_index()
    )
    return mean_rank_df

def plot_heatmap_mean_rank_imputation_upstream(mean_rank_df, out_dir):
    """
    Plota um único heatmap com o rank médio global dos upstreams por imputação.
    """
    pivot = mean_rank_df.pivot(
        index="imputation",
        columns="upstream",
        values="rank"
    )

    plt.figure(figsize=(11, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr_r",
        cbar_kws={"label": "Rank Médio (Upstream)"},
    )
    plt.title("Ranking Médio Global de Upstreams por Imputação")
    plt.xlabel("Upstream")
    plt.ylabel("Imputação")
    plt.tight_layout()

    path_complete = os.path.join(out_dir, "heatmap_mean_rank_imputation_upstream.png")
    plt.savefig(path_complete, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# 2️ Correlação de Spearman entre upstreams (baseada nos ranks médios)
# =========================================================
def compute_spearman_corr_between_upstreams(mean_rank_df, out_dir="."):
    """
    Calcula a correlação de Spearman entre upstreams
    com base nos ranks médios por (strategy, imputação).
    """
    # Pivotar: linhas = upstreams, colunas = strategy + imputação
    pivot = mean_rank_df.pivot_table(
        index="upstream", columns=["strategy", "imputation"], values="rank"
    )

    # Matriz de correlação de Spearman
    corr_matrix = pivot.T.corr(method="spearman")

    # Visualizar heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"label": "Spearman Correlation"},
    )
    plt.title("Correlação de Spearman entre Upstreams (Rank Médio)")
    plt.tight_layout()
    path_complete = os.path.join(out_dir, "compute_spearman_corr_between_upstreams")
    plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
    plt.close()

    return corr_matrix

# =========================================================
# 1️⃣ Heatmap de rank médio (Upstream × Strategy)
#     → Analisa como o UPSTREAM afeta o rank das ESTRATÉGIAS
# =========================================================
def plot_heatmap_mean_rank_upstream_strategy(mean_rank_df, out_dir):
    """
    Plota um heatmap de ranks médios (Upstream × Strategy) para cada imputação.
    Mostra como diferentes upstreams impactam o desempenho relativo das estratégias.
    """
    imputations = mean_rank_df["imputation"].unique()
    for imp in imputations:
        subset = mean_rank_df[mean_rank_df["imputation"] == imp]
        pivot = subset.pivot(index="upstream", columns="strategy", values="rank")

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu_r",
            cbar_kws={"label": "Rank Médio (menor = melhor)"},
        )
        plt.title(f"Influência do Upstream no Rank das Estratégias (Imputação: {imp})")
        plt.xlabel("Strategy")
        plt.ylabel("Upstream")
        plt.tight_layout()
        path_complete = os.path.join(out_dir, f"heatmap_mean_rank_upstream_strategy_imp{imp}")
        plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_heatmap_mean_rank_strategy_upstream(mean_rank_df, out_dir):
    """
    Plota o rank médio dos upstreams para cada estratégia.
    """
    out_dir = os.path.join(out_dir, "heatmap_mean_rank_strategy_upstream")
    _ensure_outdir(out_dir)
    imputations = mean_rank_df["imputation"].unique()
    for imp in imputations:
        subset = mean_rank_df[mean_rank_df["imputation"] == imp]
        pivot = subset.pivot(index="strategy", columns="upstream", values="rank")

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrBr_r",
            cbar_kws={"label": "Rank Médio (Upstream)"},
        )
        plt.title(f"Ranking Médio de Upstreams por Estratégia (Imputação: {imp})")
        plt.xlabel("Upstream")
        plt.ylabel("Estratégia")
        plt.tight_layout()
        path_complete = os.path.join(out_dir, "imp" + str(imp))
        plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
        plt.close()


def build_friedman_matrix(df_rank):
    """
    Constrói matriz para teste de Friedman.

    Linhas = blocos experimentais (sample, strategy, imputation)
    Colunas = upstreams
    Valores = rank
    """
    df_rank = df_rank.copy()

    df_rank["block"] = (
        df_rank["sample"].astype(str)
        + "_" + df_rank["strategy"].astype(str)
        + "_" + df_rank["imputation"].astype(str)
    )

    pivot = df_rank.pivot_table(
        index="block",
        columns="upstream",
        values="rank"
    )

    # remove blocos incompletos (necessário para Friedman)
    pivot = pivot.dropna()

    return pivot

def run_friedman_test(pivot):
    """
    Executa o teste de Friedman.
    """
    data = [pivot[col].values for col in pivot.columns]

    stat, p = friedmanchisquare(*data)

    print("\n[Friedman Test]")
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {p:.6f}")

    return stat, p

def run_nemenyi_posthoc(pivot, out_dir):
    """
    Executa teste post-hoc de Nemenyi.
    """
    nemenyi = sp.posthoc_nemenyi_friedman(pivot)

    print("\n[Nemenyi Post-hoc]")
    print(nemenyi)

    # salvar heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        nemenyi,
        annot=True,
        cmap="coolwarm_r",
        cbar_kws={"label": "p-value"}
    )
    plt.title("Nemenyi Post-hoc (p-values)")
    plt.tight_layout()

    path = os.path.join(out_dir, "nemenyi_posthoc.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return nemenyi

def compute_global_mean_rank(df_rank):
    """
    Calcula rank médio global dos upstreams.
    """
    global_rank = (
        df_rank.groupby("upstream")["rank"]
        .mean()
        .sort_values()
        .reset_index()
    )

    print("\n[Global Mean Rank]")
    print(global_rank)

    return global_rank

def plot_global_mean_rank(global_rank, out_dir):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(8, 5))
    plt.bar(global_rank["upstream"].astype(str), global_rank["rank"])
    plt.xlabel("Upstream")
    plt.ylabel("Mean Rank (lower = better)")
    plt.title("Global Ranking of Upstreams")
    plt.tight_layout()

    path = os.path.join(out_dir, "global_mean_rank.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cd_diagram(pivot, out_dir):
    """
    Plota CD diagram simplificado baseado em ranks médios.
    """
    import matplotlib.pyplot as plt
    import os

    mean_ranks = pivot.mean().sort_values()

    plt.figure(figsize=(10, 2))
    plt.scatter(mean_ranks.values, [1]*len(mean_ranks))

    for i, (name, rank) in enumerate(mean_ranks.items()):
        plt.text(rank, 1.02, str(name), ha='center')

    plt.yticks([])
    plt.xlabel("Mean Rank (lower = better)")
    plt.title("CD Diagram (simplified)")

    path = os.path.join(out_dir, "cd_diagram.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def analyze_upstreams_statistically(df_rank, out_dir):
    """
    Pipeline completo:
    - Friedman
    - Nemenyi
    - Ranking global
    - CD diagram
    """
    pivot = build_friedman_matrix(df_rank)

    run_friedman_test(pivot)

    nemenyi = run_nemenyi_posthoc(pivot, out_dir)

    global_rank = compute_global_mean_rank(df_rank)

    plot_global_mean_rank(global_rank, out_dir)

    plot_cd_diagram(pivot, out_dir)

    return {
        "pivot": pivot,
        "nemenyi": nemenyi,
        "global_rank": global_rank
    }
def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def get_jsonl_files(base_path: str):
    """
    Retorna a lista de arquivos .jsonl dentro de subpastas do base_path.
    """
    jsonl_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def load_all_results(base_path: str) -> pd.DataFrame:

    files = get_jsonl_files(base_path)
    for file in files:
        df_part = load_results(file, prefer="test")
        if 'df' not in locals():
            df = df_part
        else:
            df = pd.concat([df, df_part], ignore_index=True)

    return df

if __name__ == "__main__":
    # path to folder containing results.jsonl
    ##for experiment in ["all_experiments"]: #, "ic_upstream2", "ic_upstream3", "ic_upstream4"]:

    experiment= "all_experiments"
    path = fr"C:\usp\tabular-transfer-learning\outputs\transfer-learning-from-upstream\{experiment}"

    df = load_all_results(path)

    path = os.path.join(path, "analysis")
    rank_df = build_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)
    #distribution_of_score_by_transfer_learning(df, path)
    #plot_and_save_heatmap(rank_df, name="geral", out_dir=path)
    #plot_BoxPlots_overfitting(df, out_dir=path)
    #summarize_results(dfF, out_dir=path)
    #analyze_training_curves(df, out_dir=path)

    has_multiple_upstreams = True
    if False:
        print("Múltiplos upstreams detectados — executando análises adicionais...")

        rank_mean_df = build_rank_table(df, alpha=0.05, min_seeds=2, group_field="upstream", verbose=False)
        plot_and_save_heatmap(rank_mean_df, name="média-por-upstream", out_dir=path)

        #rank médio por (upstream, strategy, imputation)
        mean_rank_df = build_statistical_mean_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)
        compute_spearman_corr_between_upstreams(mean_rank_df, out_dir=path)
        plot_heatmap_mean_rank_upstream_strategy(mean_rank_df, out_dir=path)

        #rank relativo entre upstreams para cada imputação
        mean_rank_imp_up = build_upstream_rank_by_imputation(df)
        plot_heatmap_mean_rank_imputation_upstream(mean_rank_imp_up, out_dir=path)

        #anova_analysis(df, out_dir=path)

        mean_rank_upstream_df, df_rank = build_upstream_rank_by_strategy(df, alpha=0.05, min_seeds=2)
        re = analyze_upstreams_statistically(df_rank, out_dir=path)
        plot_heatmap_mean_rank_strategy_upstream(mean_rank_upstream_df, out_dir=path)
    
    analysis_gain_by_transfer_learning(df, compare_with_imputed_fs=True, out_path=path)
    analysis_gain_by_transfer_learning(df, compare_with_imputed_fs=False, out_path=path)

