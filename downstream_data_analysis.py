import os
import re
import json
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
import seaborn as sns
from typing import Tuple

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
            rmse_val   = stats.get("val_stats", {}).get("rmse")
            all_train_stats = stats.get("all_train_stats", [])

            # If RMSE still None, skip
            if rmse is None:
                continue

            rows.append({
                "sample": sample,
                "strategy": strategy,
                "imputation": imputation,
                "upstream": extract_upstream(cfg.get("run_id", "")),
                "rmse": float(rmse),
                "rmse_train": float(rmse_train) if rmse_train is not None else None,
                "rmse_test": float(rmse_test) if rmse_test is not None else None,
                "run_id": run_id,
                "all_train_stats": all_train_stats
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
    plt.savefig("heatmap.eps", format="eps", bbox_inches="tight")

    plt.close()

def strategies_order_per_imputation(imputation):
    if imputation == NOT_USED:
        return ['FS']
    else:
        return ['LH-E2E', 'MLP-E2E', 'LH', 'MLP']
    
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

    ##plt.show()

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
    palette = dict(zip(upstreams, sns.color_palette("tab10", len(upstreams))))

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
        palette = dict(zip(unique_upstreams, sns.color_palette("husl", len(unique_upstreams))))

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
        ##plt.show()

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
        ##plt.show()

        print(f"→ {strategy}/{imputation}: plateau detectado próximo da época {plateau_epoch}, "
              f"RMSE médio final = {mean_rmse[-1]:.4f}")

    # === gráfico comparativo final com as curvas médias de todas as combinações ===
    plot_configs = [
        {"max_epochs": None, "suffix": "", "title_suffix": ""},
        {"max_epochs": 100, "suffix": "_EpocasIniciais100", "title_suffix": " (100 Épocas Iniciais)"},
        {"max_epochs": 20, "suffix": "_EpocasIniciais20", "title_suffix": " (20 Épocas Iniciais)"},
        {"max_epochs": 10, "suffix": "_EpocasIniciais10", "title_suffix": " (10 Épocas Iniciais)"}
    ]

    for cfg in plot_configs:

        plt.figure(figsize=(10, 6))
        legend_lines = []

        for c in mean_curves:
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

            line_mean, = plt.plot(x, mean, lw=2, label=label)
            color = line_mean.get_color()
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
        #plt.show()

# ANOVA 2-way (upstream x strategy) dentro de cada imputação

def _run_two_way_anova(
    df,
    dv,
    factor1,
    factor2,
    min_rows=10,
):
    """
    Executa ANOVA 2-way (tipo II) e retorna a tabela com eta² e partial eta².
    """
    sub = (df.dropna(subset=[dv, factor1, factor2])).copy()

    if sub.shape[0] < min_rows:
        return None

    sub[factor1] = sub[factor1].astype("category")    
    sub[factor2] = sub[factor2].astype("category")


    formula = f"{dv} ~ C({factor1}) * C({factor2})"
    model = smf.ols(formula, data=sub).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    # Effect sizes
    ss_total = aov["sum_sq"].sum()
    aov["eta2"] = aov["sum_sq"] / ss_total

    ss_error = (
        aov.loc["Residual", "sum_sq"]
        if "Residual" in aov.index
        else model.ssr
    )

    aov["partial_eta2"] = np.nan
    for idx in aov.index:
        if idx != "Residual":
            ss_effect = aov.loc[idx, "sum_sq"]
            aov.loc[idx, "partial_eta2"] = ss_effect / (ss_effect + ss_error)

    return aov

def anova_upstream_strategy_all(
    df,
    dv="rmse",
    out_dir=None,
    min_rows=10,
):
    aov = _run_two_way_anova(
        df=df,
        dv=dv,
        factor1="upstream",
        factor2="strategy",
        min_rows=min_rows,
    )

    if aov is None:
        return None

    if out_dir:
        aov.to_csv(
            os.path.join(out_dir, f"anova_{dv}_2way_upstream_strategy_all.csv")
        )

    return aov

def anova_strategy_imputation_all(
    df,
    dv="rmse",
    out_dir=None,
    min_rows=10,
):
    aov = _run_two_way_anova(
        df=df,
        dv=dv,
        factor1="strategy",
        factor2="imputation",
        min_rows=min_rows,
    )

    if aov is None:
        return None

    if out_dir:
        aov.to_csv(
            os.path.join(out_dir, f"anova_{dv}_2way_strategy_imputation_all.csv")
        )

    return aov
def anova_upstream_imputation_all(
    df,
    dv="rmse",
    out_dir=None,
    min_rows=10,
):
    aov = _run_two_way_anova(
        df=df,
        dv=dv,
        factor1="upstream",
        factor2="imputation",
        min_rows=min_rows,
    )

    if aov is None:
        return None

    if out_dir:
        aov.to_csv(
            os.path.join(out_dir, f"anova_{dv}_2way_upstream_imputation_all.csv")
        )

    return aov

def anova_two_way_by_imputation(
    df,
    dv="rmse",
    factor1="upstream",
    factor2="strategy",
    min_rows=10,
    out_dir=None,
):
    results = {}

    for imp in sorted(df["imputation"].unique()):
        sub = df[df["imputation"] == imp]

        aov = _run_two_way_anova(
            df=sub,
            dv=dv,
            factor1=factor1,
            factor2=factor2,
            min_rows=min_rows,
        )

        if aov is None:
            continue

        results[imp] = aov

        if out_dir:
            aov.to_csv(
                os.path.join(
                    out_dir,
                    f"anova_{dv}_2way_imputation_{imp}.csv",
                )
            )

    return results

def anova_analysis(df, out_dir, dv="rmse"):
    """
    Executa todas as análises de ANOVA e salva os resultados em:
    out_dir/anova_analysis/
    """
    out_dir = os.path.join(out_dir, "anova_analysis")
    _ensure_outdir(out_dir)

    filtered_df = df[df["upstream"].notna()] ##remove situações sem transferências

    anova_upstream_strategy_all(
        df=filtered_df,
        dv=dv,
        out_dir=out_dir,
    )

    anova_upstream_imputation_all(
        df=filtered_df,
        dv=dv,
        out_dir=out_dir,
    )

    anova_two_way_by_imputation(
        df=filtered_df,
        dv=dv,
        out_dir=out_dir,
    )

    anova_strategy_imputation_all(
        df=filtered_df,
        dv=dv,
        out_dir=out_dir,
    )


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
                        "sample": sample,
                        "rank": r,
                    })
    df_rank = pd.DataFrame(results)
    # rank médio global (média dos samples)
    mean_rank_df = (
        df_rank.groupby(["strategy", "upstream", "imputation"])["rank"]
        .mean()
        .reset_index()
    )
    return mean_rank_df

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
    #plt.show()

    return corr_matrix

import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        path_complete = os.path.join(out_dir, "heatmap_mean_rank_strategy_upstream")
        plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")

def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def compute_best_st_per_sample(df: pd.DataFrame, rmse_col: str = "rmse_test") -> pd.DataFrame:
    """
    Returns DataFrame with columns: sample, best_st_rmse
    Considers ST rows as upstream == None.
    Uses rmse_col (default rmse_test). Drops NaNs.
    """
    st_df = df[df["upstream"].isna()].dropna(subset=[rmse_col])
    if st_df.empty:
        return pd.DataFrame(columns=["sample", "best_st_rmse"])
    best_st = st_df.groupby("sample")[rmse_col].min().reset_index().rename(columns={rmse_col: "best_st_rmse"})
    return best_st

def compute_best_tl_per_group(df: pd.DataFrame, rmse_col: str = "rmse_test") -> pd.DataFrame:
    """
    Computes best TL (min rmse_col) for each (sample, strategy, imputation, upstream).
    Returns DataFrame with those keys + best_tl_rmse.
    Excludes ST (upstream is None).
    """
    tl = df[df["upstream"].notna()].dropna(subset=[rmse_col])
    if tl.empty:
        return pd.DataFrame(columns=["sample", "strategy", "imputation", "upstream", "best_tl_rmse"])
    best_tl = (
        tl.groupby(["sample", "strategy", "imputation", "upstream"])[rmse_col]
        .min()
        .reset_index()
        .rename(columns={rmse_col: "best_tl_rmse"})
    )
    return best_tl

def build_tl_gain_per_run_df(
    df: pd.DataFrame,
    rmse_col: str = "rmse_test"
) -> pd.DataFrame:
    """
    Returns a DataFrame with gains computed PER EXECUTION.

    Columns:
    sample, strategy, imputation, upstream, rmse_test,
    best_st_rmse, abs_gain, pct_gain

    - Keeps all TL executions (upstream not None)
    - Uses best ST (min rmse) per sample as baseline
    """
    # best ST per sample (single value per sample)
    best_st = compute_best_st_per_sample(df, rmse_col=rmse_col)

    if best_st.empty:
        return pd.DataFrame(columns=[
            "sample","strategy","imputation","upstream",
            rmse_col,"best_st_rmse","abs_gain","pct_gain"
        ])

    # keep all TL executions
    tl_runs = df[df["upstream"].notna()].dropna(subset=[rmse_col]).copy()

    if tl_runs.empty:
        return pd.DataFrame(columns=[
            "sample","strategy","imputation","upstream",
            rmse_col,"best_st_rmse","abs_gain","pct_gain"
        ])

    # merge ST baseline
    tl_runs = tl_runs.merge(best_st, on="sample", how="left")

    # drop TL runs without ST baseline
    tl_runs = tl_runs.dropna(subset=["best_st_rmse"]).copy()

    # gains per execution
    tl_runs["abs_gain"] = tl_runs["best_st_rmse"] - tl_runs[rmse_col]
    tl_runs["pct_gain"] = tl_runs["abs_gain"] / tl_runs["best_st_rmse"]

    return tl_runs[[
        "sample","strategy","imputation","upstream",
        rmse_col,"best_st_rmse","abs_gain","pct_gain"
    ]]

## ===============================
## graphs gain of transfer
## ===============================

def plot_boxplot_gain(tl_gain_df: pd.DataFrame, out_dir: str, x_name: str = "strategy", hue: str = "upstream",
                      title: str = "Transfer Learning Gain by Strategy", save_name: str = "boxplot_gain.png"):
    """
    Boxplot (or violin) of pct_gain grouped by strategy.
    hue can be 'upstream' or 'sample' (or other categorical column present in tl_gain_df).
    Saves PNG into out_dir.
    """
    _ensure_outdir(out_dir)
    df = tl_gain_df.copy()
    # Convert pct to percent for plotting
    df["pct_gain_pct"] = df["pct_gain"] * 100.0

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x=x_name, y="pct_gain_pct", hue=hue, dodge=True)
    sns.stripplot(data=df, x=x_name, y="pct_gain_pct", hue=hue, dodge=True, color="black", size=3, alpha=0.3, linewidth=0)
    # Remove duplicate legend entries (stripplot added)
    handles, labels = ax.get_legend_handles_labels()
    # keep only first set
    n_unique = len(df[hue].unique()) if hue in df.columns else 0
    if n_unique > 0:
        ax.legend(handles[:n_unique], labels[:n_unique], title=hue)
    else:
        ax.get_legend().remove()

    ax.set_ylabel("Transfer Gain (%)", fontsize=12)
    ax.set_xlabel(x_name.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path_complete = os.path.join(out_dir, save_name)
    plt.savefig(path_complete, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap_gain_facets_strategies(tl_gain_df: pd.DataFrame, out_dir: str,
                                      title_template: str = "Percent Gain (Strategy: {up})",
                                      save_name_prefix: str = "heatmap_gain_strategy"):
    """
    For each strategy, produce a heatmap with rows = sample, cols = upstream, values = pct_gain (%).
    Uses the same color scale across all strategy facets.
    Saves each strategy heatmap as a separate PNG.
    """
    _ensure_outdir(out_dir)
    # prepare pivot values (pct in percent)
    tl_gain_df = tl_gain_df.copy()
    tl_gain_df["pct_gain_pct"] = tl_gain_df["pct_gain"] * 100.0

    # global vmin/vmax across all upstreams for consistent color scale
    if tl_gain_df["pct_gain_pct"].empty:
        return
    vmin = tl_gain_df["pct_gain_pct"].min()
    vmax = tl_gain_df["pct_gain_pct"].max()

    for up, grp in tl_gain_df.groupby("strategy"):
        pivot = grp.pivot_table(index="sample", columns="upstream", values="pct_gain_pct", aggfunc="mean")
        plt.figure(figsize=(2 * max(4, pivot.shape[1]), max(4, pivot.shape[0] * 0.3)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Percent Gain (%)"},
            linewidths=0.4, linecolor="white"
        )
        plt.title(title_template.format(up=up), fontsize=14)
        plt.xlabel("Upstream")
        plt.ylabel("Sample")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{save_name_prefix}_up_{up}.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()

def plot_heatmap_gain_facets_upstream(tl_gain_df: pd.DataFrame, out_dir: str,
                                      title_template: str = "Percent Gain (Upstream: {up})",
                                      save_name_prefix: str = "heatmap_gain_upstream"):
    """
    For each upstream, produce a heatmap with rows = sample, cols = strategy, values = pct_gain (%).
    Uses the same color scale across all upstream facets.
    Saves each upstream heatmap as a separate PNG.
    """
    _ensure_outdir(out_dir)
    # prepare pivot values (pct in percent)
    tl_gain_df = tl_gain_df.copy()
    tl_gain_df["pct_gain_pct"] = tl_gain_df["pct_gain"] * 100.0

    # global vmin/vmax across all upstreams for consistent color scale
    if tl_gain_df["pct_gain_pct"].empty:
        return
    vmin = tl_gain_df["pct_gain_pct"].min()
    vmax = tl_gain_df["pct_gain_pct"].max()

    for up, grp in tl_gain_df.groupby("upstream"):
        pivot = grp.pivot_table(index="sample", columns="strategy", values="pct_gain_pct", aggfunc="mean")
        plt.figure(figsize=(2 * max(4, pivot.shape[1]), max(4, pivot.shape[0] * 0.3)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Percent Gain (%)"},
            linewidths=0.4, linecolor="white"
        )
        plt.title(title_template.format(up=up), fontsize=14)
        plt.xlabel("Strategy")
        plt.ylabel("Sample")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{save_name_prefix}_up_{up}.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()

def _plot_heatmap_gain_generic(
    df: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    out_dir: str,
    title: str,
    save_name: str,
    agg_fn,
):
    _ensure_outdir(out_dir)

    # regra: imputation sempre no eixo X e deve ter >1 nível
    if col_factor == "imputation" and df["imputation"].nunique() <= 1:
        print(f"[SKIP] Apenas uma imputação encontrada. Ignorando: {save_name}")
        return

    work = df.copy()
    work["pct_gain_pct"] = work["pct_gain"] * 100.0

    agg = (
        work.groupby([row_factor, col_factor])["pct_gain_pct"]
        .agg(agg_fn)
        .reset_index()
    )

    pivot = agg.pivot(index=row_factor, columns=col_factor, values="pct_gain_pct")

    vmin = np.nanmin(agg["pct_gain_pct"])
    vmax = np.nanmax(agg["pct_gain_pct"])

    plt.figure(figsize=(2 * max(4, pivot.shape[1]), max(4, pivot.shape[0] * 0.3)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Percent Gain (%)"},
        linewidths=0.4,
        linecolor="white",
    )

    plt.title(title, fontsize=14)
    plt.xlabel(col_factor.capitalize())
    plt.ylabel(row_factor.capitalize())
    plt.tight_layout()

    path_complete = os.path.join(out_dir, save_name)
    plt.savefig(path_complete, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_heatmap_best_gain(
    df: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    out_dir: str,
    title: str,
    save_name: str,
):
    _plot_heatmap_gain_generic(
        df=df,
        row_factor=row_factor,
        col_factor=col_factor,
        out_dir=out_dir,
        title=title,
        save_name=save_name,
        agg_fn="max",
    )

def _plot_heatmap_mean_gain(
    df: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    out_dir: str,
    title: str,
    save_name: str,
):
    _plot_heatmap_gain_generic(
        df=df,
        row_factor=row_factor,
        col_factor=col_factor,
        out_dir=out_dir,
        title=title,
        save_name=save_name,
        agg_fn="mean",
    )

def plot_heatmap_sample_x_strategy_best(tl_gain_df, out_dir):
    _plot_heatmap_best_gain(
        tl_gain_df, "sample", "strategy", out_dir,
        "Best Percent Gain (Sample x Strategy)",
        "heatmap_best_sample_x_strategy.png"
    )


def plot_heatmap_sample_x_imputation_best(tl_gain_df, out_dir):
    _plot_heatmap_best_gain(
        tl_gain_df, "sample", "imputation", out_dir,
        "Best Percent Gain (Sample x Imputation)",
        "heatmap_best_sample_x_imputation.png"
    )


def plot_heatmap_strategy_x_imputation_best(tl_gain_df, out_dir):
    _plot_heatmap_best_gain(
        tl_gain_df, "strategy", "imputation", out_dir,
        "Best Percent Gain (Strategy x Imputation)",
        "heatmap_best_strategy_x_imputation.png"
    )

def plot_heatmap_sample_x_strategy_mean(tl_gain_df, out_dir):
    _plot_heatmap_mean_gain(
        tl_gain_df, "sample", "strategy", out_dir,
        "Mean Percent Gain (Sample x Strategy)",
        "heatmap_mean_sample_x_strategy.png"
    )


def plot_heatmap_sample_x_imputation_mean(tl_gain_df, out_dir):
    _plot_heatmap_mean_gain(
        tl_gain_df, "sample", "imputation", out_dir,
        "Mean Percent Gain (Sample x Imputation)",
        "heatmap_mean_sample_x_imputation.png"
    )


def plot_heatmap_strategy_x_imputation_mean(tl_gain_df, out_dir):
    _plot_heatmap_mean_gain(
        tl_gain_df, "strategy", "imputation", out_dir,
        "Mean Percent Gain (Strategy x Imputation)",
        "heatmap_mean_strategy_x_imputation.png"
    )

def plot_heatmap_best_gain_overall(tl_gain_df: pd.DataFrame, out_dir: str):
    plot_heatmap_sample_x_strategy_best(tl_gain_df, out_dir)
    plot_heatmap_sample_x_imputation_best(tl_gain_df, out_dir)
    plot_heatmap_strategy_x_imputation_best(tl_gain_df, out_dir)

def plot_heatmap_mean_gain_overall(tl_gain_df: pd.DataFrame, out_dir: str):
    plot_heatmap_sample_x_strategy_mean(tl_gain_df, out_dir)
    plot_heatmap_sample_x_imputation_mean(tl_gain_df, out_dir)
    plot_heatmap_strategy_x_imputation_mean(tl_gain_df, out_dir)

def select_best_tl_run(
    tl_gain_df: pd.DataFrame,
    rmse_col: str = "rmse_test"
) -> pd.DataFrame:
    """
    Selects the best TL execution per
    (sample, strategy, imputation, upstream).

    Best = minimal rmse_test (which also implies max gain).
    """
    if tl_gain_df.empty:
        return tl_gain_df.copy()

    idx = (
        tl_gain_df
        .groupby(["sample", "strategy", "imputation", "upstream"])[rmse_col]
        .idxmin()
    )

    return tl_gain_df.loc[idx].reset_index(drop=True)

def plot_barplot_best_tl_vs_best_st(
    tl_gain_df: pd.DataFrame,
    out_dir: str,
    title_template: str = "Best TL vs FS (Upstream: {up})",
    save_prefix: str = "barplot_best_vs_st"
):
    """
    Plota Best TL vs From Scratch (FS) por upstream.

    Espera tl_gain_df no nível PER RUN.
    Internamente:
        - seleciona a melhor execução via select_best_tl_run
        - assume best_st_rmse constante por sample
    """
    out_dir = os.path.join(out_dir, "barplot_by_upstream")
    _ensure_outdir(out_dir)

    if tl_gain_df.empty:
        return

    # ----------------------------------
    # Seleciona melhor TL por configuração
    # ----------------------------------
    best_tl = select_best_tl_run(tl_gain_df)

    # ----------------------------------
    # Prepara DF TL vs FS
    # ----------------------------------
    comp_df = best_tl[[
        "upstream",
        "strategy",
        "sample",
        "rmse_test",
        "best_st_rmse"
    ]].rename(columns={
        "rmse_test": "tl_rmse",
        "best_st_rmse": "st_rmse"
    })

    # ----------------------------------
    # Melt para formato longo
    # ----------------------------------
    plot_df = comp_df.melt(
        id_vars=["upstream", "strategy", "sample"],
        value_vars=["tl_rmse", "st_rmse"],
        var_name="kind",
        value_name="rmse"
    )

    plot_df["kind"] = plot_df["kind"].map({
        "tl_rmse": "With Transfer Learning",
        "st_rmse": "From Scratch"
    })

    # Score = -RMSE
    plot_df["score"] = -plot_df["rmse"]

    # Eixo X = (strategy, sample)
    plot_df["x"] = plot_df.apply(
        lambda r: f"{r['strategy']}-{r['sample']}",
        axis=1
    )

    ordered_categories = (
        plot_df
        .sort_values(["strategy", "sample", "kind"])["x"]
        .unique()
    )

    plot_df["x"] = pd.Categorical(
        plot_df["x"],
        categories=ordered_categories,
        ordered=True
    )

    # -----------------------------
    # PLOTS INDIVIDUAIS POR UPSTREAM
    # -----------------------------
    for up, grp in plot_df.groupby("upstream"):
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(data=grp, x="x", y="score", hue="kind")

        ax.set_title(title_template.format(up=up))
        ax.set_xlabel("(Strategy, Sample Size)")
        ax.set_ylabel("Score (-RMSE — higher is better)")
        plt.xticks(rotation=45, ha="right")

        # Annotar sample acima das barras
        for p, (_, row) in zip(ax.patches, grp.iterrows()):
            ax.annotate(
                str(row["sample"]),
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=8
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{save_prefix}_up_{up}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    # -----------------------------
    # FACETGRID — todos upstreams
    # -----------------------------
    g = sns.FacetGrid(
        plot_df,
        col="upstream",
        sharey=False,
        height=4,
        aspect=1.5
    )
    g.map_dataframe(
        sns.barplot,
        x="x", y="score", hue="kind", dodge=True
    )

    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    g.add_legend()
    g.fig.suptitle("Best TL vs From Scratch — All Upstreams", y=1.03)

    plt.savefig(
        os.path.join(out_dir, f"{save_prefix}_facetgrid.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

def graphs_analysis_gain_by_transfer_learning(df: pd.DataFrame, tl_gain_df: pd.DataFrame, out_path: str):
    # 1) Boxplot: gain by strategy, hue=upstream 
    plot_boxplot_gain(tl_gain_df, out_dir=out_path, x_name="strategy", hue="upstream", title="Transfer Learning Gain by Strategy (hue=upstream)",
                      save_name="boxplot_gain_by_strategy_hue_upstream.png")

    
    plot_boxplot_gain(tl_gain_df, out_dir=out_path, x_name="strategy", hue="imputation", title="Transfer Learning Gain by Strategy (hue=imputation)",
                      save_name="boxplot_gain_by_strategy_hue_imputation.png")
    
    plot_boxplot_gain(tl_gain_df, out_dir=out_path, x_name="imputation", hue="upstream", title="Transfer Learning Gain by Imputation (hue=upstream)",
                      save_name="boxplot_gain_by_imputation_hue_upstream.png")
    
    # 2) Heatmap facets by upstream: sample x strategy with percent gain
    plot_heatmap_gain_facets_upstream(tl_gain_df, out_dir=out_path)

    # 3) Heatmap facets by strategy: sample x upstream with percent gain
    plot_heatmap_gain_facets_strategies(tl_gain_df, out_dir=out_path)

    # 4) Heatmap best gain overall (best across upstreams)
    plot_heatmap_best_gain_overall(tl_gain_df, out_dir=out_path)
    plot_heatmap_mean_gain_overall(tl_gain_df, out_dir=out_path)
    # 5) Barplot grouped: Best TL vs Best ST per strategy (per upstream facet)
    plot_barplot_best_tl_vs_best_st(tl_gain_df, out_dir=out_path)

def analysis_gain_by_transfer_learning(df: pd.DataFrame, out_path: str):
    """
    Main entry point. Produces:
      - CSVs: tl_gain_df.csv
      - Boxplot of gains (hue=upstream by default)
      - Heatmaps per upstream (sample x strategy)
      - Heatmap of best gain overall (best upstream per sample,strategy)
      - Barplots comparing best TL vs best ST per strategy (faceted by upstream, saved one per upstream)
      - Graphs agrouped by imputation
    The modular plotting functions are called below and can be reused separately.
    """
    base_out = os.path.join(out_path, "gain_by_transfer_learning")
    _ensure_outdir(base_out)

    # Prepare the TL gain DataFrame
    tl_gain_df = build_tl_gain_per_run_df(df, rmse_col="rmse_test")
    tl_gain_csv = os.path.join(base_out, "tl_gain_df.csv")
    tl_gain_df.to_csv(tl_gain_csv, index=False)

    out_path_all = os.path.join(base_out, "all_data")
    _ensure_outdir(out_path_all)
    graphs_analysis_gain_by_transfer_learning(df, tl_gain_df, out_path=out_path_all)
    anova_analysis(tl_gain_df, out_dir=out_path_all, dv="pct_gain")
    # ==============================================================
    # 2. PROCESSO POR IMPUTAÇÃO (RECORTES)
    # ==============================================================

    imputations = sorted(df["imputation"].dropna().unique())

    for imp in imputations:
        imp_out = os.path.join(base_out, f"imputation={imp}")
        _ensure_outdir(imp_out)

        # Recorte do TL (FS continua o mesmo)
        tl_imp = tl_gain_df[tl_gain_df["imputation"] == imp]

        if tl_imp.empty:
            continue
        #o df passado não precisa ser filtrado pela imputation, pois o FS não tem imputation msm então não faz diferença
        graphs_analysis_gain_by_transfer_learning(df, tl_imp, out_path=imp_out)

    # Save a brief summary CSV as well
    summary_csv = os.path.join(base_out, "summary_gain_statistics.csv")
    if not tl_gain_df.empty:
        summary = tl_gain_df.groupby(["upstream", "strategy", "imputation"]).agg(
            mean_pct_gain = ("pct_gain", "mean"),
            median_pct_gain = ("pct_gain", "median"),
            count = ("pct_gain", "count")
        ).reset_index()
        summary["mean_pct_gain_pct"] = summary["mean_pct_gain"] * 100.0
        summary.to_csv(summary_csv, index=False)
    else:
        pd.DataFrame().to_csv(summary_csv, index=False)

    print(f"[DONE] All outputs saved in {base_out}")
    return base_out



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

    rank_df = build_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)
    plot_and_save_heatmap(rank_df, name="geral", out_dir=path)
    plot_BoxPlots_overfitting(df, out_dir=path)
    summarize_results(df, out_dir=path)
    analyze_training_curves(df, out_dir=path)

    has_multiple_upstreams = True
    if has_multiple_upstreams:
        print("Múltiplos upstreams detectados — executando análises adicionais...")

        rank_mean_df = build_rank_table(df, alpha=0.05, min_seeds=2, group_field="upstream", verbose=False)
        plot_and_save_heatmap(rank_mean_df, name="média-por-upstream", out_dir=path)
    
        mean_rank_df = build_statistical_mean_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)

        compute_spearman_corr_between_upstreams(mean_rank_df, out_dir=path)

        plot_heatmap_mean_rank_upstream_strategy(mean_rank_df, out_dir=path)
        mean_rank_upstream_df = build_upstream_rank_by_strategy(df, alpha=0.05, min_seeds=2)

        plot_heatmap_mean_rank_strategy_upstream(mean_rank_upstream_df, out_dir=path)
        
        mean_rank_imp_up = build_upstream_rank_by_imputation(df)
        plot_heatmap_mean_rank_imputation_upstream(mean_rank_imp_up, out_dir=path)
        anova_analysis(df, out_dir=path)
        analysis_gain_by_transfer_learning(df, out_path=path)

