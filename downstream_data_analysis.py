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
import matplotlib.pyplot as plt

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
    Retorna None se não houver upstream na string.
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
#                "upstream": parse_upstream_from_model_path_name(cfg.get("model", {}).get("model_path", "")),
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

def plot_and_save_heatmap(rank_df, out_dir, name= "", strategies_order=None, imputations_order=None):
    os.makedirs(out_dir, exist_ok=True)
    
    
    if imputations_order is None:
        imputations_order = sorted(rank_df['imputation'].unique())
    
    n_imputations = len(imputations_order)

    # Determine global min/max rank for consistent color scale
    vmin = rank_df["rank"].min()
    vmax = rank_df["rank"].max()

    # Equal width for each imputation subplot
    fig, axes = plt.subplots(
        1, n_imputations,
        figsize=(2 * sum_strategies(imputations_order), 4),
        sharey=True,
        gridspec_kw={"width_ratios": [1] * n_imputations}
    )

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
    path_complete = os.path.join(out_dir, f"heatmap-{name}")
    plt.savefig((path_complete + ".png"), dpi=300, bbox_inches="tight")
    #plt.savefig((path_complete + ".pdf"), bbox_inches="tight")
    #plt.savefig((path_complete + ".svg"), bbox_inches="tight")

    with open((path_complete + ".fig.pickle"), "wb") as f:
        pickle.dump(fig, f)
    plt.savefig("heatmap.eps", format="eps", bbox_inches="tight")

    #plt.show()

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
    plt.figure(figsize=(10, 6))

    for c in mean_curves:
        label = f"{c['strategy']} / {c['imputation']}"
        x = range(len(c["mean_rmse"]))

        # Curva média
        line_mean, = plt.plot(
            x,
            c["mean_rmse"],
            lw=2,
            label=label
        )

        color = line_mean.get_color()

        # Curva mediana (tracejada, mesma cor)
        plt.plot(
            x,
            c["median_rmse"],
            linestyle="--",
            color=color,
            lw=1.5
        )

        # Faixa de ±1 desvio padrão (shaded area)
        plt.fill_between(
            x,
            c["mean_rmse"] - c["std_rmse"],
            c["mean_rmse"] + c["std_rmse"],
            color=color,
            alpha=0.15
        )

    # === título e legendas ===
    plt.title("Convergência Média por Estratégia e Imputação")
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
            Line2D([0], [0], color=plt.cm.tab10(i), lw=2, label=f"{c['strategy']} / {c['imputation']}")
            for i, c in enumerate(mean_curves)
        ],
        title="Texturas e cores:",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # === salvar ===
    path_complete = os.path.join(out_dir, "ConvergenciaMedia_Todas")
    plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
    #plt.show()

# ANOVA 2-way (upstream x strategy) dentro de cada imputação
import statsmodels.api as sm
import statsmodels.formula.api as smf

def anova_two_way_by_imputation(df, dv="rmse", factor1="upstream", factor2="strategy", min_rows=10, out_dir=None):
    """
    Roda ANOVA 2-way (tipo II) para cada imputation separadamente.
    Imprime a ANOVA table (SS, DF, F, PR(>F)) e effect sizes (eta2, partial eta2).
    """
    imputations = df['imputation'].unique()
    results = {}
    for imp in sorted(imputations):
        sub = df[df['imputation'] == imp].dropna(subset=[dv, factor1, factor2])
        if sub.shape[0] < min_rows:
            print(f"[ANOVA] imputation={imp}: dados insuficientes ({sub.shape[0]} linhas), pulando.")
            continue

        # Garantir categorias
        sub[factor1] = sub[factor1].astype('category')
        sub[factor2] = sub[factor2].astype('category')

        # Formula: dv ~ C(upstream) * C(strategy)
        formula = f"{dv} ~ C({factor1}) * C({factor2})"
        model = smf.ols(formula, data=sub).fit()
        aov = sm.stats.anova_lm(model, typ=2)  # tipo II

        # effect sizes
        ss_total = sum(aov['sum_sq'])
        aov = aov.assign(eta2 = aov['sum_sq'] / ss_total)
        # partial eta2 = SS_effect / (SS_effect + SS_error)
        ss_error = aov.loc['Residual', 'sum_sq'] if 'Residual' in aov.index else model.ssr
        partials = {}
        for idx in aov.index:
            if idx == 'Residual':
                aov.loc[idx, 'partial_eta2'] = np.nan
            else:
                ss_effect = aov.loc[idx, 'sum_sq']
                aov.loc[idx, 'partial_eta2'] = ss_effect / (ss_effect + ss_error)

        print("\n" + "="*80)
        print(f"ANOVA (2-way) para imputation = {imp}")
        print("="*80)
        print(aov)  # se estiver no notebook; senão use print(aov)
        results[imp] = aov

        # opcional: salvar tabela como csv
        if out_dir:
            aov.to_csv(os.path.join(out_dir, f"anova_2way_imputation_{imp}.csv"))

    return results


# =========================================================
#  Função auxiliar: ranking estatístico (Mann-Whitney)
# =========================================================
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


# =========================================================
# 2️⃣ Rank dos upstreams para cada strategy/imputation (comparativo entre upstreams)
# =========================================================
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
        plt.title(f"Ranking de Upstreams por Estratégia (Imputação: {imp})")
        plt.xlabel("Upstream")
        plt.ylabel("Estratégia")
        plt.tight_layout()
        path_complete = os.path.join(out_dir, "heatmap_mean_rank_strategy_upstream")
        plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")

# ---------------------------
# Example usage
# ---------------------------


if __name__ == "__main__":
    # path to folder containing results.jsonl
    for experiment in ["all_experiments"]: #, "ic_upstream2", "ic_upstream3", "ic_upstream4"]:
        path = fr"C:\usp\tabular-transfer-learning\outputs\transfer-learning-from-upstream\{experiment}"

        jsonl = os.path.join(path, "results.jsonl")

        df = load_results(jsonl, prefer="test")

        rank_df = build_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)
        plot_and_save_heatmap(rank_df, name="geral", out_dir=path)
        plot_BoxPlots_overfitting(df, out_dir=path)
        summarize_results(df, out_dir=path)
        #analyze_training_curves(df, out_dir=path)
        if experiment.startswith("all_experiments"):
            rank_mean_df = build_rank_table(df, alpha=0.05, min_seeds=2, group_field="upstream", verbose=False)
            plot_and_save_heatmap(rank_mean_df, name="média-por-upstream", out_dir=path)
            
            mean_rank_df = build_statistical_mean_rank_table(df, alpha=0.05, min_seeds=2, verbose=False)

            corr_upstreams = compute_spearman_corr_between_upstreams(mean_rank_df, out_dir=path)

            plot_heatmap_mean_rank_upstream_strategy(mean_rank_df, out_dir=path)

            mean_rank_upstream_df = build_upstream_rank_by_strategy(df, alpha=0.05, min_seeds=2)

            plot_heatmap_mean_rank_strategy_upstream(mean_rank_upstream_df, out_dir=path)
            anova_two_way_by_imputation(df, out_dir=path)
