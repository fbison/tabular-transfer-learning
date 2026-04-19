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
from . import helpers as h
from .anova import anova_analysis
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_best_st_per_group(
    df: pd.DataFrame,
    compare_with_imputed_fs: bool,
    rmse_col: str = "rmse_test"
    ) -> pd.DataFrame:
    """
    Returns DataFrame with columns: sample, best_st_rmse
    Considers ST rows as upstream == None.
    Uses rmse_col (default rmse_test). Drops NaNs.
    """
    colluns_to_group_by = ["sample", "hyp_source"] #add seed?
    st_df = df[df["strategy"] == "FS"].dropna(subset=[rmse_col])
    if compare_with_imputed_fs:
        st_df = st_df[st_df["imputation"] != "Not Used"]
    else:
        st_df = st_df[st_df["imputation"] == "Not Used"]

    if st_df.empty:
        return pd.DataFrame(columns=["sample", "best_st_rmse"])
    best_st = st_df.groupby(colluns_to_group_by)[rmse_col].min().reset_index().rename(columns={rmse_col: "best_st_rmse"})
    return best_st

def build_tl_gain_per_run_df(
    df: pd.DataFrame,
    compare_with_imputed_fs: bool,
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
    best_st = compute_best_st_per_group(df, compare_with_imputed_fs, rmse_col=rmse_col)

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


def _plot_boxplot_base(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_dir: str,
    hue: str,
    ylabel: str = "",
    title: str = "",
    save_name: str = "boxplot",
):

    h.ensure_outdir(out_dir)

    # --- validações ---
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame")
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame")
    if hue is not None and hue not in df.columns:
        hue = None  # fallback seguro

    # --- limpeza de dados ---
    subset_cols = [x, y] + ([hue] if hue else [])
    df = df.dropna(subset=subset_cols)

    # --- remover categorias não usadas (resolve espaços vazios) ---
    if pd.api.types.is_categorical_dtype(df[x]):
        df[x] = df[x].cat.remove_unused_categories()

    if hue and pd.api.types.is_categorical_dtype(df[hue]):
        df[hue] = df[hue].cat.remove_unused_categories()

    # --- ordenação do eixo X ---
    x_order = sorted(df[x].dropna().unique().tolist())


    # --- ordenação do hue (apenas se for upstream) ---
    hue_order = None
    unique_vals = df[hue].dropna().unique()
    if hue in ["upstream", "sample_size", "n_samples"]:
        try:
            hue_order = sorted(unique_vals, key=lambda v: float(v))
        except:
            hue_order = sorted(unique_vals)
    
    if "Not Used" in unique_vals:
        hue_order = ["Not Used"] + sorted([v for v in unique_vals if v != "Not Used"])
    


    # --- plot ---
    plt.figure(figsize=(10, 6))

    ax = sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        dodge=True,
        order=x_order,
        hue_order=hue_order,
    )
    
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        dodge=True,
        order=x_order,
        hue_order=hue_order,
        color="black",
        size=3,
        alpha=0.3,
        linewidth=0,
    )

    # overlay manually colored points
    norm = plt.Normalize(df["sample"].min(), df["sample"].max())
    cmap = plt.cm.viridis

    # Each PathCollection = one (x, hue) group
    collections = sp.collections

    i = 0
    for (x_val) in x_order:
        for (h_val) in (hue_order if hue else [None]):
            if i >= len(collections):
                continue

            coll = collections[i]

            # filter df for this subgroup
            if hue:
                sub = df[(df[x] == x_val) & (df[hue] == h_val)]
            else:
                sub = df[df[x] == x_val]

            if len(sub) == 0:
                i += 1
                continue

            colors = cmap(norm(sub["sample"].values))

            coll.set_facecolors(colors)
            coll.set_alpha(0.6)

            i += 1

    # --- legenda sem duplicação ---
    handles, labels = ax.get_legend_handles_labels()
    if hue is not None:
        unique_vals = df[hue].dropna().unique()
        ax.legend(handles[:len(unique_vals)], labels[:len(unique_vals)], title=hue)
    else:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # --- labels ---
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(x.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Sample Size")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path_complete = os.path.join(out_dir, save_name)
    h.plot_save_fig(path_complete)

def plot_boxplot_score_distribution(
    df: pd.DataFrame,
    out_dir: str,
    x_name: str = "strategy",
    hue: str = "upstream",
    title: str = "Distribution of Score (-RMSE) by Strategy",
    save_name: str = "boxplot_score",
):
    """
    Boxplot do score (-rmse_test) agrupado por strategy.
    """

    df_plot = df.copy()

    if "rmse_test" not in df_plot.columns:
        raise ValueError("Column 'rmse_test' not found")

    # 🔥 importante: converter para score
    df_plot["score"] = -df_plot["rmse_test"]


    _plot_boxplot_base(
        df=df_plot,
        x=x_name,
        y="score",
        hue=hue,
        out_dir=out_dir,
        ylabel="Score (-RMSE)",
        title=title,
        save_name=save_name,
    )

def plot_boxplot_gain(
    tl_gain_df: pd.DataFrame,
    out_dir: str,
    x_name: str = "strategy",
    hue: str = "upstream",
    title: str = "Transfer Learning Gain by Strategy",
    save_name: str = "boxplot_gain",
):
    """
    Boxplot do ganho percentual (pct_gain * 100).
    """

    df = tl_gain_df.copy()

    if "pct_gain" not in df.columns:
        raise ValueError("Column 'pct_gain' not found")

    df["pct_gain_pct"] = df["pct_gain"] * 100.0

    _plot_boxplot_base(
        df=df,
        x=x_name,
        y="pct_gain_pct",
        hue=hue,
        out_dir=out_dir,
        ylabel="Transfer Gain (%)",
        title=title,
        save_name=save_name,
    )

def plot_heatmap_gain_facets_strategies(tl_gain_df: pd.DataFrame, out_dir: str,
                                      title_template: str = "Percent Gain (Strategy: {up})",
                                      save_name_prefix: str = "heatmap_gain_strategy"):
    """
    For each strategy, produce a heatmap with rows = sample, cols = upstream, values = pct_gain (%).
    Uses the same color scale across all strategy facets.
    Saves each strategy heatmap as a separate PNG.
    """
    h.ensure_outdir(out_dir)
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
        fname = os.path.join(out_dir, f"{save_name_prefix}_up_{up}")
        h.plot_save_fig(fname)

def plot_heatmap_gain_facets_upstream(tl_gain_df: pd.DataFrame, out_dir: str,
                                      title_template: str = "Percent Gain (Upstream: {up})",
                                      save_name_prefix: str = "heatmap_gain_upstream"):
    """
    For each upstream, produce a heatmap with rows = sample, cols = strategy, values = pct_gain (%).
    Uses the same color scale across all upstream facets.
    Saves each upstream heatmap as a separate PNG.
    """
    h.ensure_outdir(out_dir)
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
        fname = os.path.join(out_dir, f"{save_name_prefix}_up_{up}")
        h.plot_save_fig(fname)

def _plot_heatmap_gain_generic(
    df: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    out_dir: str,
    title: str,
    save_name: str,
    agg_fn,
):
    h.ensure_outdir(out_dir)

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
    h.plot_save_fig(path_complete)

AGGREGATIONS = {
    "best": ("max", "Best"),
    "mean": ("mean", "Mean"),
    "median": ("median", "Median"),
}


def _plot_heatmap_by_agg(
    df: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    out_dir: str,
    agg_key: str,
):
    agg_fn, label = AGGREGATIONS[agg_key]

    title = f"{label} Percent Gain ({row_factor.capitalize()} x {col_factor.capitalize()})"
    save_name = f"heatmap_{agg_key}_{row_factor}_x_{col_factor}"

    _plot_heatmap_gain_generic(
        df=df,
        row_factor=row_factor,
        col_factor=col_factor,
        out_dir=out_dir,
        title=title,
        save_name=save_name,
        agg_fn=agg_fn,
    )


def _plot_heatmap_all_combinations(
    df: pd.DataFrame,
    out_dir: str,
    agg_key: str,
):
    _plot_heatmap_by_agg(df, "sample", "strategy", out_dir, agg_key)
    _plot_heatmap_by_agg(df, "sample", "imputation", out_dir, agg_key)
    _plot_heatmap_by_agg(df, "strategy", "imputation", out_dir, agg_key)


def plot_heatmap_gain_overall(
    tl_gain_df: pd.DataFrame,
    out_dir: str,
    agg_key: str,
):
    out_dir = os.path.join(out_dir, f"overall_{agg_key}_gain")
    _plot_heatmap_all_combinations(tl_gain_df, out_dir, agg_key)

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
    h.ensure_outdir(out_dir)

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
        path_complete = os.path.join(out_dir, f"{save_prefix}_up_{up}")
        h.plot_save_fig(path_complete)
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

    h.plot_save_fig(os.path.join(out_dir, f"{save_prefix}_facetgrid"))
    plt.close()


def distribution_of_gain_by_transfer_learning(tl_gain_df: pd.DataFrame, compare_with_imputed_fs: bool, out_path: str):
    tittlePrefix = "Transfer Learning Gain"
    sufix = "(compared with imputed FS)" if compare_with_imputed_fs else "(compared with non-imputed FS)"
    if not compare_with_imputed_fs:
        tittlePrefix += " and Imputation Gain"

    plot_boxplot_gain(tl_gain_df, out_dir=out_path, x_name="strategy", hue="upstream", title=f"{tittlePrefix} by Strategy {sufix}",
                      save_name="boxplot_gain_by_strategy_hue_upstream")

    
    plot_boxplot_gain(tl_gain_df, out_dir=out_path, x_name="strategy", hue="imputation", title=f"{tittlePrefix} by Strategy {sufix}",
                      save_name="boxplot_gain_by_strategy_hue_imputation")
    
    plot_boxplot_gain(tl_gain_df, out_dir=out_path, x_name="imputation", hue="upstream", title=f"{tittlePrefix} by Imputation {sufix}",
                      save_name="boxplot_gain_by_imputation_hue_upstream")

def distribution_of_score_by_transfer_learning(tl_gain_df: pd.DataFrame, out_path: str):
    plot_boxplot_score_distribution(tl_gain_df, out_dir=out_path, x_name="strategy", hue="upstream", title="Score (-RMSE) by Strategy",
                      save_name="boxplot_score_by_strategy_hue_upstream")

    
    plot_boxplot_score_distribution(tl_gain_df, out_dir=out_path, x_name="strategy", hue="imputation", title="Score (-RMSE) by Strategy (hue=imputation)",
                      save_name="boxplot_score_by_strategy_hue_imputation")
    
    plot_boxplot_score_distribution(tl_gain_df, out_dir=out_path, x_name="imputation", hue="upstream", title="Score (-RMSE) by Imputation (hue=upstream)",
                      save_name="boxplot_score_by_imputation_hue_upstream")

def graphs_analysis_gain_by_transfer_learning(
        df: pd.DataFrame, tl_gain_df: pd.DataFrame,
        compare_with_imputed_fs: bool, out_path: str):
    distribution_of_gain_by_transfer_learning( tl_gain_df, compare_with_imputed_fs, out_path)
    
    # 2) Heatmap facets by upstream: sample x strategy with percent gain
    plot_heatmap_gain_facets_upstream(tl_gain_df, out_dir=out_path)

    # 3) Heatmap facets by strategy: sample x upstream with percent gain
    plot_heatmap_gain_facets_strategies(tl_gain_df, out_dir=out_path)

    # 4) Heatmap best gain overall (best across upstreams)
    plot_heatmap_gain_overall(tl_gain_df, out_path, "best")
    plot_heatmap_gain_overall(tl_gain_df, out_path, "mean")
    plot_heatmap_gain_overall(tl_gain_df, out_path, "median")

    # 5) Barplot grouped: Best TL vs Best ST per strategy (per upstream facet)
    plot_barplot_best_tl_vs_best_st(tl_gain_df, out_dir=out_path)

def analysis_per_imputation_gain_by_transfer_learning(df: pd.DataFrame, tl_gain_df: pd.DataFrame, out_path: str):
    out_path = os.path.join(out_path, "per_imputation")
    imputations = sorted(df["imputation"].dropna().unique())

    for imp in imputations:
        imp_out = os.path.join(out_path, f"imputation={imp}")
        h.ensure_outdir(imp_out)

        # Recorte do TL (FS continua o mesmo)
        tl_imp = tl_gain_df[tl_gain_df["imputation"] == imp]

        if tl_imp.empty:
            continue
        #o df passado não precisa ser filtrado pela imputation, pois o FS não tem imputation msm então não faz diferença
        graphs_analysis_gain_by_transfer_learning(df, tl_imp, out_path=imp_out)

def analysis_gain_by_transfer_learning(
        df: pd.DataFrame,
        compare_with_imputed_fs: bool,
        out_path: str
    ):
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
    directory_name = "gain_by_transfer_learning"
    if not compare_with_imputed_fs:
        directory_name += "_and_imputation"

    base_out = os.path.join(out_path, directory_name)
    h.ensure_outdir(base_out)

    # Prepare the TL gain DataFrame
    tl_gain_df = build_tl_gain_per_run_df(df, compare_with_imputed_fs, rmse_col="rmse_test")
    tl_gain_csv = os.path.join(base_out, "tl_gain_df.csv")
    tl_gain_df.to_csv(tl_gain_csv, index=False)

    out_path_all = os.path.join(base_out, "all_data")
    h.ensure_outdir(out_path_all)
    graphs_analysis_gain_by_transfer_learning(df, tl_gain_df, compare_with_imputed_fs, out_path=out_path_all)
    anova_analysis(tl_gain_df, out_dir=out_path_all, dv="pct_gain")
    # ==============================================================
    # 2. PROCESSO POR IMPUTAÇÃO (RECORTES)
    # ==============================================================

    analysis_per_imputation_gain_by_transfer_learning(tl_gain_df, compare_with_imputed_fs, out_path=base_out)

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