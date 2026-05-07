import os
import re
import json
from turtle import color
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from . import helpers as h

def sum_strategies(imputations_order: List[str]) -> int:
    strategies = []
    for imp in imputations_order:
        strategies.append(h.strategies_order_per_imputation(imp))
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
        len(h.strategies_order_per_imputation(imp))
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
        strategies_order = h.strategies_order_per_imputation(imp)
        df_imp = rank_df[rank_df['imputation'] == imp].pivot(
            index="sample", columns="strategy", values="rank"
        )
        df_imp = df_imp.reindex(columns=strategies_order)
        df_imp = df_imp.dropna(axis=1, how="all")
        sns.heatmap(
            df_imp,
            mask=df_imp.isna(),
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
    h.plot_save_fig(path_complete, fig)

