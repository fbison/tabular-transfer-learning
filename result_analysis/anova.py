import os
import re
from typing import List
from scipy import stats

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from . import helpers as h
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

def _run_three_way_anova(
    df,
    dv,
    factor1,
    factor2,
    factor3,
    min_rows=10,
):
    """
    Executa ANOVA 3-way (tipo II) e retorna a tabela com eta² e partial eta².
    """
    sub = df.dropna(subset=[dv, factor1, factor2, factor3]).copy()

    if sub.shape[0] < min_rows:
        return None

    sub[factor1] = sub[factor1].astype("category")
    sub[factor2] = sub[factor2].astype("category")
    sub[factor3] = sub[factor3].astype("category")

    formula = f"{dv} ~ C({factor1}) * C({factor2}) * C({factor3})"
    model = smf.ols(formula, data=sub).fit()
    aov = sm.stats.anova_lm(model, typ=2)

    # --- Effect sizes ---
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

def anova_three_way_all(
    df,
    dv="rmse",
    out_dir=None,
    min_rows=10,
):
    aov = _run_three_way_anova(
        df=df,
        dv=dv,
        factor1="strategy",
        factor2="imputation",
        factor3="upstream",
        min_rows=min_rows,
    )

    if aov is None:
        return None

    if out_dir:
        aov.to_csv(
            os.path.join(out_dir, f"anova_{dv}_3way_strategy_imputation_upstream.csv")
        )

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
    h.ensure_outdir(out_dir)

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

    anova_three_way_all(
        df=filtered_df,
        dv=dv,
        out_dir=out_dir,
    )

