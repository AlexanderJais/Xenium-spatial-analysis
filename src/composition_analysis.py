"""
composition_analysis.py
-----------------------
Cell type composition testing for multi-replicate Xenium studies.

Primary method: scCODA (Büttner et al. 2021, Nat Commun 12:6876)
  - Bayesian Dirichlet-multinomial model
  - Valid at n=4 per condition (no normal-distribution assumption)
  - Credible intervals for fold-changes (not inflated p-values)
  - Uses pertpy.tl.Sccoda; falls back to CLR + Welch t-test if unavailable

Fallback method: CLR + Welch t-test (Aitchison 1986)
  - Centred log-ratio transforms compositional data to unconstrained space
  - BH FDR correction applied across all cell types
  - Reports credible fold-changes from CLR space

Usage
-----
    from src.composition_analysis import run_sccoda

    results = run_sccoda(
        adata,
        cell_type_key = "cell_type",
        condition_key = "condition",
        replicate_key = "slide_id",
        reference_cell_type = "auto",   # scCODA reference; "auto" picks the largest
        output_dir = Path("figures_output"),
    )
"""

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Public API
# ===========================================================================

def run_sccoda(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    replicate_key: str = "slide_id",
    reference_cell_type: str = "auto",
    fdr_target: float = 0.05,
    n_mcmc_samples: int = 20_000,
    n_burnin: int = 5_000,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Test cell type composition differences between conditions.

    Tries scCODA via pertpy first; falls back to CLR + Welch t-test.

    Parameters
    ----------
    adata:
        AnnData with cell_type_key, condition_key, replicate_key in .obs.
    cell_type_key:
        obs column with cell type labels.
    condition_key:
        obs column with condition labels (must have exactly 2 levels).
    replicate_key:
        obs column identifying biological replicates (e.g. 'slide_id').
    reference_cell_type:
        scCODA reference cell type. 'auto' selects the most abundant type.
        Choosing a stable (housekeeping) cell type is recommended when known.
    fdr_target:
        FDR threshold for the CLR+t-test fallback (scCODA uses credible
        intervals natively and does not threshold on p-values).
    n_mcmc_samples:
        MCMC samples for scCODA (default 20 000; increase to 50 000 for
        publication-quality convergence diagnostics).
    n_burnin:
        MCMC burn-in samples discarded before inference.
    output_dir:
        If provided, saves 'composition_results.csv' here.

    Returns
    -------
    DataFrame with columns:
        cell_type, log2fc, credible_interval_lo, credible_interval_hi,
        significant, method, [pval, pval_adj for CLR fallback]
    Sorted by |log2fc| descending.
    """
    if cell_type_key not in adata.obs.columns:
        raise KeyError(
            f"cell_type_key='{cell_type_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    comp_df = _build_composition_table(adata, cell_type_key, replicate_key, condition_key)

    if comp_df.empty:
        logger.warning("run_sccoda: composition table is empty — returning empty DataFrame.")
        return pd.DataFrame()

    # Auto-select reference: largest cell type by total count
    if reference_cell_type == "auto":
        cell_type_totals = comp_df.drop(columns=[condition_key, "n_cells"], errors="ignore").sum()
        reference_cell_type = cell_type_totals.idxmax()
        logger.info(
            "scCODA reference cell type auto-selected: '%s' (largest by total count)",
            reference_cell_type,
        )

    # Try scCODA via pertpy
    try:
        results = _run_sccoda_pertpy(
            comp_df,
            condition_key=condition_key,
            reference_cell_type=reference_cell_type,
            n_mcmc_samples=n_mcmc_samples,
            n_burnin=n_burnin,
        )
        results["method"] = "scCODA"
        logger.info(
            "scCODA complete: %d cell types tested, %d significant",
            len(results), results["significant"].sum(),
        )
    except Exception as exc:
        logger.warning(
            "scCODA (pertpy) failed: %s — falling back to CLR + Welch t-test.", exc
        )
        results = _run_clr_ttest(comp_df, condition_key=condition_key, fdr_target=fdr_target)
        results["method"] = "CLR_ttest"
        logger.info(
            "CLR+t-test complete: %d cell types tested, %d significant (FDR < %.2f)",
            len(results), results["significant"].sum(), fdr_target,
        )

    results = results.sort_values("log2fc", key=np.abs, ascending=False).reset_index(drop=True)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / "composition_results.csv", index=False)
        logger.info("Composition results saved to %s/composition_results.csv", out)

    return results


# ===========================================================================
# Composition table builder
# ===========================================================================

def _build_composition_table(
    adata: ad.AnnData,
    cell_type_key: str,
    replicate_key: str,
    condition_key: str,
) -> pd.DataFrame:
    """
    Build a (n_replicates, n_cell_types) count table.

    Each row is one biological replicate (slide). Columns are cell types.
    An extra 'condition' column identifies the group.

    Parameters
    ----------
    adata:
        Full AnnData (all cells).

    Returns
    -------
    DataFrame: index = replicate_id, columns = [cell_type_1, ..., condition, n_cells]
    """
    obs = adata.obs[[cell_type_key, replicate_key, condition_key]].copy()
    obs[cell_type_key] = obs[cell_type_key].astype(str)

    counts = (
        obs.groupby([replicate_key, cell_type_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    # Attach condition (one per replicate by definition)
    cond_of_rep = obs.groupby(replicate_key, observed=True)[condition_key].first()
    counts[condition_key] = cond_of_rep

    # Attach total cells for cell-count-aware modelling
    n_cells = obs.groupby(replicate_key, observed=True).size()
    counts["n_cells"] = n_cells

    logger.info(
        "Composition table: %d replicates × %d cell types",
        len(counts), counts.shape[1] - 2,  # subtract condition + n_cells
    )
    return counts


# ===========================================================================
# scCODA via pertpy
# ===========================================================================

def _run_sccoda_pertpy(
    comp_df: pd.DataFrame,
    condition_key: str,
    reference_cell_type: str,
    n_mcmc_samples: int = 20_000,
    n_burnin: int = 5_000,
) -> pd.DataFrame:
    """
    Run scCODA using the pertpy interface.

    Builds a sample-level AnnData (replicates × cell types) that matches the
    pertpy scCODA API for pre-aggregated count tables (type="sample_level").

    Input: comp_df with rows = biological replicates, cols = cell types + condition.
    API reference: pertpy >= 0.7, Büttner et al. 2021 Nat Commun 12:6876.

    Raises ImportError if pertpy is not installed.
    """
    import pertpy as pt

    cell_type_cols = [c for c in comp_df.columns if c not in (condition_key, "n_cells")]

    # Build sample-level AnnData:
    #   .X  = integer count matrix (n_samples × n_cell_types)
    #   .obs = sample metadata with condition column
    #   .var = cell type names as index
    X = comp_df[cell_type_cols].values.astype(float)
    obs = comp_df[[condition_key]].copy()
    obs.index = obs.index.astype(str)
    var = pd.DataFrame(index=cell_type_cols)

    sample_adata = ad.AnnData(X=X, obs=obs, var=var)

    sccoda = pt.tl.Sccoda()

    # type="sample_level" is the correct API for pre-aggregated count tables
    # where each row is already one biological sample (replicate), not a single
    # cell.  type="cell_level" expects raw cell-level data and would need
    # generate_sample_level=True to aggregate — incompatible with our input.
    try:
        mdata = sccoda.load(
            sample_adata,
            type="sample_level",
            generate_sample_level=False,
            covariate_obs_key=condition_key,
        )
    except TypeError:
        # Older pertpy versions may use a different parameter set
        mdata = sccoda.load(
            sample_adata,
            covariate_obs_key=condition_key,
        )

    # Set reference cell type for model identifiability
    # Formula uses patsy C() notation to treat the condition as a categorical
    try:
        mdata = sccoda.prepare(
            mdata,
            formula=f"C({condition_key})",
            reference_cell_type=reference_cell_type,
        )
    except TypeError:
        # Some versions accept 'reference' instead of 'reference_cell_type'
        mdata = sccoda.prepare(
            mdata,
            formula=f"C({condition_key})",
            reference=reference_cell_type,
        )

    # MCMC sampling via NUTS (No-U-Turn Sampler)
    try:
        sccoda.run_nuts(
            mdata,
            num_samples=n_mcmc_samples,
            num_warmup=n_burnin,
            rng_key=42,
        )
    except TypeError:
        # Older pertpy may use random_seed instead of rng_key
        sccoda.run_nuts(
            mdata,
            num_samples=n_mcmc_samples,
            num_warmup=n_burnin,
        )

    # Extract results
    try:
        results_raw = sccoda.get_results(mdata)
    except AttributeError:
        # Some pertpy versions use credible_effects()
        results_raw = sccoda.credible_effects(mdata)

    if results_raw is None or (hasattr(results_raw, "empty") and results_raw.empty):
        raise RuntimeError("scCODA returned empty results — check input data.")

    # Tidy into canonical format
    return _tidy_sccoda_results(results_raw, reference_cell_type)


def _tidy_sccoda_results(results_raw: pd.DataFrame, reference_cell_type: str) -> pd.DataFrame:
    """
    Normalise scCODA output to the canonical composition result schema.

    scCODA returns a DataFrame with cell type as index and columns that vary
    by pertpy version. We map to:
      cell_type, log2fc, credible_interval_lo, credible_interval_hi, significant
    """
    df = results_raw.copy().reset_index()

    # Column name mapping (pertpy versions differ)
    rename = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "_")
        if "cell_type" in cl or col == "index":
            rename[col] = "cell_type"
        elif cl in ("effect", "final_parameter", "log2fc", "log2foldchange"):
            rename[col] = "log2fc"
        elif "lower" in cl or "lo_" in cl or "2.5" in cl:
            rename[col] = "credible_interval_lo"
        elif "upper" in cl or "hi_" in cl or "97.5" in cl:
            rename[col] = "credible_interval_hi"
        elif "significant" in cl or "is_credible" in cl:
            rename[col] = "significant"
    df = df.rename(columns=rename)

    # Ensure required columns exist with defaults
    if "cell_type" not in df.columns:
        df["cell_type"] = [str(i) for i in range(len(df))]
    if "log2fc" not in df.columns:
        # Try computing from credible intervals midpoint if available
        if "credible_interval_lo" in df.columns and "credible_interval_hi" in df.columns:
            df["log2fc"] = (df["credible_interval_lo"] + df["credible_interval_hi"]) / 2
        else:
            df["log2fc"] = np.nan
    if "credible_interval_lo" not in df.columns:
        df["credible_interval_lo"] = np.nan
    if "credible_interval_hi" not in df.columns:
        df["credible_interval_hi"] = np.nan
    if "significant" not in df.columns:
        # A cell type is credibly changed if 0 is outside the credible interval
        df["significant"] = (
            (df["credible_interval_lo"] > 0) | (df["credible_interval_hi"] < 0)
        )

    df["significant"] = df["significant"].astype(bool)

    # Drop the reference cell type row (its effect is always 0 by definition)
    df = df[df["cell_type"] != reference_cell_type].copy()

    return df[["cell_type", "log2fc", "credible_interval_lo", "credible_interval_hi", "significant"]]


# ===========================================================================
# CLR + Welch t-test fallback
# ===========================================================================

def _bh_correct(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. scipy-version-independent."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    order = np.argsort(pvalues)
    pv_sort = np.asarray(pvalues)[order]
    adjusted = np.empty(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        cummin = min(cummin, pv_sort[i] * n / (i + 1))
        adjusted[order[i]] = cummin
    return np.clip(adjusted, 0.0, 1.0)


def _run_clr_ttest(
    comp_df: pd.DataFrame,
    condition_key: str,
    fdr_target: float = 0.05,
) -> pd.DataFrame:
    """
    Centred log-ratio (CLR) transform + Welch t-test for compositional data.

    CLR maps simplex data to unconstrained Euclidean space:
        clr(x_i) = log(x_i / geometric_mean(x))
    This allows standard parametric tests while respecting the compositional
    structure (Aitchison 1986; Gloor et al. 2017 Front Microbiol).

    A small pseudo-count of 0.5 is added before CLR to handle zero cells
    in a replicate (recommended: Martín-Fernández et al. 2003).
    """
    from scipy import stats

    cell_type_cols = [c for c in comp_df.columns if c not in (condition_key, "n_cells")]
    conditions = sorted(comp_df[condition_key].unique())

    if len(conditions) != 2:
        raise ValueError(
            f"CLR t-test requires exactly 2 conditions; found {conditions}."
        )
    cond_a, cond_b = conditions

    # Convert to proportions and apply CLR transform
    counts = comp_df[cell_type_cols].values.astype(float) + 0.5
    props = counts / counts.sum(axis=1, keepdims=True)
    geom_means = np.exp(np.log(props).mean(axis=1, keepdims=True))
    clr = np.log(props / geom_means)

    clr_df = pd.DataFrame(clr, index=comp_df.index, columns=cell_type_cols)
    clr_df[condition_key] = comp_df[condition_key].values

    rows = []
    for ct in cell_type_cols:
        a_vals = clr_df.loc[clr_df[condition_key] == cond_a, ct].values
        b_vals = clr_df.loc[clr_df[condition_key] == cond_b, ct].values

        if len(a_vals) < 2 or len(b_vals) < 2:
            logger.warning(
                "CLR t-test: skipping '%s' — fewer than 2 replicates per condition.", ct
            )
            continue

        t_stat, pval = stats.ttest_ind(b_vals, a_vals, equal_var=False)

        # log2FC from CLR space (CLR values are in natural log; divide by log(2))
        log2fc = (b_vals.mean() - a_vals.mean()) / np.log(2)

        # 95% CI via t-distribution
        n_a, n_b = len(a_vals), len(b_vals)
        se = np.sqrt(a_vals.var(ddof=1) / n_a + b_vals.var(ddof=1) / n_b)
        df_welch = (
            (a_vals.var(ddof=1) / n_a + b_vals.var(ddof=1) / n_b) ** 2
            / (
                (a_vals.var(ddof=1) / n_a) ** 2 / (n_a - 1)
                + (b_vals.var(ddof=1) / n_b) ** 2 / (n_b - 1)
            )
        )
        t_crit = stats.t.ppf(0.975, df_welch)
        ci_lo = (b_vals.mean() - a_vals.mean() - t_crit * se) / np.log(2)
        ci_hi = (b_vals.mean() - a_vals.mean() + t_crit * se) / np.log(2)

        rows.append({
            "cell_type": ct,
            "log2fc": log2fc,
            "credible_interval_lo": ci_lo,
            "credible_interval_hi": ci_hi,
            "pval": pval,
        })

    if not rows:
        logger.warning(
            "CLR t-test: no cell types had sufficient replicates (need >= 2 per condition). "
            "Returning empty DataFrame."
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["pval_adj"] = _bh_correct(df["pval"].values)
    df["significant"] = df["pval_adj"] < fdr_target

    return df[["cell_type", "log2fc", "credible_interval_lo", "credible_interval_hi",
               "significant", "pval", "pval_adj"]]
