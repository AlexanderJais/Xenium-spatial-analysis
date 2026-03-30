"""
dge_analysis.py
---------------
Differential gene expression between two conditions.

Supports:
  * PyDESeq2   (pseudobulk, recommended for Xenium)
  * Wilcoxon   (cell-level, scanpy)
  * t-test     (cell-level, scanpy)

For PyDESeq2, cells are aggregated ("pseudobulk") per donor or per
spatial region. If no donor / replicate column exists, a simple
per-condition aggregate is used with a bootstrap-based variance
estimate.
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ===========================================================================
# PyDESeq2 pseudobulk DGE
# ===========================================================================

def pseudobulk_deseq2(
    adata: ad.AnnData,
    condition_key: str = "condition",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    replicate_key: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    cell_type: Optional[str] = None,
    min_cells: int = 5,
    min_counts: int = 10,
    n_bootstrap_replicates: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run PyDESeq2 on pseudobulk aggregates.

    If ``replicate_key`` is provided, cells are aggregated per
    (condition, replicate). Otherwise, cells are randomly split into
    ``n_bootstrap_replicates`` pseudo-replicates per condition to enable
    within-condition variance estimation.

    Parameters
    ----------
    adata:
        Full AnnData; raw counts must be in ``.layers['counts']``.
    condition_key:
        obs column with condition labels.
    condition_a / condition_b:
        The two conditions to compare. Inferred if None.
    replicate_key:
        Optional obs column with biological replicate labels.
    cell_type_key / cell_type:
        If provided, subset adata to cells of this type before DGE.
    min_cells:
        Pseudobulk samples with fewer cells are excluded.
    min_counts:
        Genes with summed counts < min_counts across all samples dropped.
    n_bootstrap_replicates:
        Number of pseudo-replicates when no replicate_key is given.
    random_state:
        RNG seed for pseudo-replicate creation.

    Returns
    -------
    DataFrame with columns:
        gene, log2FoldChange, lfcSE, stat, pvalue, padj, baseMean
    Sorted by adjusted p-value.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError(
            "pydeseq2 is required for pseudobulk DGE. "
            "Install with: pip install pydeseq2\n"
            "Alternatively, switch to dge_method='stringent_wilcoxon' in "
            "PipelineConfig, which requires no extra dependencies."
        )

    adata = _subset_cell_type(adata, cell_type_key, cell_type)
    cond_a, cond_b = _resolve_conditions(adata, condition_key, condition_a, condition_b)

    # Use raw counts
    X_counts = _get_counts(adata)

    if replicate_key is not None:
        bulk_df, sample_meta = _aggregate_by_replicate(
            X_counts, adata.obs, condition_key, replicate_key, adata.var_names
        )
    else:
        logger.warning(
            "PyDESeq2 is running WITHOUT real biological replicates. "
            "Pseudo-replicates are created by randomly splitting cells, which "
            "violates DESeq2 assumptions and inflates statistical power. "
            "Results should NOT be used for publication without real biological "
            "replication. Pass replicate_key='slide_id' if multiple slides exist "
            "per condition."
        )
        logger.info(
            "No replicate_key; creating %d pseudo-replicates per condition.",
            n_bootstrap_replicates,
        )
        bulk_df, sample_meta = _split_pseudobulk(
            X_counts,
            adata.obs,
            condition_key,
            adata.var_names,
            n_reps=n_bootstrap_replicates,
            random_state=random_state,
        )

    # Filter low-count samples
    sample_meta = sample_meta[sample_meta["n_cells"] >= min_cells]
    bulk_df = bulk_df.loc[sample_meta.index]

    # Filter low-count genes
    gene_mask = bulk_df.sum(axis=0) >= min_counts
    bulk_df = bulk_df.loc[:, gene_mask]

    # Exclude zero-filled custom genes (present only in a subset of slides).
    # These have structural zeros in slides that did not probe the gene, making
    # them appear differentially expressed regardless of biology. DESeq2 cannot
    # distinguish structural zeros from biological absence.
    if "zero_filled" in adata.var.columns:
        valid_genes = bulk_df.columns.intersection(
            adata.var.index[~adata.var["zero_filled"].fillna(False)]
        )
        n_excluded = len(bulk_df.columns) - len(valid_genes)
        if n_excluded > 0:
            logger.warning(
                "DESeq2: excluding %d zero-filled custom gene(s) from testing "
                "(present only in a subset of slides; structural zeros would "
                "produce spurious fold changes).",
                n_excluded,
            )
        bulk_df = bulk_df[valid_genes]

    logger.info(
        "PyDESeq2 input: %d samples x %d genes", bulk_df.shape[0], bulk_df.shape[1]
    )

    # Build DeseqDataSet
    dds = DeseqDataSet(
        counts=bulk_df,
        metadata=sample_meta,
        design_factors=condition_key,
        ref_level=[condition_key, cond_a],
        refit_cooks=True,
        quiet=True,
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=[condition_key, cond_b, cond_a], quiet=True)
    stat_res.summary()

    # lfc_shrink() requires an exact coefficient name from dds.coefficients.
    # PyDESeq2 normalises strings (spaces -> underscores, special chars stripped),
    # so we must look it up rather than construct it by hand.
    available_coeffs = list(dds.varm.keys()) if hasattr(dds, "varm") else []
    # PyDESeq2 >= 0.4 stores coefficients in dds.LFC.columns
    try:
        available_coeffs = list(dds.LFC.columns)
    except AttributeError:
        pass
    expected_coeff = f"{condition_key}_{cond_b}_vs_{cond_a}"
    # Find the best match (exact, then case-insensitive, then skip shrinkage)
    matched_coeff = None
    for c in available_coeffs:
        if c == expected_coeff:
            matched_coeff = c
            break
    if matched_coeff is None:
        for c in available_coeffs:
            if c.lower() == expected_coeff.lower():
                matched_coeff = c
                break
    if matched_coeff is not None:
        stat_res.lfc_shrink(coeff=matched_coeff)
        logger.info("LFC shrinkage applied with coeff: %s", matched_coeff)
    else:
        logger.warning(
            "lfc_shrink skipped: coefficient '%s' not found in dds. "
            "Available: %s. Results use unshrunken LFC.",
            expected_coeff, available_coeffs,
        )

    results = stat_res.results_df.reset_index().sort_values("padj")
    # reset_index() names the column after the index name; normalise to "gene"
    if "gene" not in results.columns:
        results = results.rename(columns={results.columns[0]: "gene"})
    logger.info(
        "DESeq2 done: %d significant genes (padj < 0.05)",
        (results["padj"] < 0.05).sum(),
    )
    return results


# ===========================================================================
# Scanpy Wilcoxon / t-test DGE
# ===========================================================================

def scanpy_dge(
    adata: ad.AnnData,
    condition_key: str = "condition",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    cell_type: Optional[str] = None,
    method: str = "wilcoxon",
) -> pd.DataFrame:
    """
    Cell-level DGE using scanpy rank_genes_groups.

    Compares condition_b (reference = condition_a) across all cells or
    within a specific cell type. Uses log-normalised values.

    Parameters
    ----------
    adata:
        Preprocessed AnnData; ``.layers['lognorm']`` must exist.
    condition_key:
        obs column with condition labels.
    condition_a:
        Reference condition (denominator). Inferred if None.
    condition_b:
        Test condition (numerator). Inferred if None.
    cell_type_key / cell_type:
        Restrict DGE to a single cell type.
    method:
        'wilcoxon' | 't-test' | 't-test_overestim_var'.

    Returns
    -------
    DataFrame with columns: gene, log2fc, pval, pval_adj, score,
    pct_A (fraction of cells in cond A expressing gene),
    pct_B (fraction in cond B).
    """
    import scanpy as sc

    adata = _subset_cell_type(adata, cell_type_key, cell_type)
    cond_a, cond_b = _resolve_conditions(adata, condition_key, condition_a, condition_b)

    # Exclude zero-filled custom genes — structural zeros from harmonisation
    # bias the Wilcoxon test toward calling absent genes as down-regulated.
    if "zero_filled" in adata.var.columns:
        keep = ~adata.var["zero_filled"].fillna(False)
        n_excluded = int((~keep).sum())
        if n_excluded > 0:
            logger.warning(
                "scanpy_dge: excluding %d zero-filled custom gene(s) from testing.",
                n_excluded,
            )
            adata = adata[:, keep].copy()

    # Use log-normalised values
    if "lognorm" in adata.layers:
        adata = adata.copy()
        adata.X = adata.layers["lognorm"]

    # Restrict to two conditions
    mask = adata.obs[condition_key].isin([cond_a, cond_b])
    adata = adata[mask].copy()

    sc.tl.rank_genes_groups(
        adata,
        groupby=condition_key,
        groups=[cond_b],
        reference=cond_a,
        method=method,
        use_raw=False,
        pts=True,
    )

    results = sc.get.rank_genes_groups_df(adata, group=cond_b)
    # scanpy already returns log2 fold changes (via np.log2 internally)
    results = results.rename(
        columns={
            "names"        : "gene",
            "logfoldchanges": "log2fc",
            "pvals"        : "pval",
            "pvals_adj"    : "pval_adj",
            "scores"       : "score",
        }
    )
    # Attach per-group expression fractions
    if "pts" in adata.uns["rank_genes_groups"]:
        pts_dict = adata.uns["rank_genes_groups"]["pts"]
        if cond_a in pts_dict:
            results["pct_A"] = results["gene"].map(pts_dict[cond_a])
        if cond_b in pts_dict:
            results["pct_B"] = results["gene"].map(pts_dict[cond_b])

    results = results.sort_values("pval_adj").reset_index(drop=True)
    logger.info(
        "%s DGE done: %d significant genes (adj-p < 0.05)",
        method, (results["pval_adj"] < 0.05).sum(),
    )
    return results


# ===========================================================================
# Dispatcher
# ===========================================================================

def stringent_wilcoxon_dge(
    adata: ad.AnnData,
    condition_key: str = "condition",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    cell_type: Optional[str] = None,
    # Stringency parameters — all tightened vs plain Wilcoxon
    lfc_threshold: float = 1.0,          # 2-fold change minimum (vs 0.5 default)
    pval_threshold: float = 0.01,         # tighter p-value (vs 0.05)
    min_pct: float = 0.10,               # gene must be expressed in >=10% of cells
    min_mean_expr: float = 0.1,          # minimum mean log-norm expression
    min_consistent_replicates: int = 3,   # must be same direction in >=3/4 slides
    replicate_key: str = "slide_id",
    **kwargs,
) -> pd.DataFrame:
    """
    Wilcoxon rank-sum DGE with multiple stringency filters applied post-hoc.

    STATISTICAL CAVEAT — PSEUDOREPLICATION:
    The Wilcoxon rank-sum test treats every cell as an independent observation,
    but cells on the same tissue section are spatially correlated. With N_cells
    per condition the test uses N_cells degrees of freedom, whereas the true
    biological unit is the slide (animal). This inflates z-scores and
    p-values by up to 10–50× for spatially autocorrelated genes (Crowell et al.
    2020 Nature Communications; Squair et al. 2021 Nature Communications).

    For publication-quality DGE, use ``method='pydeseq2'``, which aggregates
    cells to pseudobulk per replicate before testing and correctly uses N=4
    biological replicates as degrees of freedom.

    This function partially mitigates pseudoreplication via:
      1. Raising the log2FC threshold to 1.0 (2-fold change)
      2. Tightening the adjusted p-value threshold to 0.01
      3. Requiring the gene to be expressed in >=10% of cells per condition
      4. Requiring minimum mean log-normalised expression
      5. Requiring consistent direction in >=3 out of 4 biological replicates

    Filter 5 (replicate consistency) is the most important guard: it verifies
    the effect replicates across independent animals. It is applied BEFORE the
    p-value filter to avoid pre-filtering on inflated p-values that bias the
    gene candidate set entering the consistency check.

    Parameters
    ----------
    lfc_threshold:
        Minimum absolute log2 fold change (default 1.0 = 2-fold).
    pval_threshold:
        Adjusted p-value threshold (default 0.01).
    min_pct:
        Minimum fraction of cells expressing the gene in at least one condition.
    min_mean_expr:
        Minimum mean log-normalised expression in at least one condition.
    min_consistent_replicates:
        Number of replicates (slides) that must show the same direction of
        change. Default 3 out of 4 (75% consistency).
    replicate_key:
        obs column identifying biological replicates (default: slide_id).
    """
    # Exclude zero-filled custom genes before any testing or filter computation.
    # Structural zeros from panel harmonisation inflate zero counts for slides
    # that did not probe the gene, biasing per-replicate means and pct filters.
    if "zero_filled" in adata.var.columns:
        keep_genes = ~adata.var["zero_filled"].fillna(False)
        n_excluded = int((~keep_genes).sum())
        if n_excluded > 0:
            logger.warning(
                "stringent_wilcoxon_dge: excluding %d zero-filled custom gene(s) "
                "from all filters.",
                n_excluded,
            )
            adata = adata[:, keep_genes].copy()

    # Step 1: Run standard Wilcoxon
    results = scanpy_dge(
        adata,
        condition_key=condition_key,
        condition_a=condition_a,
        condition_b=condition_b,
        cell_type_key=cell_type_key,
        cell_type=cell_type,
        method="wilcoxon",
        **kwargs,
    )

    if results.empty:
        return results

    cond_a, cond_b = _resolve_conditions(adata, condition_key, condition_a, condition_b)
    adata = _subset_cell_type(adata, cell_type_key, cell_type)

    # Standardise column names
    lfc_col = "log2FoldChange" if "log2FoldChange" in results.columns else "log2fc"
    p_col   = "padj" if "padj" in results.columns else "pval_adj"
    g_col   = "gene" if "gene" in results.columns else results.columns[0]

    n_before = len(results)

    # ── Filter 1: log2FC threshold ────────────────────────────────────────────
    results = results[results[lfc_col].abs() >= lfc_threshold]
    logger.info("Stringent Wilcoxon: %d genes after |log2FC| >= %.1f",
                len(results), lfc_threshold)

    if results.empty:
        logger.warning("Stringent Wilcoxon: no genes survived LFC filter.")
        return results

    # ── Filter 2: replicate consistency (BEFORE p-value filter) ──────────────
    # Applied before p-value filtering to avoid using inflated cell-level
    # p-values (which are anti-conservative due to spatial autocorrelation and
    # pseudoreplication) to pre-select the gene candidate set. Replicating the
    # effect across independent animals is a stronger biological criterion than
    # a p-value derived from N_cells degrees of freedom.
    X = adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    if replicate_key in adata.obs.columns:
        replicates_a = adata.obs.loc[
            adata.obs[condition_key] == cond_a, replicate_key
        ].unique()
        replicates_b = adata.obs.loc[
            adata.obs[condition_key] == cond_b, replicate_key
        ].unique()

        # Warn early if there are not enough replicates for the filter to pass
        # any gene at all — otherwise the pipeline silently returns zero results.
        if len(replicates_b) < min_consistent_replicates or \
                len(replicates_a) < min_consistent_replicates:
            logger.warning(
                "Stringent Wilcoxon: replicate consistency filter requires "
                ">= %d replicates per condition, but found %d for '%s' and %d "
                "for '%s'. ALL genes will fail this filter and the result will "
                "be empty. Either reduce min_consistent_replicates or use a "
                "DGE method that does not require replication (e.g. 'wilcoxon').",
                min_consistent_replicates,
                len(replicates_b), cond_b,
                len(replicates_a), cond_a,
            )

        # Compute per-replicate mean expression for each gene
        consistent_genes = []
        for gene in results[g_col]:
            if gene not in var_idx:
                continue  # gene was filtered out — do not falsely count as consistent
            gi   = var_idx[gene]
            expr = X[:, gi]
            if sp.issparse(expr):
                expr = np.array(expr.todense()).ravel()
            else:
                expr = np.array(expr).ravel()

            # Global direction
            mask_a = adata.obs[condition_key] == cond_a
            mask_b = adata.obs[condition_key] == cond_b
            global_lfc = expr[mask_b].mean() - expr[mask_a].mean()
            global_sign = np.sign(global_lfc)

            grand_mean_a = expr[mask_a].mean()
            grand_mean_b = expr[mask_b].mean()

            # cond_b replicates vs cond_a grand mean
            n_consistent_b = 0
            for rep_b in replicates_b:
                mask_rep = (
                    (adata.obs[condition_key] == cond_b) &
                    (adata.obs[replicate_key] == rep_b)
                )
                rep_mean = expr[mask_rep].mean()
                if np.sign(rep_mean - grand_mean_a) == global_sign:
                    n_consistent_b += 1

            # cond_a replicates vs cond_b grand mean (symmetric check)
            n_consistent_a = 0
            for rep_a in replicates_a:
                mask_rep = (
                    (adata.obs[condition_key] == cond_a) &
                    (adata.obs[replicate_key] == rep_a)
                )
                rep_mean = expr[mask_rep].mean()
                # For cond_a, consistent means the replicate is on the
                # opposite side of grand_mean_b (i.e. same global direction)
                if np.sign(grand_mean_b - rep_mean) == global_sign:
                    n_consistent_a += 1

            if (n_consistent_b >= min_consistent_replicates and
                    n_consistent_a >= min_consistent_replicates):
                consistent_genes.append(gene)

        results = results[results[g_col].isin(consistent_genes)]
        logger.info(
            "Stringent Wilcoxon: %d genes after replicate consistency "
            "(>= %d/%d replicates same direction in both conditions)",
            len(results), min_consistent_replicates, max(len(replicates_b), len(replicates_a)),
        )
    else:
        logger.warning(
            "Stringent Wilcoxon: replicate_key '%s' not in obs — "
            "skipping consistency filter.", replicate_key,
        )

    if results.empty:
        logger.warning("Stringent Wilcoxon: no genes survived replicate consistency filter.")
        return results

    # ── Filter 3: adjusted p-value ────────────────────────────────────────────
    results = results[results[p_col] < pval_threshold]
    logger.info("Stringent Wilcoxon: %d genes after adj.p < %.3f",
                len(results), pval_threshold)

    if results.empty:
        logger.warning("Stringent Wilcoxon: no genes survived p-value filter.")
        return results

    # ── Filter 4: minimum expression fraction ─────────────────────────────────
    if "pct_A" in results.columns and "pct_B" in results.columns:
        pct_mask = (results["pct_A"] >= min_pct) | (results["pct_B"] >= min_pct)
        results  = results[pct_mask]
        logger.info("Stringent Wilcoxon: %d genes after pct >= %.0f%%",
                    len(results), min_pct * 100)

    # ── Filter 5: minimum mean expression ─────────────────────────────────────
    keep = []
    for gene in results[g_col]:
        if gene not in var_idx:
            keep.append(False)  # gene was filtered out — do not falsely pass
            continue
        gi   = var_idx[gene]
        expr = X[:, gi]
        if sp.issparse(expr):
            expr = np.array(expr.todense()).ravel()
        else:
            expr = np.array(expr).ravel()
        mask_a = adata.obs[condition_key] == cond_a
        mask_b = adata.obs[condition_key] == cond_b
        mean_a = expr[mask_a].mean()
        mean_b = expr[mask_b].mean()
        keep.append(max(mean_a, mean_b) >= min_mean_expr)
    results = results[keep]
    logger.info("Stringent Wilcoxon: %d genes after mean expr >= %.2f",
                len(results), min_mean_expr)

    logger.info(
        "Stringent Wilcoxon: %d / %d genes passed all filters",
        len(results), n_before,
    )
    results["method"] = "stringent_wilcoxon"
    return results


def run_dge(
    adata: ad.AnnData,
    method: str = "pydeseq2",
    condition_key: str = "condition",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    cell_type: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Unified DGE dispatcher.

    Parameters
    ----------
    method:
        'pydeseq2' | 'wilcoxon' | 't-test'
    All other parameters forwarded to the respective function.

    Returns
    -------
    Standardised results DataFrame (always contains 'gene', 'log2fc' or
    'log2FoldChange', 'pval_adj' or 'padj' columns).
    """
    # Separate shared kwargs (replicate_key used by both pydeseq2 and
    # stringent_wilcoxon) from stringent-only kwargs (which would crash
    # scanpy_dge or pseudobulk_deseq2 if forwarded).
    _replicate_key = kwargs.pop("replicate_key", "slide_id")
    _sw_defaults = {
        "lfc_threshold"            : 1.0,
        "pval_threshold"           : 0.01,
        "min_pct"                  : 0.10,
        "min_mean_expr"            : 0.1,
        "min_consistent_replicates": 3,
    }
    _sw_kwargs = {k: kwargs.pop(k, v) for k, v in _sw_defaults.items()}
    _sw_kwargs["replicate_key"] = _replicate_key

    # Warn if the caller passed stringent_wilcoxon-specific kwargs but chose a
    # different method -- those kwargs are not forwarded and would be silently
    # ignored otherwise.
    if method != "stringent_wilcoxon":
        non_default = [
            k for k, v in _sw_defaults.items()
            if _sw_kwargs[k] != v
        ]
        if non_default:
            logger.warning(
                "run_dge: the following stringent_wilcoxon kwargs were supplied "
                "but will be ignored because method='%s': %s",
                method, non_default,
            )

    if method == "pydeseq2":
        result = pseudobulk_deseq2(
            adata,
            condition_key=condition_key,
            condition_a=condition_a,
            condition_b=condition_b,
            cell_type_key=cell_type_key,
            cell_type=cell_type,
            replicate_key=_replicate_key,  # forward to pydeseq2 correctly
            **kwargs,
        )
    elif method in {"wilcoxon", "t-test", "t-test_overestim_var"}:
        result = scanpy_dge(
            adata,
            condition_key=condition_key,
            condition_a=condition_a,
            condition_b=condition_b,
            cell_type_key=cell_type_key,
            cell_type=cell_type,
            method=method,
            **kwargs,
        )
    elif method == "stringent_wilcoxon":
        result = stringent_wilcoxon_dge(
            adata,
            condition_key=condition_key,
            condition_a=condition_a,
            condition_b=condition_b,
            cell_type_key=cell_type_key,
            cell_type=cell_type,
            **{**_sw_kwargs, **kwargs},   # merge back any remaining kwargs
        )
    elif method == "cside":
        result = cside_pseudobulk_dge(
            adata,
            condition_key=condition_key,
            condition_a=condition_a,
            condition_b=condition_b,
            replicate_key=_replicate_key,
            cell_type_key=cell_type_key or "cell_type",
        )
    else:
        raise ValueError(
            f"Unknown DGE method: {method}. "
            "Choose pydeseq2, cside, wilcoxon, stringent_wilcoxon or t-test."
        )

    # Normalise column names to a single canonical schema so all downstream
    # code can rely on 'log2fc' and 'pval_adj' unconditionally.
    result = result.rename(columns={
        "log2FoldChange": "log2fc",
        "padj":           "pval_adj",
        "pvals_adj":      "pval_adj",
        "pvals":          "pval",
    })
    return result


# ===========================================================================
# Internal helpers
# ===========================================================================

def _get_counts(adata: ad.AnnData) -> np.ndarray:
    """Return dense integer count matrix.

    Prefers ``adata.layers['counts']`` (raw integer counts saved before
    normalisation by ``normalise_and_select_hvg``).  Falls back to ``.X``
    with a warning — .X may be log-normalised, which would corrupt pseudobulk
    aggregation.
    """
    if "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        logger.warning(
            "_get_counts: 'counts' layer not found in adata.layers. "
            "Falling back to .X, which may be log-normalised rather than raw "
            "counts. This will corrupt pseudobulk aggregation. "
            "Ensure normalise_and_select_hvg() ran before this call so that "
            "adata.layers['counts'] is populated."
        )
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.round(X).astype(int)


def _subset_cell_type(
    adata: ad.AnnData,
    cell_type_key: Optional[str],
    cell_type: Optional[str],
) -> ad.AnnData:
    if cell_type_key is not None and cell_type is not None:
        mask = adata.obs[cell_type_key] == cell_type
        adata = adata[mask].copy()
        logger.info("Subsetting to %s = %s: %d cells", cell_type_key, cell_type, adata.n_obs)
    return adata


def _resolve_conditions(
    adata: ad.AnnData,
    condition_key: str,
    cond_a: Optional[str],
    cond_b: Optional[str],
):
    conditions = adata.obs[condition_key].unique().tolist()
    if cond_a is None or cond_b is None:
        if len(conditions) != 2:
            raise ValueError(
                f"Expected exactly 2 conditions; found {conditions}. "
                "Please specify condition_a and condition_b explicitly."
            )
        cond_a, cond_b = sorted(conditions)
        logger.info("Auto-detected conditions: A=%s (ref), B=%s (test)", cond_a, cond_b)
    return cond_a, cond_b


def _aggregate_by_replicate(
    X: np.ndarray,
    obs: pd.DataFrame,
    condition_key: str,
    replicate_key: str,
    var_names,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum counts per (condition, replicate) group."""
    groups = obs.groupby([condition_key, replicate_key], observed=True)
    rows, meta_rows = [], []
    for (cond, rep), idx in groups.groups.items():
        indexer = obs.index.get_indexer(idx)
        indexer = indexer[indexer >= 0]  # drop unmatched (-1) indices
        if len(indexer) == 0:
            logger.warning(
                "Pseudobulk: no cells matched for group (%s, %s); skipping.",
                cond, rep,
            )
            continue
        counts = X[indexer].sum(axis=0)
        sample_id = f"{cond}__{rep}"
        rows.append(pd.Series(counts, index=var_names, name=sample_id))
        meta_rows.append({"sample": sample_id, condition_key: cond, "n_cells": len(idx)})
    bulk_df = pd.DataFrame(rows)
    sample_meta = pd.DataFrame(meta_rows).set_index("sample")
    return bulk_df, sample_meta


def _split_pseudobulk(
    X: np.ndarray,
    obs: pd.DataFrame,
    condition_key: str,
    var_names,
    n_reps: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split cells into n_reps pseudobulk samples per condition.

    Uses a random permutation followed by np.array_split (split without
    replacement, not bootstrapping).  Each cell appears in exactly one
    pseudo-replicate.
    """
    rng = np.random.default_rng(random_state)
    rows, meta_rows = [], []
    for cond in obs[condition_key].unique():
        idx = np.where(obs[condition_key] == cond)[0]
        splits = np.array_split(rng.permutation(idx), n_reps)
        for k, split in enumerate(splits):
            if len(split) == 0:
                continue
            counts = X[split].sum(axis=0)
            sample_id = f"{cond}__rep{k}"
            rows.append(pd.Series(counts, index=var_names, name=sample_id))
            meta_rows.append({
                "sample": sample_id,
                condition_key: cond,
                "n_cells": len(split),
            })
    bulk_df = pd.DataFrame(rows)
    sample_meta = pd.DataFrame(meta_rows).set_index("sample")
    return bulk_df, sample_meta


# ===========================================================================
# C-SIDE pseudobulk DGE (Cable et al. 2022, Nat Methods 19:1076)
# ===========================================================================

def cside_pseudobulk_dge(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    replicate_key: str = "slide_id",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    min_cells_per_sample: int = 5,
    min_total_counts: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    C-SIDE-inspired per-cell-type pseudobulk DGE for multi-replicate Xenium.

    For each cell type, cells are aggregated (summed) per biological replicate
    (slide) to form a pseudobulk count matrix, then PyDESeq2 is run using the
    real biological replicates as samples.

    This is the multi-replicate spatial DGE approach introduced by Cable et al.
    2022 (C-SIDE, Nat Methods 19:1076). For Xenium (single-cell resolution,
    no spot mixing), the implementation reduces to per-cell-type pseudobulk
    DESeq2, which matches the C-SIDE recommendation for datasets where cell
    type identities are known per observation.

    Parameters
    ----------
    adata:
        AnnData; raw counts must be in ``.layers['counts']``.
    cell_type_key:
        obs column with cell type labels.
    condition_key:
        obs column with condition labels.
    replicate_key:
        obs column with biological replicate labels (e.g. 'slide_id').
    condition_a / condition_b:
        The two conditions. Auto-inferred if None.
    min_cells_per_sample:
        Pseudobulk samples with fewer cells are excluded.
    min_total_counts:
        Genes with fewer summed counts across all samples are excluded.
    random_state:
        RNG seed (not used directly here but kept for API consistency).

    Returns
    -------
    Long-format DataFrame (same schema as run_cluster_dge output):
        group, gene, log2fc, pval_adj, baseMean, ...
    sorted by group then padj.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError(
            "pydeseq2 is required for C-SIDE pseudobulk DGE. "
            "Install with: pip install pydeseq2"
        )

    if cell_type_key not in adata.obs.columns:
        raise KeyError(f"cell_type_key='{cell_type_key}' not in adata.obs")
    if replicate_key not in adata.obs.columns:
        raise KeyError(
            f"replicate_key='{replicate_key}' not in adata.obs. "
            "C-SIDE requires real biological replicates."
        )

    cond_a, cond_b = _resolve_conditions(adata, condition_key, condition_a, condition_b)

    # Exclude zero-filled custom genes (structural zeros corrupt pseudobulk sums)
    if "zero_filled" in adata.var.columns:
        keep_genes = ~adata.var["zero_filled"].fillna(False)
        n_excl = int((~keep_genes).sum())
        if n_excl > 0:
            logger.info(
                "C-SIDE: excluding %d zero-filled custom gene(s) from testing.", n_excl
            )
            adata = adata[:, keep_genes].copy()

    X_counts = _get_counts(adata)
    cell_types = sorted(adata.obs[cell_type_key].astype(str).unique())

    all_results = []

    for ct in cell_types:
        ct_mask = adata.obs[cell_type_key].astype(str) == ct
        sub_obs = adata.obs[ct_mask]
        sub_X = X_counts[ct_mask]

        # Count cells per (condition, replicate)
        n_a = (sub_obs[condition_key] == cond_a).sum()
        n_b = (sub_obs[condition_key] == cond_b).sum()
        if n_a < min_cells_per_sample or n_b < min_cells_per_sample:
            logger.info(
                "C-SIDE: skipping cell type '%s' (n_%s=%d, n_%s=%d)",
                ct, cond_a, n_a, cond_b, n_b,
            )
            continue

        # Pseudobulk: sum counts per (condition, replicate)
        bulk_df, sample_meta = _aggregate_by_replicate(
            sub_X, sub_obs, condition_key, replicate_key, adata.var_names
        )

        # Filter low-count samples and genes
        sample_meta = sample_meta[sample_meta["n_cells"] >= min_cells_per_sample]
        bulk_df = bulk_df.loc[sample_meta.index]

        gene_mask = bulk_df.sum(axis=0) >= min_total_counts
        bulk_df = bulk_df.loc[:, gene_mask]

        if bulk_df.shape[0] < 4:
            logger.warning(
                "C-SIDE: cell type '%s' has only %d samples after filtering "
                "(need >= 4 for DESeq2). Skipping.",
                ct, bulk_df.shape[0],
            )
            continue

        if bulk_df.shape[1] == 0:
            logger.info("C-SIDE: cell type '%s' — no genes passed count filter.", ct)
            continue

        logger.info(
            "C-SIDE: cell type '%s': %d samples × %d genes",
            ct, bulk_df.shape[0], bulk_df.shape[1],
        )

        try:
            dds = DeseqDataSet(
                counts=bulk_df,
                metadata=sample_meta,
                design_factors=condition_key,
                ref_level=[condition_key, cond_a],
                refit_cooks=True,
                quiet=True,
            )
            dds.deseq2()

            stat_res = DeseqStats(dds, contrast=[condition_key, cond_b, cond_a], quiet=True)
            stat_res.summary()

            # LFC shrinkage
            try:
                avail = list(dds.LFC.columns)
            except AttributeError:
                avail = []
            expected = f"{condition_key}_{cond_b}_vs_{cond_a}"
            coeff = next((c for c in avail if c == expected), None) or \
                    next((c for c in avail if c.lower() == expected.lower()), None)
            if coeff:
                stat_res.lfc_shrink(coeff=coeff)

            res = (
                stat_res.results_df
                .reset_index()
                .rename(columns={"log2FoldChange": "log2fc", "padj": "pval_adj"})
                .sort_values("pval_adj")
            )
            # reset_index() names the column after the index name; normalise to "gene"
            if "gene" not in res.columns:
                res = res.rename(columns={res.columns[0]: "gene"})
            res.insert(0, "group", ct)
            res["method"] = "cside_pseudobulk"
            all_results.append(res)

        except Exception as exc:
            logger.warning("C-SIDE DESeq2 failed for cell type '%s': %s", ct, exc)
            continue

    if not all_results:
        logger.warning("C-SIDE: no cell types produced results.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        "C-SIDE complete: %d cell types, %d gene-tests, %d significant (padj < 0.05)",
        combined["group"].nunique(),
        len(combined),
        (combined["pval_adj"] < 0.05).sum(),
    )
    return combined
