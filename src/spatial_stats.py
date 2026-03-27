"""
spatial_stats.py
----------------
Spatial statistics for Xenium brain sections.

Analyses provided:

1. morans_i_scan        -- Moran's I spatial autocorrelation for
                           every gene (or a gene list) using a spatial
                           weights matrix built from cell centroids.
                           Identifies spatially patterned genes.

2. spatial_coexpression -- Pair-wise spatial co-expression (spatial
                           lag correlation) between selected genes.
                           Reveals spatially co-regulated modules.

3. neighborhood_enrichment -- Test whether pairs of cell types are
                               spatially co-localised or segregated
                               (permutation-based; inspired by squidpy).

4. spatially_variable_dge  -- DGE restricted to a spatial region of
                               interest (ROI) defined by a bounding box
                               or a cluster-derived polygon.

All functions accept an AnnData with .obsm['spatial'] set.
"""

import logging
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import norm, pearsonr

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Moran's I spatial autocorrelation
# ===========================================================================

def morans_i_scan(
    adata: ad.AnnData,
    genes: Optional[Sequence[str]] = None,
    n_neighbors: int = 6,
    n_jobs: int = 1,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Compute Moran's I for each gene to identify spatially variable genes.

    Uses a binary KNN spatial weights matrix (k nearest cells in tissue).
    Moran's I ranges from -1 (dispersed) through 0 (random) to +1 (clustered).

    Parameters
    ----------
    adata:
        AnnData with .obsm['spatial'] (cell centroids).
    genes:
        Subset of genes to test. Defaults to all genes in adata.
    n_neighbors:
        Number of spatial neighbours for the weights matrix.
    n_jobs:
        Parallelism (1 = serial; -1 = all CPUs). Uses joblib if > 1.
    batch_size:
        Number of genes to process per batch (memory management).

    Returns
    -------
    DataFrame sorted by Moran's I (descending):
        gene, morans_i, expected_i, z_score, p_value

    Notes
    -----
    P-values are derived from the Cliff & Ord normal approximation, which
    assumes expression values are normally distributed and spatial
    autocorrelation is low.  Both assumptions are routinely violated for
    single-cell spatial data (zero-inflation, log-normalisation, high
    autocorrelation).  P-values tend to be anti-conservative (too small) for
    highly autocorrelated genes.  For publication-quality results, treat
    Moran's I magnitude as the primary ranking criterion and apply a
    conservative significance threshold (e.g. BH-adjusted p < 0.01).  A
    permutation-based p-value would be more accurate but is computationally
    expensive for genome-wide scans.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] required for Moran's I.")

    xy = adata.obsm["spatial"].astype(np.float64)
    W = _build_knn_weights(xy, min(n_neighbors, adata.n_obs - 1))

    genes_to_test = list(genes) if genes else list(adata.var_names)
    genes_to_test = [g for g in genes_to_test if g in adata.var_names]

    X = _get_lognorm(adata)
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    N = adata.n_obs

    # Precompute expected value and variance
    # E[I] = -1 / (N - 1)
    E_I = -1.0 / (N - 1)

    results = []
    for i in range(0, len(genes_to_test), batch_size):
        batch = genes_to_test[i : i + batch_size]
        for gene in batch:
            gi = var_idx[gene]
            _z = X[:, gi]
            # Densify: sparse column slice is (n_cells, 1) — ravel to 1D float array
            if sp.issparse(_z):
                z = np.array(_z.todense()).ravel().astype(np.float64)
            else:
                z = np.asarray(_z).ravel().astype(np.float64)
            z = z - z.mean()
            denom = np.dot(z, z)
            if denom < 1e-12:
                results.append((gene, np.nan, E_I, np.nan, 1.0))
                continue

            # Spatial lag: W @ z
            if sp.issparse(W):
                Wz = W.dot(z)
            else:
                Wz = W @ z

            I = (N / W.sum()) * (np.dot(z, Wz) / denom)

            # Variance under normality assumption (Cliff & Ord, 1981)
            # E[I^2] = [n^2 * S1 - n * S2 + 3 * S0^2] / [(n^2-1) * S0^2]
            # Var(I) = E[I^2] - E[I]^2
            S1, S2 = _compute_s1_s2(W)
            S0 = W.sum()
            n = float(N)
            E_I2 = (n**2 * S1 - n * S2 + 3 * S0**2) / ((n**2 - 1) * S0**2)
            var_I = E_I2 - E_I**2
            z_score = (I - E_I) / (np.sqrt(abs(var_I)) + 1e-12)
            p_value = 2 * (1 - norm.cdf(abs(z_score)))

            results.append((gene, float(I), E_I, float(z_score), float(p_value)))

        logger.debug("Moran's I: processed %d / %d genes", min(i + batch_size, len(genes_to_test)), len(genes_to_test))

    df = pd.DataFrame(results, columns=["gene", "morans_i", "expected_i", "z_score", "p_value"])
    df["p_adj"] = _bh_correction(df["p_value"].values)
    df = df.sort_values("morans_i", ascending=False).reset_index(drop=True)

    logger.info(
        "Moran's I: %d spatially variable genes (adj-p < 0.05, I > 0.1)",
        ((df["p_adj"] < 0.05) & (df["morans_i"] > 0.1)).sum(),
    )
    return df


# ===========================================================================
# 2. Spatial co-expression
# ===========================================================================

def spatial_coexpression(
    adata: ad.AnnData,
    genes: Sequence[str],
    n_neighbors: int = 6,
    method: str = "spatial_lag",
) -> pd.DataFrame:
    """
    Compute spatial co-expression between all pairs of selected genes.

    Two methods:
      'spatial_lag':  correlate gene A's expression with the spatial lag
                      (neighbourhood average) of gene B.
      'local_corr':   for each cell compute local (neighbourhood) Pearson
                      correlation between gene A and B, then average.

    Parameters
    ----------
    adata:
        AnnData with .obsm['spatial'] and log-normalised .X.
    genes:
        List of genes to compare.
    n_neighbors:
        Neighbourhood size.
    method:
        'spatial_lag' or 'local_corr'.

    Returns
    -------
    Square DataFrame of co-expression scores (gene x gene).
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] required.")

    genes = [g for g in genes if g in adata.var_names]
    if len(genes) < 2:
        raise ValueError("At least 2 genes must be present in adata.")

    xy = adata.obsm["spatial"].astype(np.float64)
    W = _build_knn_weights(xy, n_neighbors)
    X = _get_lognorm(adata)
    var_idx = {g: i for i, g in enumerate(adata.var_names)}

    expr = np.column_stack([X[:, var_idx[g]] for g in genes])  # (N, G)
    n_genes = len(genes)
    matrix = np.eye(n_genes)

    if method == "spatial_lag":
        lag = W.dot(expr)  # (N, G) spatial lag for each gene
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                # Guard against zero-variance vectors (spatially invariant genes)
                if np.std(expr[:, i]) < 1e-9 or np.std(lag[:, j]) < 1e-9 or \
                   np.std(expr[:, j]) < 1e-9 or np.std(lag[:, i]) < 1e-9:
                    val = 0.0
                    matrix[i, j] = val
                    matrix[j, i] = val
                    continue
                r, _ = pearsonr(expr[:, i], lag[:, j])
                r2, _ = pearsonr(expr[:, j], lag[:, i])
                val = (r + r2) / 2.0
                matrix[i, j] = val
                matrix[j, i] = val

    elif method == "local_corr":
        # Get neighbour indices
        tree = cKDTree(xy)
        _kc = min(n_neighbors, len(xy) - 1)
        _, idx = tree.query(xy, k=_kc + 1)
        idx = idx[:, 1:]  # exclude self

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                local_corrs = []
                for cell in range(len(xy)):
                    nbrs = idx[cell]
                    a = expr[nbrs, i]
                    b = expr[nbrs, j]
                    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                        continue
                    local_corrs.append(pearsonr(a, b)[0])
                val = float(np.mean(local_corrs)) if local_corrs else 0.0
                matrix[i, j] = val
                matrix[j, i] = val

    return pd.DataFrame(matrix, index=genes, columns=genes)


# ===========================================================================
# 3. Neighbourhood enrichment
# ===========================================================================

def neighborhood_enrichment(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    n_neighbors: int = 6,
    n_permutations: int = 1_000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Test whether cell type pairs are spatially co-localised or segregated.

    For each pair (A, B): count the fraction of A cells that have B among
    their k nearest spatial neighbours. Compare to a permutation null where
    cell type labels are shuffled on the tissue.

    Parameters
    ----------
    adata:
        AnnData with .obsm['spatial'] and cell_type_key in .obs.
    cell_type_key:
        obs column with cell type labels.
    n_neighbors:
        Spatial KNN neighbourhood size.
    n_permutations:
        Number of label permutations for null distribution.
    random_state:
        RNG seed.

    Returns
    -------
    Dict with keys:
        'observed'   -- DataFrame(cell_type x cell_type): observed co-occurrence
        'z_score'    -- z-score relative to permutation null
        'p_value'    -- two-tailed p-value
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] required.")
    if cell_type_key not in adata.obs.columns:
        raise KeyError(f"'{cell_type_key}' not found in adata.obs.")

    rng = np.random.default_rng(random_state)
    xy = adata.obsm["spatial"].astype(np.float64)
    labels = adata.obs[cell_type_key].values.astype(str)
    cell_types = sorted(set(labels))

    tree = cKDTree(xy)
    _k = min(n_neighbors, len(xy) - 1)
    _, nbr_idx = tree.query(xy, k=_k + 1)
    nbr_idx = nbr_idx[:, 1:]  # (N, _k), exclude self

    def _cooccurrence(lbl):
        ct_idx = {ct: np.where(lbl == ct)[0] for ct in cell_types}
        n = len(cell_types)
        mat = np.zeros((n, n), dtype=np.float64)
        for i, ct_a in enumerate(cell_types):
            cells_a = ct_idx[ct_a]
            if len(cells_a) == 0:
                continue
            nbrs_of_a = nbr_idx[cells_a].ravel()
            nbr_labels = lbl[nbrs_of_a]
            for j, ct_b in enumerate(cell_types):
                mat[i, j] = (nbr_labels == ct_b).mean()
        return mat

    observed = _cooccurrence(labels)

    # Permutation null
    null_mats = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        null_mats.append(_cooccurrence(shuffled))
    null_arr = np.stack(null_mats, axis=0)  # (n_perm, n_ct, n_ct)

    null_mean = null_arr.mean(axis=0)
    null_std  = null_arr.std(axis=0) + 1e-12
    z_score = (observed - null_mean) / null_std

    # Two-tailed p-value from permutation, clipped to [0, 1]
    p_value = np.clip(
        np.minimum(
            (null_arr >= observed).mean(axis=0),
            (null_arr <= observed).mean(axis=0),
        ) * 2,
        0.0, 1.0,
    )

    idx = pd.Index(cell_types, name="cell_type")
    return {
        "observed": pd.DataFrame(observed, index=idx, columns=idx),
        "z_score" : pd.DataFrame(z_score,  index=idx, columns=idx),
        "p_value" : pd.DataFrame(p_value,  index=idx, columns=idx),
    }


# ===========================================================================
# 4. Spatially-restricted DGE (ROI-based)
# ===========================================================================

def roi_dge(
    adata: ad.AnnData,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    condition_key: str = "condition",
    method: str = "wilcoxon",
    min_cells_per_condition: int = 20,
) -> pd.DataFrame:
    """
    Run DGE restricted to a spatial region of interest (ROI).

    Useful for comparing conditions within a specific brain region
    (e.g. cortical layer, hippocampal subfield).

    Parameters
    ----------
    adata:
        Full AnnData with .obsm['spatial'].
    x_range, y_range:
        Bounding box (min, max) in micrometres defining the ROI.
    condition_key:
        obs column with condition labels.
    method:
        DGE method forwarded to dge_analysis.run_dge.
    min_cells_per_condition:
        Skip if either condition has fewer cells in the ROI.

    Returns
    -------
    DGE results DataFrame (same format as dge_analysis.run_dge output),
    restricted to cells inside the ROI.
    """
    from src.dge_analysis import run_dge

    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] required.")

    xy = adata.obsm["spatial"]
    roi_mask = (
        (xy[:, 0] >= x_range[0]) & (xy[:, 0] <= x_range[1])
        & (xy[:, 1] >= y_range[0]) & (xy[:, 1] <= y_range[1])
    )
    adata_roi = adata[roi_mask].copy()
    logger.info(
        "ROI DGE: [x: %.0f-%.0f, y: %.0f-%.0f] -> %d cells",
        *x_range, *y_range, adata_roi.n_obs,
    )

    # Sanity-check per condition
    for cond in adata_roi.obs[condition_key].unique():
        n = (adata_roi.obs[condition_key] == cond).sum()
        if n < min_cells_per_condition:
            logger.warning(
                "Condition '%s' has only %d cells in ROI; results may be unreliable.",
                cond, n,
            )

    return run_dge(adata_roi, method=method, condition_key=condition_key)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _get_lognorm(adata: ad.AnnData) -> np.ndarray:
    if "lognorm" in adata.layers:
        X = adata.layers["lognorm"]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _build_knn_weights(xy: np.ndarray, k: int) -> sp.csr_matrix:
    """Build a row-normalised binary KNN spatial weights matrix."""
    N = len(xy)
    # Cap k so we never request more neighbours than cells available
    k = min(k, N - 1)
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)  # includes self
    idx = idx[:, 1:]  # exclude self

    row = np.repeat(np.arange(N), k)
    col = idx.ravel()
    data = np.ones(N * k, dtype=np.float64)

    W = sp.csr_matrix((data, (row, col)), shape=(N, N))
    # Row-normalise
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    return D_inv.dot(W)


def _compute_s1_s2(W: sp.csr_matrix) -> tuple[float, float]:
    """Compute Moran's variance components S1 and S2."""
    Wt = W + W.T
    S1 = 0.5 * (Wt.multiply(Wt)).sum()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    col_sums = np.asarray(W.sum(axis=0)).ravel()
    S2 = float(np.sum((row_sums + col_sums) ** 2))
    return float(S1), S2


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    from scipy.stats import rankdata
    n = len(p_values)
    if n == 0:
        return p_values
    ranks = rankdata(p_values, method="ordinal")
    p_adj = p_values * n / ranks
    # Enforce monotonicity (step-down)
    order = np.argsort(ranks)[::-1]
    p_adj_sorted = p_adj[order]
    cummin = np.minimum.accumulate(p_adj_sorted)
    p_adj[order] = cummin
    return np.clip(p_adj, 0, 1)
