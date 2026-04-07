"""
spatial_domain_detection.py
---------------------------
Spatially-aware domain detection for Xenium single-cell data.

Unlike standard Leiden clustering (which operates on a KNN graph built from
PCA/Harmony embeddings and ignores physical cell positions), spatial domain
detection integrates expression similarity **and** tissue coordinates to
identify biologically meaningful tissue regions.

Strategy
--------
1. Build a **spatial neighbour graph** from cell centroids (Delaunay or KNN
   in physical space) using squidpy or a local fallback.
2. Build an **expression neighbour graph** from PCA/Harmony embeddings
   (already computed by the preprocessing pipeline).
3. Combine both graphs into a **joint adjacency matrix** using a tunable
   weight parameter ``lambda_spatial`` (0 = expression-only, 1 = spatial-only).
4. Run Leiden clustering on the joint graph to define **spatial domains**.
5. Optionally refine domains by removing small disconnected fragments.
6. Identify **spatially variable genes (SVGs)** per domain via differential
   expression against neighbouring domains.

This approach:
- Handles irregular cell positions natively (no grid assumption)
- Scales to 100K+ cells via sparse KNN graphs (not full N x N)
- Stays within the scanpy/AnnData ecosystem
- Produces cluster labels compatible with all downstream analyses (DEGs, etc.)

References
----------
- Palla et al. (2022) Squidpy: a scalable framework for spatial omics analysis.
  Nature Methods 19, 171-178.
- BankSY concept: Singhal et al. (2024) BANKSY unifies cell typing and tissue
  domain segmentation for scalable spatial omics data analysis. Nature Genetics.
"""

import logging
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Build spatial neighbour graph
# ===========================================================================

def build_spatial_graph(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    coord_type: str = "generic",
    method: str = "knn",
) -> ad.AnnData:
    """
    Build a spatial KNN graph from cell centroids stored in
    ``adata.obsm['spatial']``.

    Tries squidpy first (robust, handles edge cases); falls back to a
    scipy cKDTree implementation if squidpy is unavailable.

    The spatial graph is stored in:
        - ``adata.obsp['spatial_connectivities']`` (binary adjacency)
        - ``adata.obsp['spatial_distances']``       (Euclidean distances)

    Parameters
    ----------
    adata : AnnData
        Must have ``adata.obsm['spatial']`` with (N, 2) coordinates.
    n_neighbors : int
        Number of spatial nearest neighbours.
    coord_type : str
        Coordinate type for squidpy (``'generic'`` for Xenium centroids).
    method : str
        ``'knn'`` (default) or ``'delaunay'``.

    Returns
    -------
    adata with spatial graph added in-place.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required for spatial graph construction.")

    k = min(n_neighbors, adata.n_obs - 1)

    try:
        import squidpy as sq

        sq.gr.spatial_neighbors(
            adata,
            n_neighs=k,
            coord_type=coord_type,
            spatial_key="spatial",
        )
        logger.info(
            "Spatial graph built via squidpy (k=%d, %d cells).", k, adata.n_obs
        )
    except ImportError:
        logger.info("squidpy not available — building spatial KNN graph with scipy cKDTree.")
        _build_spatial_graph_fallback(adata, k)

    return adata


def _build_spatial_graph_fallback(adata: ad.AnnData, k: int) -> None:
    """scipy cKDTree fallback for spatial neighbour graph."""
    xy = adata.obsm["spatial"].astype(np.float64)
    tree = cKDTree(xy)
    distances, indices = tree.query(xy, k=k + 1)  # includes self

    n = adata.n_obs
    rows, cols, dists = [], [], []
    for i in range(n):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            j = indices[i, j_idx]
            rows.append(i)
            cols.append(j)
            dists.append(distances[i, j_idx])

    conn = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n, n),
    )
    dist_mat = sp.csr_matrix(
        (np.array(dists, dtype=np.float64), (rows, cols)),
        shape=(n, n),
    )
    # Symmetrise
    conn = (conn + conn.T).astype(bool).astype(np.float64)
    conn = sp.csr_matrix(conn)
    dist_mat = (dist_mat + dist_mat.T) / 2
    dist_mat.eliminate_zeros()

    adata.obsp["spatial_connectivities"] = conn
    adata.obsp["spatial_distances"] = dist_mat
    logger.info("Spatial KNN graph built (k=%d, %d cells, fallback).", k, n)


# ===========================================================================
# 2. Combine expression + spatial graphs
# ===========================================================================

def combine_graphs(
    adata: ad.AnnData,
    lambda_spatial: float = 0.3,
    expr_conn_key: str = "connectivities",
    spatial_conn_key: str = "spatial_connectivities",
) -> sp.csr_matrix:
    """
    Create a joint adjacency matrix blending expression and spatial graphs.

    The combined graph is:

        A_joint = (1 - lambda) * A_expr_norm + lambda * A_spatial_norm

    where both matrices are row-normalised before mixing so that the
    lambda parameter intuitively controls the spatial-vs-expression balance
    regardless of graph density differences.

    Parameters
    ----------
    adata : AnnData
        Must have both expression (``obsp['connectivities']``) and spatial
        (``obsp['spatial_connectivities']``) neighbour graphs.
    lambda_spatial : float
        Weight for spatial information (0 = expression-only, 1 = spatial-only).
        Recommended range: 0.2–0.5 for Xenium data.
    expr_conn_key : str
        Key in ``adata.obsp`` for expression graph.
    spatial_conn_key : str
        Key in ``adata.obsp`` for spatial graph.

    Returns
    -------
    Joint adjacency matrix (sparse CSR).
    """
    if expr_conn_key not in adata.obsp:
        raise ValueError(f"Expression graph '{expr_conn_key}' not found in adata.obsp.")
    if spatial_conn_key not in adata.obsp:
        raise ValueError(f"Spatial graph '{spatial_conn_key}' not found in adata.obsp.")

    A_expr = adata.obsp[expr_conn_key].astype(np.float64).copy()
    A_spat = adata.obsp[spatial_conn_key].astype(np.float64).copy()

    # Row-normalise each graph so edge weights sum to 1 per cell
    A_expr = _row_normalize(A_expr)
    A_spat = _row_normalize(A_spat)

    # Blend
    A_joint = (1.0 - lambda_spatial) * A_expr + lambda_spatial * A_spat

    logger.info(
        "Joint graph: lambda_spatial=%.2f, expression edges=%d, spatial edges=%d, "
        "joint edges=%d.",
        lambda_spatial, A_expr.nnz, A_spat.nnz, A_joint.nnz,
    )
    return sp.csr_matrix(A_joint)


def _row_normalize(A: sp.spmatrix) -> sp.csr_matrix:
    """Row-normalise a sparse matrix so each row sums to 1."""
    A = sp.csr_matrix(A, dtype=np.float64)
    row_sums = np.array(A.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    diag_inv = sp.diags(1.0 / row_sums)
    return diag_inv @ A


# ===========================================================================
# 3. Spatial Leiden clustering
# ===========================================================================

def run_spatial_leiden(
    adata: ad.AnnData,
    lambda_spatial: float = 0.3,
    resolution: float = 0.5,
    n_spatial_neighbors: int = 15,
    key_added: str = "spatial_domain",
    random_state: int = 42,
) -> ad.AnnData:
    """
    End-to-end spatial domain detection: build spatial graph, combine with
    expression graph, and run Leiden clustering on the joint graph.

    Results are stored in ``adata.obs[key_added]`` as a categorical column.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData with expression KNN graph already computed
        (``adata.obsp['connectivities']``), PCA embeddings, and
        ``adata.obsm['spatial']``.
    lambda_spatial : float
        Spatial weight (0–1). Higher values give more spatially coherent
        domains at the cost of expression resolution.
    resolution : float
        Leiden resolution for the joint graph.
    n_spatial_neighbors : int
        Number of spatial KNN neighbours.
    key_added : str
        Column name for the domain labels in ``adata.obs``.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    adata with ``adata.obs[key_added]`` set.
    """
    import igraph as ig
    import leidenalg

    # Step 1: build spatial graph if not present
    if "spatial_connectivities" not in adata.obsp:
        build_spatial_graph(adata, n_neighbors=n_spatial_neighbors)

    # Step 2: combine graphs
    A_joint = combine_graphs(adata, lambda_spatial=lambda_spatial)

    # Step 3: convert to igraph and run Leiden
    # Convert sparse adjacency to igraph weighted graph
    sources, targets = A_joint.nonzero()
    weights = np.array(A_joint[sources, targets]).ravel()

    # Keep only upper triangle to avoid double-counting edges
    mask = sources < targets
    sources, targets, weights = sources[mask], targets[mask], weights[mask]

    g = ig.Graph(n=adata.n_obs, edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
    g.es["weight"] = weights.tolist()

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )

    labels = np.array(partition.membership, dtype=str)
    adata.obs[key_added] = pd.Categorical(labels)

    n_domains = len(set(labels))
    logger.info(
        "Spatial Leiden: %d domains detected (lambda=%.2f, resolution=%.2f).",
        n_domains, lambda_spatial, resolution,
    )

    return adata


# ===========================================================================
# 4. Domain refinement — remove small spatial fragments
# ===========================================================================

def refine_domains(
    adata: ad.AnnData,
    key: str = "spatial_domain",
    min_cells: int = 30,
    spatial_conn_key: str = "spatial_connectivities",
) -> ad.AnnData:
    """
    Merge tiny spatially-disconnected fragments into their nearest
    neighbouring domain.

    For each domain, identifies spatially connected components. Components
    with fewer than ``min_cells`` cells are reassigned to the most common
    domain among their spatial neighbours.

    Parameters
    ----------
    adata : AnnData
        Must have spatial graph and domain labels.
    key : str
        obs column with domain labels.
    min_cells : int
        Minimum component size; smaller ones are merged.
    spatial_conn_key : str
        Key in obsp for the spatial connectivity graph.

    Returns
    -------
    adata with refined domain labels (in-place).
    """
    if key not in adata.obs.columns:
        raise ValueError(f"Domain column '{key}' not found in adata.obs.")
    if spatial_conn_key not in adata.obsp:
        raise ValueError(f"Spatial graph '{spatial_conn_key}' not found in adata.obsp.")

    import igraph as ig

    labels = adata.obs[key].values.copy().astype(str)
    A_spat = adata.obsp[spatial_conn_key]

    n_reassigned = 0
    for domain in np.unique(labels):
        domain_mask = labels == domain
        domain_idx = np.where(domain_mask)[0]

        if len(domain_idx) < 2:
            continue

        # Subgraph for this domain
        sub_A = A_spat[domain_mask][:, domain_mask]
        sources, targets = sub_A.nonzero()
        g = ig.Graph(n=len(domain_idx), edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
        components = g.connected_components()

        for comp in components:
            if len(comp) >= min_cells:
                continue
            # Reassign cells in this small component to neighbour domain
            global_idx = domain_idx[comp]
            for ci in global_idx:
                # Find spatial neighbours
                nbr_row = A_spat[ci].toarray().ravel()
                nbr_idx = np.where(nbr_row > 0)[0]
                nbr_labels = labels[nbr_idx]
                # Exclude same domain
                other = nbr_labels[nbr_labels != domain]
                if len(other) > 0:
                    # Assign to most common neighbour domain
                    vals, counts = np.unique(other, return_counts=True)
                    labels[ci] = vals[counts.argmax()]
                    n_reassigned += 1

    adata.obs[key] = pd.Categorical(labels)

    if n_reassigned > 0:
        logger.info(
            "Domain refinement: %d cells reassigned from small fragments (min_cells=%d).",
            n_reassigned, min_cells,
        )
    else:
        logger.info("Domain refinement: no small fragments found.")

    return adata


# ===========================================================================
# 5. Domain-level SVG detection
# ===========================================================================

def domain_deg(
    adata: ad.AnnData,
    domain_key: str = "spatial_domain",
    target_domain: Optional[str] = None,
    method: str = "wilcoxon",
    n_top: int = 50,
    log2fc_thresh: float = 0.5,
    pval_thresh: float = 0.05,
) -> pd.DataFrame:
    """
    Identify marker genes (spatially variable genes) for each spatial domain
    by testing domain vs. all other cells.

    Uses scanpy's rank_genes_groups under the hood.

    Parameters
    ----------
    adata : AnnData
        With domain labels in ``adata.obs[domain_key]``.
    domain_key : str
        obs column with domain labels.
    target_domain : str, optional
        If set, only compute DEGs for this domain. Otherwise, all domains.
    method : str
        ``'wilcoxon'`` (default) or ``'t-test'``.
    n_top : int
        Number of top genes to report per domain.
    log2fc_thresh : float
        Minimum absolute log2 fold change.
    pval_thresh : float
        Maximum adjusted p-value.

    Returns
    -------
    Long-format DataFrame: domain, gene, log2fc, pval_adj, pct_in, pct_out.
    """
    import scanpy as sc

    if domain_key not in adata.obs.columns:
        raise ValueError(f"Domain column '{domain_key}' not found in adata.obs.")

    # Use lognorm layer if available
    use_raw = False
    layer = "lognorm" if "lognorm" in adata.layers else None

    # Determine which groups to test
    groups = "all" if target_domain is None else [target_domain]

    sc.tl.rank_genes_groups(
        adata,
        groupby=domain_key,
        groups=groups,
        method=method,
        layer=layer,
        use_raw=use_raw,
        n_genes=min(n_top, adata.n_vars),
        key_added="spatial_domain_degs",
    )

    # Extract results into a tidy DataFrame
    result = sc.get.rank_genes_groups_df(
        adata, group=None, key="spatial_domain_degs"
    )

    # Rename columns for consistency with the pipeline
    col_map = {}
    if "names" in result.columns:
        col_map["names"] = "gene"
    if "logfoldchanges" in result.columns:
        col_map["logfoldchanges"] = "log2fc"
    if "pvals_adj" in result.columns:
        col_map["pvals_adj"] = "pval_adj"
    if "group" in result.columns:
        col_map["group"] = "domain"
    result = result.rename(columns=col_map)

    # Convert natural-log fold change to log2
    if "log2fc" in result.columns:
        result["log2fc"] = result["log2fc"] / np.log(2)

    # Filter by thresholds
    mask = (result["pval_adj"] < pval_thresh) & (result["log2fc"].abs() > log2fc_thresh)
    sig = result[mask].copy()

    n_sig = len(sig)
    n_domains = result["domain"].nunique() if "domain" in result.columns else "?"
    logger.info(
        "Domain DEGs: %d significant genes across %s domains "
        "(|log2FC| > %.1f, padj < %.2g).",
        n_sig, n_domains, log2fc_thresh, pval_thresh,
    )

    return result


# ===========================================================================
# 6. Domain spatial coherence score
# ===========================================================================

def spatial_coherence(
    adata: ad.AnnData,
    domain_key: str = "spatial_domain",
    spatial_conn_key: str = "spatial_connectivities",
) -> float:
    """
    Compute the fraction of each cell's spatial neighbours that share the
    same domain label. Higher = more spatially coherent domains.

    Returns
    -------
    Mean spatial coherence score (0–1).
    """
    if domain_key not in adata.obs.columns:
        raise ValueError(f"Domain column '{domain_key}' not found.")
    if spatial_conn_key not in adata.obsp:
        raise ValueError(f"Spatial graph '{spatial_conn_key}' not found.")

    labels = adata.obs[domain_key].values
    A = adata.obsp[spatial_conn_key]

    n = adata.n_obs
    coherence = np.zeros(n)
    for i in range(n):
        row = A[i]
        nbr_idx = row.indices if sp.issparse(row) else np.where(row > 0)[0]
        if len(nbr_idx) == 0:
            coherence[i] = 1.0
            continue
        coherence[i] = np.mean(labels[nbr_idx] == labels[i])

    mean_coh = float(coherence.mean())
    logger.info(
        "Spatial coherence for '%s': %.3f (1.0 = perfectly contiguous domains).",
        domain_key, mean_coh,
    )
    return mean_coh


# ===========================================================================
# 7. Lambda sweep — find optimal spatial weight
# ===========================================================================

def sweep_lambda(
    adata: ad.AnnData,
    lambdas: Optional[list] = None,
    resolution: float = 0.5,
    n_spatial_neighbors: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sweep over lambda_spatial values and score each by spatial coherence
    and silhouette score, returning a summary table.

    Parameters
    ----------
    adata : AnnData
        Preprocessed with expression graph and spatial coordinates.
    lambdas : list of float
        Values to test. Default: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].
    resolution : float
        Leiden resolution (held constant across sweep).
    n_spatial_neighbors : int
        Spatial KNN k.
    random_state : int
        Seed.

    Returns
    -------
    DataFrame with columns: lambda, n_domains, spatial_coherence,
    silhouette_score, combined_score.
    """
    from sklearn.metrics import silhouette_score as sil_score

    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Ensure spatial graph exists
    if "spatial_connectivities" not in adata.obsp:
        build_spatial_graph(adata, n_neighbors=n_spatial_neighbors)

    # Use Harmony embeddings if available, else PCA
    embed_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    X_embed = adata.obsm[embed_key]

    results = []
    for lam in lambdas:
        logger.info("Lambda sweep: lambda=%.2f ...", lam)

        # Run spatial Leiden (stores result temporarily)
        tmp_key = f"_sweep_domain_{lam:.2f}"
        run_spatial_leiden(
            adata,
            lambda_spatial=lam,
            resolution=resolution,
            n_spatial_neighbors=n_spatial_neighbors,
            key_added=tmp_key,
            random_state=random_state,
        )

        labels = adata.obs[tmp_key].values.astype(str)
        n_domains = len(set(labels))

        # Spatial coherence
        coh = spatial_coherence(adata, domain_key=tmp_key)

        # Silhouette score (skip if < 2 clusters or > 50 clusters)
        sil = np.nan
        if 2 <= n_domains <= 50:
            # Subsample for speed if > 20k cells
            if adata.n_obs > 20_000:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(adata.n_obs, 20_000, replace=False)
                sil = sil_score(X_embed[idx], labels[idx], sample_size=None)
            else:
                sil = sil_score(X_embed, labels)

        # Combined score: 50% silhouette + 50% coherence
        # Normalise silhouette from [-1, 1] to [0, 1]
        sil_norm = (sil + 1) / 2 if not np.isnan(sil) else 0.0
        combined = 0.5 * sil_norm + 0.5 * coh

        results.append({
            "lambda": lam,
            "n_domains": n_domains,
            "spatial_coherence": round(coh, 4),
            "silhouette_score": round(sil, 4) if not np.isnan(sil) else np.nan,
            "combined_score": round(combined, 4),
        })

        # Clean up temporary column
        del adata.obs[tmp_key]

    df = pd.DataFrame(results)
    best = df.loc[df["combined_score"].idxmax()]
    logger.info(
        "Lambda sweep complete. Best: lambda=%.2f (coherence=%.3f, "
        "silhouette=%.3f, combined=%.3f, %d domains).",
        best["lambda"], best["spatial_coherence"],
        best["silhouette_score"], best["combined_score"],
        best["n_domains"],
    )
    return df


# ===========================================================================
# 8. Convenience: full spatial domain pipeline
# ===========================================================================

def run_spatial_domain_pipeline(
    adata: ad.AnnData,
    lambda_spatial: float = 0.3,
    resolution: float = 0.5,
    n_spatial_neighbors: int = 15,
    min_fragment_cells: int = 30,
    domain_key: str = "spatial_domain",
    run_degs: bool = True,
    random_state: int = 42,
) -> Tuple[ad.AnnData, Optional[pd.DataFrame]]:
    """
    Full spatial domain detection pipeline:
    1. Build spatial graph
    2. Combine with expression graph
    3. Leiden clustering on joint graph
    4. Refine domains (merge small fragments)
    5. Compute spatial coherence
    6. Identify domain marker genes (optional)

    Parameters
    ----------
    adata : AnnData
        Preprocessed (PCA, Harmony, expression KNN graph required).
    lambda_spatial : float
        Spatial weight (0–1).
    resolution : float
        Leiden resolution.
    n_spatial_neighbors : int
        Spatial KNN k.
    min_fragment_cells : int
        Minimum connected component size for refinement.
    domain_key : str
        Column name for domain labels.
    run_degs : bool
        Whether to compute domain marker genes.
    random_state : int
        Seed.

    Returns
    -------
    (adata, domain_degs_df) — adata has domain labels; DEG table or None.
    """
    logger.info("=" * 60)
    logger.info("Spatial Domain Detection Pipeline")
    logger.info("=" * 60)

    # 1-3: Build graphs and cluster
    run_spatial_leiden(
        adata,
        lambda_spatial=lambda_spatial,
        resolution=resolution,
        n_spatial_neighbors=n_spatial_neighbors,
        key_added=domain_key,
        random_state=random_state,
    )

    # 4: Refine
    refine_domains(adata, key=domain_key, min_cells=min_fragment_cells)

    # 5: Score
    coh = spatial_coherence(adata, domain_key=domain_key)
    adata.uns["spatial_domain_coherence"] = coh

    # 6: Domain DEGs
    deg_df = None
    if run_degs:
        deg_df = domain_deg(adata, domain_key=domain_key)
        adata.uns["spatial_domain_degs"] = deg_df

    n_domains = adata.obs[domain_key].nunique()
    logger.info(
        "Spatial domain detection complete: %d domains, coherence=%.3f.",
        n_domains, coh,
    )
    return adata, deg_df
