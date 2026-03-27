"""
preprocessing.py
----------------
Quality control, normalisation, dimensionality reduction,
Harmony integration, UMAP and Leiden clustering for Xenium data.
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


# ===========================================================================
# QC
# ===========================================================================

def run_qc(
    adata: ad.AnnData,
    min_counts: int = 10,
    max_counts: int = 5_000,
    min_genes: int = 5,
    max_genes: int = 500,
    min_cells_per_gene: int = 10,
    filter_control_probes: bool = True,
    filter_control_codewords: bool = True,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Compute per-cell QC metrics and apply hard thresholds.

    Adds to .obs:
        n_genes_by_counts, total_counts
    Adds to .var:
        n_cells_by_counts, mean_counts, pct_dropout_by_counts

    Parameters
    ----------
    adata:
        Raw counts AnnData (cells x genes).
    min/max_counts:
        Total transcript count thresholds per cell.
    min/max_genes:
        Unique-gene count thresholds per cell.
    min_cells_per_gene:
        Genes detected in fewer than this many cells are removed.
    filter_control_probes:
        Remove cells where control_probe_counts > 0.  These are cells that
        captured at least one non-biological probe signal — 10x Genomics
        sets a warning threshold at 0.02 and error at 0.05 for this metric.
        The Mouse Brain panel has 27 negative control probe sets (more than
        any other Xenium panel), making this filter especially important.
    filter_control_codewords:
        Remove cells where control_codeword_counts > 0 (spurious barcode
        decoding events).
    inplace:
        Modify and return adata, or return a copy.

    Returns
    -------
    Filtered AnnData.
    """
    if not inplace:
        adata = adata.copy()

    # Make sure raw counts are in .X before QC
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]

    sc.pp.calculate_qc_metrics(
        adata,
        expr_type="counts",
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    n_before = adata.n_obs
    gene_before = adata.n_vars

    # Cell filters
    mask_cells = (
        (adata.obs["total_counts"] >= min_counts)
        & (adata.obs["total_counts"] <= max_counts)
        & (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
    )

    # ── Negative control probe filters (Xenium-specific) ─────────────────
    # The Mouse Brain panel has 27 negative control probe sets.  Any cell
    # with a positive control signal has a compromised measurement and should
    # be excluded regardless of its transcript count.
    n_ctrl_probe   = 0
    n_ctrl_codeword = 0
    if filter_control_probes and "control_probe_counts" in adata.obs.columns:
        ctrl_mask = adata.obs["control_probe_counts"] == 0
        n_ctrl_probe = int((~ctrl_mask).sum())
        mask_cells   = mask_cells & ctrl_mask
        logger.info(
            "QC: %d cells removed (control_probe_counts > 0)", n_ctrl_probe
        )
    elif filter_control_probes:
        logger.debug(
            "QC: control_probe_counts not in obs — skipping control probe filter "
            "(column absent; may be an older XOA version)."
        )

    if filter_control_codewords and "control_codeword_counts" in adata.obs.columns:
        cword_mask = adata.obs["control_codeword_counts"] == 0
        n_ctrl_codeword = int((~cword_mask).sum())
        mask_cells = mask_cells & cword_mask
        logger.info(
            "QC: %d cells removed (control_codeword_counts > 0)", n_ctrl_codeword
        )
    elif filter_control_codewords:
        logger.debug(
            "QC: control_codeword_counts not in obs — skipping codeword filter."
        )

    adata = adata[mask_cells].copy()

    # Gene filter
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    logger.info(
        "QC: removed %d cells (%d kept) and %d genes (%d kept)",
        n_before - adata.n_obs,
        adata.n_obs,
        gene_before - adata.n_vars,
        adata.n_vars,
    )
    return adata


# ===========================================================================
# Normalisation and feature selection
# ===========================================================================


def _apply_cell_area_norm(adata: ad.AnnData, enabled: bool) -> None:
    """
    Optionally divide log-normalised expression by cell area (in situ).

    This corrects for the spatial technical confound in brain tissue where
    neurons have much larger cross-sectional areas than glia, and partially-
    captured surface cells appear artificially small.  Applied in-place to
    adata.X and adata.layers['lognorm'].

    Only runs if enabled=True AND adata.obs['cell_area'] exists and is > 0.
    """
    if not enabled:
        return
    if "cell_area" not in adata.obs.columns:
        logger.warning(
            "normalize_by_cell_area=True but 'cell_area' not in obs — skipping. "
            "This column comes from cells.parquet; check your Xenium output version."
        )
        return

    areas = adata.obs["cell_area"].values.astype(float)
    invalid = (areas <= 0) | np.isnan(areas)
    if invalid.any():
        logger.warning(
            "%d cells have cell_area <= 0 or NaN; these cells are excluded from "
            "area normalisation (their values are left unchanged).", int(invalid.sum())
        )
        areas[invalid] = 1.0   # no-op for those cells

    # Scale to median area so magnitudes remain interpretable
    median_area = float(np.median(areas[~invalid]))
    scale = median_area / areas    # cells larger than median scaled down
    scale[invalid] = 1.0

    import scipy.sparse as _sp
    if _sp.issparse(adata.X):
        adata.X = adata.X.multiply(scale[:, None]).tocsr()
    else:
        adata.X = adata.X * scale[:, None]

    if "lognorm" in adata.layers:
        if _sp.issparse(adata.layers["lognorm"]):
            adata.layers["lognorm"] = adata.layers["lognorm"].multiply(scale[:, None]).tocsr()
        else:
            adata.layers["lognorm"] = adata.layers["lognorm"] * scale[:, None]

    logger.info(
        "Cell area normalisation applied (median area = %.1f µm²; "
        "range %.1f – %.1f µm²).",
        median_area,
        float(areas[~invalid].min()),
        float(areas[~invalid].max()),
    )

def normalise_and_select_hvg(
    adata: ad.AnnData,
    target_sum: float = 100.0,
    n_top_genes: int = 0,
    normalize_by_cell_area: bool = False,
    flavor: str = "seurat_v3",
) -> ad.AnnData:
    """
    Normalise counts and log-transform.  HVG selection is skipped by default.

    For targeted spatial panels such as Xenium (typically 100-500 genes),
    HVG selection should be omitted entirely.  Xenium benchmarking work
    (Janesick et al., 2023; bioRxiv) found that ~96% of panel genes are
    spatially variable, so HVG filtering retains nearly all genes anyway
    while introducing unnecessary stochasticity.  Using all genes gives a
    more reproducible, information-maximising PCA.

    HVG selection is only run when ``n_top_genes > 0`` AND
    ``n_top_genes < adata.n_vars`` (i.e. the threshold would actually
    discard some genes).  In all other cases it is skipped.

    Parameters
    ----------
    adata:
        QC-filtered AnnData (raw counts in .X or .layers['counts']).
    target_sum:
        Library-size normalisation target (default 10 000).
    n_top_genes:
        If 0 (default) or >= adata.n_vars, skip HVG selection and use all
        genes for PCA.  Set to a positive integer smaller than the panel
        size only if you have a specific reason to restrict features.
    normalize_by_cell_area:
        If True and adata.obs['cell_area'] is available, divide each cell's
        log-normalised expression by its cross-sectional area (in square
        microns).  Reduces the spatial technical confound caused by variable
        cell sizes in brain tissue (neurons are dramatically larger than glia
        and partially-captured surface cells are smaller).  Applied after
        library-size normalisation and log transformation.  Default False.
    flavor:
        HVG method used only when n_top_genes is active.

    Returns
    -------
    adata with log-normalised values in .X and .layers['lognorm'];
    raw counts preserved in .layers['counts'].
    If HVG selection is active, .var['highly_variable'] is also set.
    """
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    run_hvg = (n_top_genes > 0) and (n_top_genes < adata.n_vars)

    if not run_hvg:
        # Targeted panel path (recommended for Xenium): use ALL genes.
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.layers["lognorm"] = adata.X.copy()
        # Mark all genes as highly_variable so downstream checks still work.
        adata.var["highly_variable"] = True
        logger.info(
            "HVG selection skipped — using all %d genes for PCA "
            "(targeted panel; n_top_genes=%d).",
            adata.n_vars, n_top_genes,
        )
        _apply_cell_area_norm(adata, normalize_by_cell_area)
        return adata

    # HVG path — only reached when explicitly requested
    n_top_genes = min(n_top_genes, adata.n_vars)
    _effective_flavor = flavor
    if flavor == "seurat_v3":
        try:
            import skmisc  # noqa: F401
        except ImportError:
            logger.warning(
                "skmisc not installed — falling back to flavor='seurat' for HVG selection."
            )
            _effective_flavor = "seurat"

    _apply_cell_area_norm(adata, normalize_by_cell_area)

    if _effective_flavor == "seurat_v3":
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes,
            flavor="seurat_v3", batch_key="condition", subset=False,
        )
        logger.info(
            "Selected %d highly variable genes (of %d total) using seurat_v3",
            adata.var["highly_variable"].sum(), adata.n_vars,
        )
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.layers["lognorm"] = adata.X.copy()
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.layers["lognorm"] = adata.X.copy()
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes,
            flavor=_effective_flavor, batch_key="condition", subset=False,
        )
        logger.info(
            "Selected %d highly variable genes (of %d total) using %s",
            adata.var["highly_variable"].sum(), adata.n_vars, _effective_flavor,
        )
    return adata

def run_pca(
    adata: ad.AnnData,
    n_pcs: int = 50,
    use_hvg: bool = True,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Run PCA on the log-normalised HVG matrix.

    Stores results in ``.obsm['X_pca']`` and ``.uns['pca']``.
    """
    # n_comps must be < min(n_obs, n_vars) — cap against both
    n_comps = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    if n_comps < 2:
        raise ValueError(
            f"Too few cells ({adata.n_obs}) or genes ({adata.n_vars}) for PCA. "
            "Check that your ROI is not empty and QC thresholds are not too strict."
        )
    if use_hvg and "highly_variable" in adata.var.columns:
        # use_highly_variable is deprecated in scanpy >= 1.10.
        # mask_var is the modern equivalent. We try the new API first
        # and fall back to the old one for older scanpy versions.
        try:
            sc.tl.pca(
                adata,
                n_comps=n_comps,
                mask_var="highly_variable",
                svd_solver="arpack",
                random_state=random_state,
            )
        except TypeError:
            sc.tl.pca(
                adata,
                n_comps=n_comps,
                use_highly_variable=True,
                svd_solver="arpack",
                random_state=random_state,
            )
    else:
        sc.tl.pca(
            adata,
            n_comps=n_comps,
            svd_solver="arpack",
            random_state=random_state,
        )

    logger.info(
        "PCA: %d components explain %.1f%% variance",
        n_comps,
        adata.uns["pca"]["variance_ratio"][:n_comps].sum() * 100,
    )
    return adata


# ===========================================================================
# Harmony integration
# ===========================================================================

def run_harmony(
    adata: ad.AnnData,
    batch_key: str = "slide_id",
    max_iter: int = 20,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Correct batch effects with Harmony on the PCA embedding.

    Requires ``harmonypy`` to be installed.
    Stores the corrected embedding in ``.obsm['X_pca_harmony']``.

    Parameters
    ----------
    adata:
        AnnData with ``.obsm['X_pca']`` already computed.
    batch_key:
        Column in ``.obs`` distinguishing technical batches.
        MUST be 'slide_id' (or 'run_name') for a multi-slide study --
        using 'condition' here removes the biological effect you are
        trying to detect.
    max_iter:
        Maximum number of Harmony iterations.
    """
    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        logger.warning(
            "Harmony skipped: only %d unique value(s) for batch_key='%s'. "
            "Copying X_pca to X_pca_harmony unchanged.",
            n_batches, batch_key,
        )
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()
        return adata

    logger.info("Running Harmony integration on key '%s' (%d batches) …", batch_key, n_batches)

    # The scanpy harmony wrapper (sce.pp.harmony_integrate) has a known shape-
    # mismatch bug in certain scanpy/harmonypy version combinations where it
    # stores the transposed embedding (shape (n_pcs,) instead of (n_cells, n_pcs)).
    # We probe the wrapper on a tiny subsample first; if it produces the wrong
    # shape we fall through to the robust direct-harmonypy path immediately,
    # without emitting a confusing WARNING on every production run.

    _wrapper_ok = False
    try:
        import scanpy.external as sce
        # Quick sanity-check: run on a 20-cell subsample to see if the wrapper
        # stores the result with the correct shape before touching the full data.
        _n_probe = min(20, adata.n_obs)
        _probe   = adata[:_n_probe].copy()
        sce.pp.harmony_integrate(
            _probe,
            key=batch_key,
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
            max_iter_harmony=1,
            random_state=random_state,
            verbose=False,
        )
        if ("X_pca_harmony" in _probe.obsm
                and _probe.obsm["X_pca_harmony"].shape[0] == _n_probe):
            _wrapper_ok = True
        del _probe
    except Exception:
        pass

    if _wrapper_ok:
        sce.pp.harmony_integrate(
            adata,
            key=batch_key,
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
            max_iter_harmony=max_iter,
            random_state=random_state,
        )
        logger.info(
            "Harmony complete via scanpy wrapper. Embedding shape: %s",
            adata.obsm["X_pca_harmony"].shape,
        )
    else:
        # Wrapper unavailable or produces wrong shape — use harmonypy directly.
        logger.debug("scanpy harmony wrapper unavailable or shape mismatch; using harmonypy directly.")
        try:
            import harmonypy as hm
        except ImportError:
            raise ImportError("harmonypy is required. Install with: pip install harmonypy")

        # Silence harmonypy's own logger for this call — it emits verbose
        # KMeans initialisation and iteration INFO lines that add no value.
        import logging as _logging
        _logging.getLogger("harmonypy").setLevel(_logging.WARNING)

        pca_mat = adata.obsm["X_pca"].copy()
        meta    = adata.obs[[batch_key]].copy()
        ho = hm.run_harmony(
            pca_mat, meta, batch_key,
            max_iter_harmony=max_iter,
            random_state=random_state,
            verbose=False,
        )
        # Extract corrected embeddings — handle all known API shapes
        n_cells, n_pcs = pca_mat.shape
        Z = None
        for attr in ["Z_corr", "result", "embedding"]:
            candidate = getattr(ho, attr, None)
            if candidate is not None:
                Z = np.array(candidate)
                break
        if Z is None:
            raise AttributeError(
                f"harmonypy result has no recognised embedding attribute. "
                f"Available attrs: {[a for a in dir(ho) if not a.startswith('_')]}"
            )
        # Ensure shape is (n_cells, n_pcs)
        if Z.ndim == 1:
            raise ValueError(
                f"harmonypy returned a 1D array of shape {Z.shape}. "
                "Please upgrade harmonypy: pip install --upgrade harmonypy"
            )
        if Z.shape == (n_pcs, n_cells):
            Z = Z.T   # old API: (n_pcs, n_cells) -> (n_cells, n_pcs)
        elif Z.shape != (n_cells, n_pcs):
            raise ValueError(
                f"harmonypy returned unexpected shape {Z.shape}; "
                f"expected ({n_cells}, {n_pcs}) or ({n_pcs}, {n_cells})."
            )
        adata.obsm["X_pca_harmony"] = Z.astype(np.float32)
        logger.info(
            "Harmony complete via direct harmonypy. Embedding shape: %s",
            adata.obsm["X_pca_harmony"].shape,
        )
    return adata


# ===========================================================================
# Neighbourhood graph, UMAP, clustering
# ===========================================================================

def build_graph_and_umap(
    adata: ad.AnnData,
    use_harmony: bool = True,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    umap_min_dist: float = 0.3,
    umap_spread: float = 1.0,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Build the KNN graph and compute UMAP.

    Parameters
    ----------
    use_harmony:
        If True and ``.obsm['X_pca_harmony']`` exists, use the Harmony-
        corrected embedding; otherwise fall back to raw PCA.
    n_neighbors:
        Number of neighbours for the KNN graph.
    n_pcs:
        Number of PCs / harmony components to use.
    """
    rep = (
        "X_pca_harmony"
        if (use_harmony and "X_pca_harmony" in adata.obsm)
        else "X_pca"
    )
    logger.info("Building KNN graph using representation '%s' …", rep)

    # Cap n_pcs to what PCA actually computed
    actual_pcs = adata.obsm[rep].shape[1] if rep in adata.obsm else n_pcs
    n_pcs_use  = min(n_pcs, actual_pcs)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(n_neighbors, adata.n_obs - 1),
        n_pcs=n_pcs_use,
        use_rep=rep,
        random_state=random_state,
    )
    sc.tl.umap(
        adata,
        min_dist=umap_min_dist,
        spread=umap_spread,
        random_state=random_state,
    )
    logger.info("UMAP computed.")
    return adata


def run_leiden(
    adata: ad.AnnData,
    resolution: float = 0.5,
    key_added: str = "leiden",
    random_state: int = 42,
) -> ad.AnnData:
    """
    Cluster cells with the Leiden algorithm.

    Requires ``leidenalg`` and ``igraph``.
    Results stored in ``.obs[key_added]``.
    """
    # Use flavor="igraph" explicitly to suppress the FutureWarning that the
    # default backend is changing from leidenalg to igraph in a future scanpy
    # release. igraph is faster and produces equivalent results; directed=False
    # is required by igraph's implementation.
    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except TypeError:
        # Older scanpy versions do not support flavor/n_iterations/directed;
        # fall back to the plain call which uses leidenalg.
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            random_state=random_state,
        )
    # Ensure Leiden output is categorical (required for groupby observed=True)
    adata.obs[key_added] = adata.obs[key_added].astype("category")
    n_clusters = adata.obs[key_added].nunique()
    logger.info("Leiden clustering: %d clusters at resolution %.2f", n_clusters, resolution)
    return adata


# ===========================================================================
# Marker genes
# ===========================================================================

def find_marker_genes(
    adata: ad.AnnData,
    groupby: str = "leiden",
    method: str = "wilcoxon",
    n_genes: int = 25,
    use_raw: bool = False,
) -> ad.AnnData:
    """
    Rank genes per cluster using Wilcoxon rank-sum (or t-test).

    Results stored in ``.uns['rank_genes_groups']``.

    Parameters
    ----------
    groupby:
        obs column to group cells (clusters or cell types).
    method:
        Statistical test: 'wilcoxon' | 't-test' | 'logreg'.
    n_genes:
        Number of top marker genes to retain per group.
    use_raw:
        Whether to use .raw or .X for the test. Keep False when .X
        holds log-normalised values.
    """
    # Use log-normalised values for the test, but work on a shallow copy so
    # we do not silently leave the caller's .X pointing at the lognorm layer.
    if "lognorm" in adata.layers:
        adata = adata.copy()
        adata.X = adata.layers["lognorm"]

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        use_raw=use_raw,
        pts=True,        # store fraction of cells expressing gene
    )
    logger.info("Marker genes computed for %s.", groupby)
    return adata


# ===========================================================================
# Full preprocessing pipeline (convenience wrapper)
# ===========================================================================

def full_preprocessing_pipeline(adata: ad.AnnData, cfg) -> ad.AnnData:
    """
    Run the end-to-end preprocessing pipeline using a PipelineConfig.

    Steps:
        1. QC filtering (count thresholds + Xenium negative control probe filter)
        2. Normalisation (target_sum=100 for Xenium) + optional cell area normalisation
        3. PCA (all genes; HVG selection skipped for targeted panels)
        4. Harmony integration
        5. KNN graph + UMAP
        6. Leiden clustering
        7. Marker gene detection
    """
    adata = run_qc(
        adata,
        min_counts               = cfg.min_counts,
        max_counts               = cfg.max_counts,
        min_genes                = cfg.min_genes,
        max_genes                = cfg.max_genes,
        min_cells_per_gene       = cfg.min_cells_per_gene,
        filter_control_probes    = getattr(cfg, "filter_control_probes",    True),
        filter_control_codewords = getattr(cfg, "filter_control_codewords", True),
    )
    adata = normalise_and_select_hvg(
        adata,
        target_sum             = cfg.target_sum,
        n_top_genes            = cfg.n_top_genes,
        normalize_by_cell_area = getattr(cfg, "normalize_by_cell_area", False),
    )
    adata = run_pca(adata, n_pcs=cfg.n_pcs, random_state=cfg.random_state)
    adata = run_harmony(
        adata,
        batch_key    = cfg.harmony_key,
        max_iter     = cfg.harmony_max_iter,
        random_state = cfg.random_state,
    )
    adata = build_graph_and_umap(
        adata,
        use_harmony    = True,
        n_neighbors    = cfg.n_neighbors,
        n_pcs          = min(30, cfg.n_pcs),
        umap_min_dist  = cfg.umap_min_dist,
        umap_spread    = cfg.umap_spread,
        random_state   = cfg.random_state,
    )
    adata = run_leiden(
        adata,
        resolution   = cfg.leiden_resolution,
        key_added    = cfg.cluster_key,
        random_state = cfg.random_state,
    )
    adata = find_marker_genes(adata, groupby=cfg.cluster_key)
    return adata
