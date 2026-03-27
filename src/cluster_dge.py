"""
cluster_dge.py
--------------
Per-cluster (or per-cell-type) differential gene expression between
two conditions.

Rationale: global DGE averages over all cell types. For spatial
transcriptomics of brain sections, the key biological question is
often: "which genes change in Neurons vs Astrocytes vs Microglia?"

This module iterates over every group (cluster or annotated cell type)
and runs the chosen DGE method independently, then collates results
into a long-format table and computes a cross-cluster summary.

Outputs
-------
  cluster_dge_results.csv   -- long format: cluster, gene, log2fc, padj, ...
  cluster_dge_summary.csv   -- number of DEGs per cluster
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_cluster_dge(
    adata: ad.AnnData,
    group_key: str = "leiden",
    condition_key: str = "condition",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    method: str = "stringent_wilcoxon",
    replicate_key: Optional[str] = None,
    min_cells_per_condition: int = 10,
    log2fc_thresh: float = 1.0,   # aligned with stringent_wilcoxon lfc filter
    pval_thresh: float = 0.01,    # aligned with stringent_wilcoxon p filter
    output_dir=None,
) -> pd.DataFrame:
    """
    Run DGE between two conditions within each cluster / cell type.

    Parameters
    ----------
    adata:
        Preprocessed AnnData with group_key and condition_key in .obs.
    group_key:
        obs column defining groups (e.g. 'leiden' or 'cell_type').
    condition_key:
        obs column with condition labels.
    condition_a / condition_b:
        The two conditions. Auto-inferred if None.
    method:
        'wilcoxon' | 'pydeseq2' | 't-test'. Forwarded to dge_analysis.
    min_cells_per_condition:
        Groups where either condition has < N cells are skipped.
    log2fc_thresh:
        Threshold for significance classification.
    pval_thresh:
        Adjusted p-value threshold for significance classification.
    output_dir:
        If provided, saves CSVs here.

    Returns
    -------
    Long-format DataFrame with all results across all groups:
        group, gene, log2fc, padj, significant, direction
    """
    from src.dge_analysis import run_dge

    # Remove zero-filled custom genes once before the cluster loop.
    # Each cluster subset would otherwise re-apply the masking redundantly,
    # and the zero_filled flags are consistent across all clusters.
    if "zero_filled" in adata.var.columns:
        keep = ~adata.var["zero_filled"].fillna(False)
        n_excl = int((~keep).sum())
        if n_excl > 0:
            logger.info(
                "run_cluster_dge: excluding %d zero-filled custom gene(s) "
                "from all cluster-level tests.",
                n_excl,
            )
            adata = adata[:, keep].copy()

    groups = sorted(adata.obs[group_key].unique(), key=lambda x: (0, int(x)) if str(x).lstrip('-').isdigit() else (1, str(x)))
    conditions = adata.obs[condition_key].unique().tolist()

    if condition_a is None or condition_b is None:
        if len(conditions) != 2:
            raise ValueError(
                f"Found {len(conditions)} conditions; specify condition_a/b."
            )
        condition_a, condition_b = sorted(conditions)

    all_results = []
    skipped = []

    for grp in groups:
        mask = adata.obs[group_key] == grp
        sub = adata[mask].copy()

        # Cell count guard
        n_a = (sub.obs[condition_key] == condition_a).sum()
        n_b = (sub.obs[condition_key] == condition_b).sum()
        if n_a < min_cells_per_condition or n_b < min_cells_per_condition:
            logger.info(
                "Skipping group '%s': n_%s=%d, n_%s=%d (min=%d)",
                grp, condition_a, n_a, condition_b, n_b, min_cells_per_condition
            )
            skipped.append(grp)
            continue

        logger.info(
            "DGE for group '%s': %d vs %d cells", grp, n_a, n_b
        )

        try:
            res = run_dge(
                sub,
                method=method,
                condition_key=condition_key,
                condition_a=condition_a,
                condition_b=condition_b,
                replicate_key=replicate_key,
                # Forward the user's thresholds so stringent_wilcoxon's internal
                # filters use the same values as the post-hoc significance flags.
                lfc_threshold=log2fc_thresh,
                pval_threshold=pval_thresh,
            )
        except Exception as exc:
            logger.warning("DGE failed for group '%s': %s", grp, exc)
            skipped.append(grp)
            continue

        # run_dge() guarantees canonical column names: 'log2fc' and 'pval_adj'.
        res.insert(0, "group", str(grp))
        res["n_cells_a"] = int(n_a)
        res["n_cells_b"] = int(n_b)

        # Significance flag — uses canonical column names guaranteed by run_dge()
        res["significant"] = (
            (res["pval_adj"] < pval_thresh)
            & (res["log2fc"].abs() >= log2fc_thresh)
        )
        res["direction"] = "ns"
        res.loc[res["significant"] & (res["log2fc"] > 0), "direction"] = "up"
        res.loc[res["significant"] & (res["log2fc"] < 0), "direction"] = "down"

        all_results.append(res)

    if not all_results:
        logger.warning("No groups produced DGE results.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Summary table
    summary = (
        combined.groupby("group", observed=True)
        .apply(
            lambda df: pd.Series({
                "n_up"   : (df["direction"] == "up").sum(),
                "n_down" : (df["direction"] == "down").sum(),
                "n_total": df["significant"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    logger.info("Per-cluster DGE summary:")
    for _, row in summary.iterrows():
        logger.info(
            "  %-25s  up=%3d  down=%3d  total=%3d",
            row["group"], row["n_up"], row["n_down"], row["n_total"],
        )

    if skipped:
        logger.info("Skipped groups: %s", skipped)

    # Save
    if output_dir is not None:
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out / "cluster_dge_results.csv", index=False)
        summary.to_csv(out / "cluster_dge_summary.csv", index=False)
        logger.info("Saved cluster DGE results to %s/", out)

    combined.attrs["summary"] = summary
    return combined


def top_genes_per_group(
    cluster_dge: pd.DataFrame,
    n: int = 5,
    direction: str = "both",
) -> pd.DataFrame:
    """
    Extract the top N DEGs per group.

    Parameters
    ----------
    cluster_dge:
        Output of run_cluster_dge.
    n:
        Number of genes per group per direction.
    direction:
        'up' | 'down' | 'both' (both returns top N up + N down).

    Returns
    -------
    Filtered DataFrame.
    """
    sig = cluster_dge[cluster_dge["significant"]].copy()
    # Clear .attrs to avoid pandas >= 2.1 bug where concat fails when
    # DataFrame.attrs contains non-scalar values (e.g. another DataFrame).
    sig.attrs = {}
    rows = []
    for grp, gdf in sig.groupby("group", observed=True):
        gdf = gdf.copy()
        gdf.attrs = {}
        if direction in ("up", "both"):
            sub = gdf[gdf["direction"] == "up"].copy()
            sub.attrs = {}
            rows.append(sub.nlargest(n, "log2fc"))
        if direction in ("down", "both"):
            sub = gdf[gdf["direction"] == "down"].copy()
            sub.attrs = {}
            rows.append(sub.nsmallest(n, "log2fc"))
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result.attrs = {}
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
