"""
galanin_resistance.py
---------------------
Core analysis functions for studying galanin signalling resistance
in the ageing brain.

Hypothesis: ageing leads to elevated Gal ligand expression but reduced
or altered receptor expression (Galr1, Galr3), creating functional
resistance to galanin signalling.

Analyses:
  1. Resistance index     -- Gal / (Galr1 + Galr3 + 1) per cell
  2. Co-expression status -- classify cells as co-expressing, single-positive, or negative
  3. Niche receptor score -- fraction of spatial neighbours expressing receptors
  4. Ligand-receptor proximity -- nearest-neighbour distances between Gal+ and receptor+ cells

All functions accept an AnnData with .obsm['spatial'] and log-normalised .X.
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, ranksums

logger = logging.getLogger(__name__)

# Canonical gene names (mouse Xenium panel nomenclature)
GAL_GENE = "Gal"
GALR1_GENE = "Galr1"
GALR3_GENE = "Galr3"
GAL_SYSTEM_GENES = [GAL_GENE, GALR1_GENE, GALR3_GENE]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_expr_vector(adata: ad.AnnData, gene: str) -> np.ndarray:
    """Return dense 1-D expression vector for *gene* from lognorm layer."""
    if gene not in adata.var_names:
        return np.zeros(adata.n_obs, dtype=np.float32)
    idx = adata.var_names.get_loc(gene)
    X = adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X
    col = X[:, idx]
    if sp.issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel().astype(np.float32)


def _check_genes(adata: ad.AnnData) -> dict[str, bool]:
    """Return dict of gene -> present-in-panel."""
    return {g: g in adata.var_names for g in GAL_SYSTEM_GENES}


# ---------------------------------------------------------------------------
# 1. Galanin resistance index
# ---------------------------------------------------------------------------

def compute_resistance_index(
    adata: ad.AnnData,
    store_in_obs: bool = True,
) -> np.ndarray:
    """
    Compute per-cell galanin resistance index:
        GRI = Gal / (Galr1 + Galr3 + 1)

    Higher values indicate more ligand relative to receptor expression,
    i.e. greater potential for functional resistance.

    Parameters
    ----------
    adata : AnnData
        Log-normalised AnnData.
    store_in_obs : bool
        If True, stores result in adata.obs['galanin_resistance_index'].

    Returns
    -------
    1-D array of resistance index values (one per cell).
    """
    gal = _get_expr_vector(adata, GAL_GENE)
    galr1 = _get_expr_vector(adata, GALR1_GENE)
    galr3 = _get_expr_vector(adata, GALR3_GENE)

    gri = gal / (galr1 + galr3 + 1.0)

    if store_in_obs:
        adata.obs["galanin_resistance_index"] = gri

    present = _check_genes(adata)
    missing = [g for g, v in present.items() if not v]
    if missing:
        logger.warning(
            "Resistance index: genes %s not in panel (treated as zero).", missing
        )

    logger.info(
        "Resistance index: mean=%.3f, median=%.3f, max=%.3f",
        float(np.mean(gri)), float(np.median(gri)), float(np.max(gri)),
    )
    return gri


# ---------------------------------------------------------------------------
# 2. Co-expression classification
# ---------------------------------------------------------------------------

def classify_coexpression(
    adata: ad.AnnData,
    threshold: float = 0.0,
    store_in_obs: bool = True,
) -> pd.Series:
    """
    Classify each cell's galanin co-expression status.

    Categories:
      - Gal+Galr1+  : co-expresses Gal and Galr1 (± Galr3)
      - Gal+Galr3+  : co-expresses Gal and Galr3 (but not Galr1)
      - Gal+only    : expresses Gal but neither receptor
      - Galr1+only  : expresses Galr1 only
      - Galr3+only  : expresses Galr3 only
      - Galr1+Galr3+: co-expresses both receptors but not Gal
      - Negative     : none of the three

    Parameters
    ----------
    adata : AnnData
    threshold : float
        Expression threshold for calling a gene "positive" (on log-normalised
        scale). Default 0 means any non-zero expression counts.
    store_in_obs : bool
        If True, stores in adata.obs['gal_coexpr_status'].

    Returns
    -------
    Categorical Series with co-expression labels.
    """
    gal = _get_expr_vector(adata, GAL_GENE) > threshold
    galr1 = _get_expr_vector(adata, GALR1_GENE) > threshold
    galr3 = _get_expr_vector(adata, GALR3_GENE) > threshold

    labels = np.full(adata.n_obs, "Negative", dtype=object)

    # Single-positive
    labels[gal & ~galr1 & ~galr3] = "Gal+only"
    labels[~gal & galr1 & ~galr3] = "Galr1+only"
    labels[~gal & ~galr1 & galr3] = "Galr3+only"

    # Double-positive (no Gal)
    labels[~gal & galr1 & galr3] = "Galr1+Galr3+"

    # Co-expression with Gal
    labels[gal & galr1] = "Gal+Galr1+"
    labels[gal & galr3 & ~galr1] = "Gal+Galr3+"
    # Triple positive falls under Gal+Galr1+ (already set above since galr1 is True)

    result = pd.Categorical(
        labels,
        categories=[
            "Gal+Galr1+", "Gal+Galr3+", "Gal+only",
            "Galr1+only", "Galr3+only", "Galr1+Galr3+", "Negative",
        ],
    )
    series = pd.Series(result, index=adata.obs.index)

    if store_in_obs:
        adata.obs["gal_coexpr_status"] = series

    # Log summary
    counts = series.value_counts()
    logger.info("Co-expression classification:\n%s", counts.to_string())
    return series


def coexpression_proportions(
    adata: ad.AnnData,
    condition_key: str = "condition",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Quantify the proportion of Gal+ cells that co-express Galr1 or Galr3,
    broken down by condition.

    Returns
    -------
    DataFrame with columns: condition, n_gal_pos, n_gal_galr1, n_gal_galr3,
    pct_galr1, pct_galr3.
    """
    gal = _get_expr_vector(adata, GAL_GENE) > threshold
    galr1 = _get_expr_vector(adata, GALR1_GENE) > threshold
    galr3 = _get_expr_vector(adata, GALR3_GENE) > threshold
    conditions = adata.obs[condition_key].values

    rows = []
    for cond in sorted(set(conditions)):
        mask = conditions == cond
        n_gal = int((gal & mask).sum())
        # n_galr1/n_galr3 may overlap (triple-positive cells);
        # n_any_receptor uses union to avoid double-counting in stacked bars
        n_galr1_any = int((gal & galr1 & mask).sum())
        n_galr3_any = int((gal & galr3 & mask).sum())
        n_both_rec = int((gal & galr1 & galr3 & mask).sum())
        n_any_rec = int((gal & (galr1 | galr3) & mask).sum())
        rows.append({
            "condition": cond,
            "n_gal_pos": n_gal,
            "n_gal_galr1": n_galr1_any,
            "n_gal_galr3": n_galr3_any,
            "n_gal_both_receptors": n_both_rec,
            "n_gal_any_receptor": n_any_rec,
            "pct_galr1": 100.0 * n_galr1_any / max(n_gal, 1),
            "pct_galr3": 100.0 * n_galr3_any / max(n_gal, 1),
            "pct_any_receptor": 100.0 * n_any_rec / max(n_gal, 1),
        })

    df = pd.DataFrame(rows)
    logger.info("Gal+ co-expression proportions:\n%s", df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 3. Niche receptor availability
# ---------------------------------------------------------------------------

def niche_receptor_score(
    adata: ad.AnnData,
    k: int = 15,
    threshold: float = 0.0,
    store_in_obs: bool = True,
) -> np.ndarray:
    """
    For each Gal+ cell, compute the fraction of its k spatial neighbours
    that express Galr1 or Galr3 ("receptor availability in niche").

    For non-Gal+ cells, the score is NaN.

    Parameters
    ----------
    adata : AnnData
        Must have .obsm['spatial'].
    k : int
        Number of spatial nearest neighbours.
    threshold : float
        Expression threshold for positivity.
    store_in_obs : bool
        If True, stores in adata.obs['niche_receptor_score'].

    Returns
    -------
    1-D array (n_cells,) with receptor niche scores (NaN for non-Gal+ cells).
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] required.")

    gal = _get_expr_vector(adata, GAL_GENE) > threshold
    galr1 = _get_expr_vector(adata, GALR1_GENE) > threshold
    galr3 = _get_expr_vector(adata, GALR3_GENE) > threshold
    receptor_pos = galr1 | galr3

    xy = adata.obsm["spatial"].astype(np.float64)
    _k = min(k, len(xy) - 1)
    tree = cKDTree(xy)
    _, nbr_idx = tree.query(xy, k=_k + 1)
    nbr_idx = nbr_idx[:, 1:]  # exclude self

    scores = np.full(adata.n_obs, np.nan, dtype=np.float64)
    gal_indices = np.where(gal)[0]

    if len(gal_indices) > 0:
        # Vectorised: look up all neighbour indices at once
        gal_nbrs = nbr_idx[gal_indices]          # (n_gal, _k)
        scores[gal_indices] = receptor_pos[gal_nbrs].sum(axis=1) / _k

    if store_in_obs:
        adata.obs["niche_receptor_score"] = scores

    n_gal = len(gal_indices)
    valid = scores[~np.isnan(scores)]
    logger.info(
        "Niche receptor score: %d Gal+ cells, mean=%.3f, median=%.3f",
        n_gal,
        float(np.mean(valid)) if len(valid) > 0 else 0.0,
        float(np.median(valid)) if len(valid) > 0 else 0.0,
    )
    return scores


# ---------------------------------------------------------------------------
# 4. Ligand-receptor spatial proximity
# ---------------------------------------------------------------------------

def ligand_receptor_distances(
    adata: ad.AnnData,
    threshold: float = 0.0,
    condition_key: str = "condition",
) -> dict[str, np.ndarray]:
    """
    Compute nearest-neighbour distances from each Gal+ cell to the closest
    Galr1+ cell and closest Galr3+ cell.

    Parameters
    ----------
    adata : AnnData
        Must have .obsm['spatial'].
    threshold : float
        Expression threshold for positivity.
    condition_key : str
        obs column with condition labels.

    Returns
    -------
    Dict with keys:
      'gal_to_galr1' : 1-D array of distances (one per Gal+ cell)
      'gal_to_galr3' : 1-D array of distances (one per Gal+ cell)
      'gal_cell_idx' : indices of Gal+ cells in adata
      'condition'     : condition label for each Gal+ cell
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] required.")

    gal = _get_expr_vector(adata, GAL_GENE) > threshold
    galr1 = _get_expr_vector(adata, GALR1_GENE) > threshold
    galr3 = _get_expr_vector(adata, GALR3_GENE) > threshold

    xy = adata.obsm["spatial"].astype(np.float64)
    gal_idx = np.where(gal)[0]

    result = {
        "gal_cell_idx": gal_idx,
        "condition": adata.obs[condition_key].values[gal_idx] if condition_key in adata.obs.columns else np.full(len(gal_idx), "unknown"),
    }

    for name, receptor_mask in [("gal_to_galr1", galr1), ("gal_to_galr3", galr3)]:
        rec_idx = np.where(receptor_mask)[0]
        if len(rec_idx) == 0 or len(gal_idx) == 0:
            result[name] = np.full(len(gal_idx), np.nan)
            logger.warning("No %s cells found for distance computation.", name.split("_to_")[1])
            continue
        tree = cKDTree(xy[rec_idx])
        dists, _ = tree.query(xy[gal_idx], k=1)
        result[name] = dists

    logger.info(
        "Ligand-receptor distances: %d Gal+ cells, "
        "median dist to Galr1+=%.1f µm, to Galr3+=%.1f µm",
        len(gal_idx),
        float(np.nanmedian(result["gal_to_galr1"])),
        float(np.nanmedian(result["gal_to_galr3"])),
    )
    return result


# ---------------------------------------------------------------------------
# 5. Per-region breakdown (uses cell_type or leiden clusters)
# ---------------------------------------------------------------------------

def regional_expression_summary(
    adata: ad.AnnData,
    region_key: str = "cell_type",
    condition_key: str = "condition",
) -> pd.DataFrame:
    """
    Compute mean expression of Gal, Galr1, Galr3 per region × condition,
    plus the mean resistance index.

    Falls back to 'leiden' if region_key is not in obs.

    Returns
    -------
    Long-form DataFrame: region, condition, gene, mean_expr, pct_expressing,
    n_cells, resistance_index_mean.
    """
    if region_key not in adata.obs.columns:
        region_key = "leiden" if "leiden" in adata.obs.columns else None
    if region_key is None:
        logger.warning("No region/cluster key found; skipping regional summary.")
        return pd.DataFrame()

    # Ensure resistance index is computed
    if "galanin_resistance_index" not in adata.obs.columns:
        compute_resistance_index(adata, store_in_obs=True)

    rows = []
    for region in sorted(adata.obs[region_key].dropna().unique()):
        for cond in sorted(adata.obs[condition_key].unique()):
            mask = (adata.obs[region_key] == region) & (adata.obs[condition_key] == cond)
            n = int(mask.sum())
            if n == 0:
                continue
            sub = adata[mask.values]
            gri = sub.obs["galanin_resistance_index"].values

            for gene in GAL_SYSTEM_GENES:
                expr = _get_expr_vector(sub, gene)
                rows.append({
                    "region": region,
                    "condition": cond,
                    "gene": gene,
                    "mean_expr": float(expr.mean()),
                    "pct_expressing": float((expr > 0).sum()) / n * 100,
                    "n_cells": n,
                    "resistance_index_mean": float(gri.mean()),
                })

    df = pd.DataFrame(rows)
    logger.info("Regional summary: %d region x condition x gene entries.", len(df))
    return df


# ---------------------------------------------------------------------------
# 6. Statistical tests (convenience wrappers)
# ---------------------------------------------------------------------------

def test_condition_difference(
    values: np.ndarray,
    conditions: np.ndarray,
    cond_a: str = "ADULT",
    cond_b: str = "AGED",
) -> dict:
    """Wilcoxon rank-sum test comparing values between two conditions."""
    va = values[conditions == cond_a]
    vb = values[conditions == cond_b]
    va = va[~np.isnan(va)]
    vb = vb[~np.isnan(vb)]
    if len(va) < 3 or len(vb) < 3:
        return {"stat": np.nan, "pval": 1.0, "n_a": len(va), "n_b": len(vb)}
    stat, pval = ranksums(vb, va)
    return {"stat": float(stat), "pval": float(pval), "n_a": len(va), "n_b": len(vb)}


def test_coexpression_proportion(
    adata: ad.AnnData,
    condition_key: str = "condition",
    threshold: float = 0.0,
) -> dict:
    """
    Chi-squared test for whether the proportion of Gal+ cells co-expressing
    receptors differs between conditions.

    Returns dict with chi2, p-value, and the contingency table.
    """
    gal = _get_expr_vector(adata, GAL_GENE) > threshold
    receptor = (
        (_get_expr_vector(adata, GALR1_GENE) > threshold)
        | (_get_expr_vector(adata, GALR3_GENE) > threshold)
    )
    conditions = adata.obs[condition_key].values
    cond_labels = sorted(set(conditions))

    if len(cond_labels) < 2:
        return {"chi2": np.nan, "pval": 1.0, "table": None}

    # Build 2x2 contingency: (Gal+receptor+ vs Gal+receptor-) x condition
    table = []
    for cond in cond_labels:
        mask = (conditions == cond) & gal
        n_coexpr = int((mask & receptor).sum())
        n_single = int((mask & ~receptor).sum())
        table.append([n_coexpr, n_single])

    table = np.array(table)
    if table.min() < 1:
        return {"chi2": np.nan, "pval": 1.0, "table": table}

    chi2, pval, _, _ = chi2_contingency(table)
    return {"chi2": float(chi2), "pval": float(pval), "table": table}
