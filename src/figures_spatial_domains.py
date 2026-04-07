"""
figures_spatial_domains.py
--------------------------
Nature-grade figures for spatial domain detection results.

Figures produced:
  Fig SD1: Spatial domain map — domains overlaid on tissue coordinates
  Fig SD2: Domain composition — bar chart of cell counts per domain
  Fig SD3: Domain marker genes — dot plot of top SVGs per domain
  Fig SD4: Lambda sweep — combined score vs lambda_spatial
  Fig SD5: Domain comparison — spatial domains vs Leiden clusters side-by-side
"""

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.figures import (
    SINGLE, DOUBLE, WONG,
    CONDITION_COLOURS,
    apply_nature_style, _savefig,
    _safe_cluster_sort_key,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain_palette(n: int) -> list:
    """Distinct colour palette for spatial domains."""
    if n <= 8:
        return WONG[:n]
    if n <= 20:
        cmap = mpl.colormaps.get_cmap("tab20")
        return [mcolors.to_hex(cmap(i / 20)) for i in range(n)]
    cmap = mpl.colormaps.get_cmap("gist_ncar")
    return [mcolors.to_hex(cmap(i / n)) for i in range(n)]


def get_domain_colours(adata: ad.AnnData, domain_key: str = "spatial_domain") -> dict:
    """Return canonical {domain_id: hex_colour} mapping."""
    domains = sorted(
        adata.obs[domain_key].unique().tolist(), key=_safe_cluster_sort_key
    )
    pal = _domain_palette(len(domains))
    return {d: pal[i] for i, d in enumerate(domains)}


# ===========================================================================
# Fig SD1: Spatial domain map
# ===========================================================================

def plot_spatial_domains(
    adata: ad.AnnData,
    domain_key: str = "spatial_domain",
    condition_key: str = "condition",
    output_dir: Path = Path("figures_output"),
    spot_size: float = 3.0,
    fmt: str = "pdf",
    dpi: int = 300,
    representative_slides: Optional[dict] = None,
) -> Optional[Path]:
    """
    Spatial scatter coloured by spatial domain — one panel per slide or
    one representative per condition.
    """
    apply_nature_style()
    if "spatial" not in adata.obsm:
        logger.warning("No spatial coordinates; skipping fig_spatial_domains.")
        return None

    domain_colours = get_domain_colours(adata, domain_key)
    domains = sorted(domain_colours.keys(), key=_safe_cluster_sort_key)

    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
    conditions = adata.obs[condition_key].astype("category").cat.categories.tolist()

    # Determine slides to plot
    if slide_col:
        slides = adata.obs[slide_col].unique().tolist()
    else:
        slides = conditions

    n_slides = len(slides)
    ncols = min(4, n_slides)
    nrows = (n_slides + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(DOUBLE, 2.2 * nrows),
        squeeze=False,
    )

    for idx, slide in enumerate(slides):
        ax = axes[idx // ncols][idx % ncols]
        if slide_col:
            mask = adata.obs[slide_col] == slide
        else:
            mask = adata.obs[condition_key] == slide

        sub = adata[mask]
        xy = sub.obsm["spatial"]
        labels = sub.obs[domain_key].values

        for domain in domains:
            dm = labels == domain
            if dm.sum() == 0:
                continue
            ax.scatter(
                xy[dm, 0], xy[dm, 1],
                c=domain_colours[domain],
                s=spot_size, alpha=0.6, edgecolors="none",
                rasterized=True,
            )
        ax.set_title(str(slide), fontsize=7)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    # Turn off unused axes
    for idx in range(n_slides, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # Legend
    handles = [
        mpatches.Patch(color=domain_colours[d], label=f"Domain {d}")
        for d in domains
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(8, len(domains)), fontsize=5,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Spatial Domains", fontsize=8, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = _savefig(fig, output_dir / "fig_sd1_spatial_domains", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Fig SD2: Domain composition bar chart
# ===========================================================================

def plot_domain_composition(
    adata: ad.AnnData,
    domain_key: str = "spatial_domain",
    condition_key: str = "condition",
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """Stacked bar chart of cell counts per domain, split by condition."""
    apply_nature_style()

    df = adata.obs[[domain_key, condition_key]].copy()
    ct = pd.crosstab(df[domain_key], df[condition_key])
    ct = ct.loc[sorted(ct.index, key=_safe_cluster_sort_key)]

    conditions = ct.columns.tolist()
    cond_colors = [CONDITION_COLOURS.get(c, WONG[i % len(WONG)]) for i, c in enumerate(conditions)]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))

    # Absolute counts
    ct.plot.bar(ax=axes[0], color=cond_colors, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Spatial Domain")
    axes[0].set_ylabel("Cell count")
    axes[0].set_title("a  Domain cell counts", fontsize=8, fontweight="bold", loc="left")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=5, frameon=False)

    # Proportions
    ct_prop = ct.div(ct.sum(axis=1), axis=0)
    ct_prop.plot.bar(stacked=True, ax=axes[1], color=cond_colors, edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("Spatial Domain")
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("b  Condition proportions", fontsize=8, fontweight="bold", loc="left")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=5, frameon=False)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    out = _savefig(fig, output_dir / "fig_sd2_domain_composition", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Fig SD3: Domain marker genes dot plot
# ===========================================================================

def plot_domain_markers(
    adata: ad.AnnData,
    deg_df: pd.DataFrame,
    domain_key: str = "spatial_domain",
    n_genes: int = 5,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Dot plot of top marker genes per spatial domain. Size = fraction
    expressing, colour = mean expression.
    """
    apply_nature_style()

    if deg_df is None or deg_df.empty:
        logger.warning("No domain DEG results; skipping fig_sd3.")
        return None

    # Select top genes per domain
    top_genes = []
    for domain in sorted(deg_df["domain"].unique(), key=_safe_cluster_sort_key):
        sub = deg_df[
            (deg_df["domain"] == domain)
            & (deg_df["pval_adj"] < 0.05)
            & (deg_df["log2fc"] > 0)
        ].nlargest(n_genes, "log2fc")
        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))  # deduplicate, preserve order
    if not top_genes:
        logger.warning("No significant domain markers found; skipping fig_sd3.")
        return None

    # Limit to manageable number
    top_genes = top_genes[:40]

    import scanpy as sc
    sc.pl.dotplot(
        adata, var_names=top_genes, groupby=domain_key,
        standard_scale="var", show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(DOUBLE, max(2.0, 0.25 * len(top_genes)))
    fig.suptitle("Spatial Domain Marker Genes", fontsize=8, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = _savefig(fig, output_dir / "fig_sd3_domain_markers", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Fig SD4: Lambda sweep results
# ===========================================================================

def plot_lambda_sweep(
    sweep_df: pd.DataFrame,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """Line plots of spatial coherence, silhouette, and combined score vs lambda."""
    apply_nature_style()

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))

    lam = sweep_df["lambda"]

    # Spatial coherence
    axes[0].plot(lam, sweep_df["spatial_coherence"], "o-", color=WONG[1], markersize=4, linewidth=1)
    axes[0].set_xlabel(r"$\lambda_{\mathrm{spatial}}$")
    axes[0].set_ylabel("Spatial coherence")
    axes[0].set_title("a  Spatial coherence", fontsize=8, fontweight="bold", loc="left")

    # Silhouette
    axes[1].plot(lam, sweep_df["silhouette_score"], "s-", color=WONG[2], markersize=4, linewidth=1)
    axes[1].set_xlabel(r"$\lambda_{\mathrm{spatial}}$")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("b  Silhouette score", fontsize=8, fontweight="bold", loc="left")

    # Combined
    axes[2].plot(lam, sweep_df["combined_score"], "D-", color=WONG[5], markersize=4, linewidth=1)
    best_idx = sweep_df["combined_score"].idxmax()
    axes[2].axvline(sweep_df.loc[best_idx, "lambda"], color="#999", ls="--", lw=0.6)
    axes[2].set_xlabel(r"$\lambda_{\mathrm{spatial}}$")
    axes[2].set_ylabel("Combined score")
    axes[2].set_title("c  Combined score", fontsize=8, fontweight="bold", loc="left")

    fig.tight_layout()
    out = _savefig(fig, output_dir / "fig_sd4_lambda_sweep", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Fig SD5: Domains vs Leiden clusters comparison
# ===========================================================================

def plot_domain_vs_leiden(
    adata: ad.AnnData,
    domain_key: str = "spatial_domain",
    cluster_key: str = "leiden",
    condition_key: str = "condition",
    output_dir: Path = Path("figures_output"),
    spot_size: float = 3.0,
    fmt: str = "pdf",
    dpi: int = 300,
    representative_slides: Optional[dict] = None,
) -> Optional[Path]:
    """
    Side-by-side spatial scatter: Leiden clusters vs spatial domains for
    one representative slide per condition.
    """
    apply_nature_style()
    if "spatial" not in adata.obsm:
        logger.warning("No spatial coordinates; skipping fig_sd5.")
        return None

    conditions = adata.obs[condition_key].astype("category").cat.categories.tolist()
    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None

    # Pick representative slides
    slides = {}
    for cond in conditions:
        cond_mask = adata.obs[condition_key] == cond
        if representative_slides and cond in representative_slides:
            slides[cond] = representative_slides[cond]
        elif slide_col:
            slides[cond] = adata.obs.loc[cond_mask, slide_col].iloc[0]
        else:
            slides[cond] = cond

    n_cond = len(conditions)
    fig, axes = plt.subplots(n_cond, 2, figsize=(DOUBLE, 2.5 * n_cond), squeeze=False)

    domain_colours = get_domain_colours(adata, domain_key)
    from src.figures import get_cluster_colours
    cluster_colours = get_cluster_colours(adata, cluster_key)

    for row, cond in enumerate(conditions):
        sid = slides[cond]
        if slide_col:
            mask = adata.obs[slide_col] == sid
        else:
            mask = adata.obs[condition_key] == cond

        sub = adata[mask]
        xy = sub.obsm["spatial"]

        # Leiden clusters (left)
        ax_l = axes[row][0]
        for cl in sorted(cluster_colours.keys(), key=_safe_cluster_sort_key):
            cm = sub.obs[cluster_key].values == cl
            if cm.sum() == 0:
                continue
            ax_l.scatter(
                xy[cm, 0], xy[cm, 1],
                c=cluster_colours[cl], s=spot_size, alpha=0.6,
                edgecolors="none", rasterized=True,
            )
        ax_l.set_title(f"Leiden — {sid}", fontsize=7)
        ax_l.set_aspect("equal")
        ax_l.invert_yaxis()
        ax_l.axis("off")

        # Spatial domains (right)
        ax_r = axes[row][1]
        for dom in sorted(domain_colours.keys(), key=_safe_cluster_sort_key):
            dm = sub.obs[domain_key].values == dom
            if dm.sum() == 0:
                continue
            ax_r.scatter(
                xy[dm, 0], xy[dm, 1],
                c=domain_colours[dom], s=spot_size, alpha=0.6,
                edgecolors="none", rasterized=True,
            )
        ax_r.set_title(f"Spatial Domains — {sid}", fontsize=7)
        ax_r.set_aspect("equal")
        ax_r.invert_yaxis()
        ax_r.axis("off")

    fig.suptitle(
        "Expression Clusters vs Spatial Domains",
        fontsize=8, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = _savefig(fig, output_dir / "fig_sd5_domain_vs_leiden", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out
