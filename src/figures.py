"""
figures.py
----------
Nature-grade figure generation for the Xenium DGE pipeline.

All figures follow Nature Publishing Group style guidelines:
  - Single column: 3.50 in (89 mm)
  - Double column: 7.20 in (183 mm)
  - Font: Arial, 7 pt body / 8 pt titles / 9 pt panel labels
  - 300 DPI minimum for print
  - Colour-blind-safe palettes (Wong 2011)
  - White background, minimal gridlines, no chartjunk

Figures produced:
  Fig 1: QC violin plots (counts, genes, spatial density)
  Fig 2: Integration overview (UMAP x condition, x cluster)
  Fig 3: Spatial map of clusters on tissue section
  Fig 4: Marker gene dot plot per cluster
  Fig 5: DGE volcano plot (condition A vs B)
  Fig 6: DGE heatmap (top N genes x cells, split by condition)
  Fig 7: Spatial expression maps for top DGE genes
  Fig 8: Summary panel (multi-panel composite figure)
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Nature style settings
# ---------------------------------------------------------------------------
NATURE_RC = {
    # Font
    "font.family"           : "Arial",
    "font.size"             : 7,
    "axes.titlesize"        : 8,
    "axes.labelsize"        : 7,
    "xtick.labelsize"       : 6,
    "ytick.labelsize"       : 6,
    "legend.fontsize"       : 6,
    "legend.title_fontsize" : 7,
    # Lines
    "axes.linewidth"        : 0.5,
    "xtick.major.width"     : 0.5,
    "ytick.major.width"     : 0.5,
    "xtick.minor.width"     : 0.4,
    "ytick.minor.width"     : 0.4,
    "xtick.major.size"      : 2.5,
    "ytick.major.size"      : 2.5,
    "lines.linewidth"       : 0.8,
    # Layout
    "axes.spines.top"       : False,
    "axes.spines.right"     : False,
    "axes.grid"             : False,
    "figure.dpi"            : 300,
    "savefig.dpi"           : 300,
    "savefig.bbox"          : "tight",
    "savefig.pad_inches"    : 0.02,
    "pdf.fonttype"          : 42,  # editable text in PDF/Illustrator
    "ps.fonttype"           : 42,
}

SINGLE = 3.50   # inches
DOUBLE = 7.20   # inches
WONG   = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]
CONDITION_COLOURS = {
    "Control"  : "#0072B2",
    "Treatment": "#D55E00",
    "ADULT"    : "#0072B2",
    "AGED"     : "#D55E00",
}

# Colour map constants (imported by figures_extended.py and figures_panel.py)
DIVERGING_CMAP  = "RdBu_r"
SEQUENTIAL_CMAP = "Reds"
SPATIAL_CMAP    = "magma"   # legacy — fig7 uses inline grey_red cmap; update per-figure as needed


def apply_nature_style():
    """Apply Nature RC params globally."""
    mpl.rcParams.update(NATURE_RC)


def _savefig(fig: plt.Figure, path: Path, fmt: str = "pdf", dpi: int = 300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, format=fmt, dpi=dpi, transparent=False)
    logger.info("Saved: %s", out)
    return out


def _panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.05):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top", ha="left",
    )


def _safe_cluster_sort_key(x):
    """Sort cluster labels numerically if possible, otherwise alphabetically."""
    try:
        return (0, int(x))
    except (ValueError, TypeError):
        return (1, str(x))


def _cluster_palette(n: int) -> list[str]:
    """Generate a discrete palette for n clusters."""
    if n <= len(WONG):
        return WONG[:n]
    cmap = mpl.colormaps.get_cmap("tab20")
    return [mcolors.to_hex(cmap(i / n)) for i in range(n)]


def get_cluster_colours(adata: "ad.AnnData", cluster_key: str = "leiden") -> dict:
    """
    Return a canonical {cluster_id: hex_colour} dict, sorted numerically where
    possible.  Call this ONCE per figure and pass the dict to every panel that
    needs it — this guarantees identical colours across Fig 2b, Fig 3, and Fig 8.
    """
    clusters = sorted(adata.obs[cluster_key].unique(), key=_safe_cluster_sort_key)
    pal      = _cluster_palette(len(clusters))
    return dict(zip(clusters, pal))


# Canonical cell type palette — 20 distinguishable colours.
# Used by every figure that colours by cell type (Fig 9, 14, 15, 16, 17, …).
# Import this palette (or call get_cell_type_colours) everywhere to guarantee
# the same cell type ↔ colour mapping across all figures.
CELL_TYPE_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#D37295", "#A0CBE8", "#FF6F61", "#6B5B95", "#88B04B",
    "#F7CAC9", "#92A8D1", "#955251", "#B565A7", "#009B77",
]


def get_cell_type_colours(
    adata: "ad.AnnData",
    cell_type_key: str = "cell_type",
) -> dict:
    """
    Return a canonical {cell_type: hex_colour} dict, sorted alphabetically.

    Uses CELL_TYPE_PALETTE so every figure that shows cell types gets the
    same colour mapping.  Falls back to tab20 if >20 cell types.
    """
    cell_types = sorted(adata.obs[cell_type_key].dropna().unique(),
                        key=_safe_cluster_sort_key)
    n = len(cell_types)
    if n <= len(CELL_TYPE_PALETTE):
        pal = CELL_TYPE_PALETTE[:n]
    else:
        cmap = mpl.colormaps.get_cmap("tab20")
        pal = [mcolors.to_hex(cmap(i / n)) for i in range(n)]
    return dict(zip(cell_types, pal))



def _get_lognorm(adata: ad.AnnData) -> np.ndarray:
    if "lognorm" in adata.layers:
        X = adata.layers["lognorm"]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return X


# ===========================================================================
# Figure 1: QC violin plots
# ===========================================================================

def plot_qc(
    adata: ad.AnnData,
    condition_key: str = "condition",
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Three-panel QC figure:
      A: total transcript counts per cell (violin, split by condition)
      B: number of genes per cell (violin, split by condition)
      C: cells detected per slide (dot/bar chart coloured by condition)
    """
    apply_nature_style()

    # Ensure QC columns exist (they may be absent if adata was loaded from a
    # cache that pre-dates run_qc, or if the caller skipped the QC step).
    if "total_counts" not in adata.obs.columns or "n_genes_by_counts" not in adata.obs.columns:
        import scanpy as sc
        logger.info("plot_qc: QC metrics missing from obs; computing them now.")
        sc.pp.calculate_qc_metrics(
            adata, expr_type="counts", percent_top=None, log1p=False, inplace=True
        )

    fig = plt.figure(figsize=(DOUBLE, 2.4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.50)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    conditions = adata.obs[condition_key].astype("category").cat.categories.tolist()
    palette = {c: CONDITION_COLOURS.get(c, WONG[i]) for i, c in enumerate(conditions)}

    # Panel A: total counts per cell
    _violin_split(ax_a, adata, "total_counts", condition_key, palette)
    ax_a.set_ylabel("Total counts per cell")
    ax_a.set_xlabel("")
    _panel_label(ax_a, "a")

    # Panel B: genes detected per cell
    _violin_split(ax_b, adata, "n_genes_by_counts", condition_key, palette)
    ax_b.set_ylabel("Genes detected per cell")
    ax_b.set_xlabel("")
    _panel_label(ax_b, "b")

    # Panel C: cells per slide (dot plot coloured by condition)
    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
    if slide_col is not None:
        counts = (
            adata.obs.groupby([condition_key, slide_col], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        # Order slides within each condition alphabetically
        counts = counts.sort_values([condition_key, slide_col]).reset_index(drop=True)
        counts["x"] = range(len(counts))

        for _, row in counts.iterrows():
            col = palette.get(row[condition_key], "#888888")
            ax_c.scatter(
                row["x"], row["n_cells"],
                color=col, s=60, zorder=3,
                edgecolors="white", linewidths=0.6,
            )
            ax_c.vlines(
                row["x"], 0, row["n_cells"],
                color=col, linewidth=1.2, alpha=0.5,
            )

        # Slide labels on x-axis — use short IDs
        short_ids = [
            row[slide_col].replace("ADULT_","A").replace("AGED_","G")
            for _, row in counts.iterrows()
        ]
        ax_c.set_xticks(counts["x"])
        ax_c.set_xticklabels(short_ids, rotation=45, ha="right", fontsize=6)
        ax_c.set_ylabel("Cells in MBH ROI")
        ax_c.set_xlim(-0.6, len(counts) - 0.4)
        ax_c.set_ylim(0, counts["n_cells"].max() * 1.15)
        ax_c.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        ax_c.set_title("Cells per slide", fontsize=7)
        # Condition mean lines
        for cond in conditions:
            sub = counts[counts[condition_key] == cond]
            mean_n = sub["n_cells"].mean()
            xs = sub["x"].values
            ax_c.hlines(
                mean_n, xs.min() - 0.3, xs.max() + 0.3,
                color=palette[cond], linewidth=1.0,
                linestyle="--", alpha=0.7,
                label=f"{cond} mean",
            )
    else:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "slide_id not available",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=7, color="#888")
    _panel_label(ax_c, "c")

    # Shared legend (condition colours only)
    handles = [mpatches.Patch(color=palette[c], label=c) for c in conditions]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(conditions),
        frameon=False,
        fontsize=6,
        bbox_to_anchor=(0.35, -0.10),
    )
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)   # tight_layout + external legend
        fig.tight_layout(pad=0.5)
    out = _savefig(fig, output_dir / "fig1_qc", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


def _violin_split(ax, adata, key, condition_key, palette):
    data = pd.DataFrame({
        key: adata.obs[key].values,
        "condition": adata.obs[condition_key].values,
    })
    order = sorted(data["condition"].unique())
    # seaborn >= 0.13 requires palette to be passed via hue, not as a bare
    # keyword argument. Passing hue="condition" + legend=False is the correct
    # forward-compatible form and produces identical output.
    sns.violinplot(
        data=data, x="condition", y=key, order=order,
        hue="condition", palette=palette, legend=False,
        inner="quartile", linewidth=0.5, ax=ax, cut=0,
    )
    # set_ticks before set_ticklabels to avoid the FixedLocator UserWarning
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")


# ===========================================================================
# Figure 2: UMAP coloured by condition and cluster
# ===========================================================================

def plot_umap(
    adata: ad.AnnData,
    condition_key: str = "condition",
    cluster_key: str = "leiden",
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Two-panel UMAP figure:
      A: cells coloured by condition
      B: cells coloured by Leiden cluster
    """
    apply_nature_style()
    if "X_umap" not in adata.obsm:
        logger.warning("X_umap not found in adata.obsm; skipping fig2.")
        fig, ax = plt.subplots(figsize=(SINGLE, 1.5))
        ax.text(0.5, 0.5, "UMAP not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888")
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig2_umap", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out
    # Extra right margin so the cluster legend sits outside panel B
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.8),
                             gridspec_kw={"wspace": 0.08})
    umap = adata.obsm["X_umap"]

    # Panel A: condition
    conditions = adata.obs[condition_key].astype("category").cat.categories.tolist()
    cond_palette = {c: CONDITION_COLOURS.get(c, WONG[i]) for i, c in enumerate(conditions)}
    for cond in conditions:
        mask = adata.obs[condition_key] == cond
        axes[0].scatter(
            umap[mask, 0], umap[mask, 1],
            c=cond_palette[cond], s=0.5, alpha=0.4,
            linewidths=0, label=cond, rasterized=True,
        )
    # No axis labels — the small arrows from _add_scale_arrow are sufficient
    axes[0].set_xlabel(""); axes[0].set_ylabel("")
    axes[0].set_title("Condition", fontsize=8)
    leg = axes[0].legend(
        markerscale=6, frameon=False, fontsize=6.5,
        handletextpad=0.3, borderpad=0,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    for h in leg.legend_handles:
        h.set_alpha(1)
    _panel_label(axes[0], "a")

    # Panel B: clusters — use get_cluster_colours() so colours are identical
    # to Fig 3 (spatial) and Fig 8 (summary).
    cl_col_umap = get_cluster_colours(adata, cluster_key)
    clusters    = sorted(cl_col_umap.keys(), key=_safe_cluster_sort_key)
    for cl in clusters:
        mask = adata.obs[cluster_key] == cl
        axes[1].scatter(
            umap[mask, 0], umap[mask, 1],
            c=cl_col_umap[cl], s=0.5, alpha=0.5,
            linewidths=0, label=cl, rasterized=True,
        )
    axes[1].set_xlabel(""); axes[1].set_ylabel("")
    axes[1].set_title("Leiden clusters", fontsize=8)
    leg2 = axes[1].legend(
        markerscale=6, frameon=False, fontsize=6,
        handletextpad=0.3, ncol=2,
        title="Cluster", title_fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),   # outside panel B, to the right
        borderaxespad=0,
    )
    for h in leg2.legend_handles:
        h.set_alpha(1)
    _panel_label(axes[1], "b")

    for ax in axes:
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
        _add_scale_arrow(ax)

    # tight_layout must not clip the external legend — use rect to leave right margin
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)   # tight_layout + external legend
        fig.tight_layout(pad=0.4, rect=[0, 0, 0.88, 1])
    out = _savefig(fig, output_dir / "fig2_umap", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


def _add_scale_arrow(ax):
    """Add small axis arrows indicating UMAP directions."""
    x0, y0 = 0.05, 0.05
    length = 0.12
    kw = dict(transform=ax.transAxes, color="black",
              arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7),
              fontsize=6, ha="center", va="center")
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0), **kw)
    ax.annotate("", xy=(x0, y0 + length), xytext=(x0, y0), **kw)
    ax.text(x0 + length / 2, y0 - 0.04, "UMAP 1",
            transform=ax.transAxes, fontsize=6, ha="center")
    ax.text(x0 - 0.04, y0 + length / 2, "UMAP 2",
            transform=ax.transAxes, fontsize=6, ha="center", rotation=90)


# ===========================================================================
# Figure 3: Spatial map of clusters on tissue
# ===========================================================================

def plot_spatial_clusters(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    condition_key: str = "condition",
    output_dir: Path = Path("figures_output"),
    spot_size: float = 3.0,
    fmt: str = "pdf",
    dpi: int = 300,
    representative_slides: dict | None = None,
) -> Path:
    """
    Spatial maps coloured by cluster.

    Produces two figures:
      fig3_spatial_clusters          — all slides, one panel each (2-row grid)
      fig3_spatial_clusters_repr     — one representative slide per condition

    Parameters
    ----------
    representative_slides:
        Dict mapping condition label to slide_id, e.g.
        {"ADULT": "ADULT_1", "AGED": "AGED_3"}.
        If None, the first slide per condition is used.
    """
    apply_nature_style()
    if "spatial" not in adata.obsm:
        logger.warning("spatial coordinates not found; skipping fig3.")
        fig, ax = plt.subplots(figsize=(SINGLE, 1.5))
        ax.text(0.5, 0.5, "Spatial coordinates not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888")
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig3_spatial_clusters", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out

    conditions  = adata.obs[condition_key].astype("category").cat.categories.tolist()
    slide_col   = "slide_id" if "slide_id" in adata.obs.columns else None
    # Build cluster colours identically to Fig 2b and Fig 8.
    cluster_colour = get_cluster_colours(adata, cluster_key)
    clusters       = sorted(cluster_colour.keys(), key=_safe_cluster_sort_key)

    # Use cell type labels in legend if available, else raw cluster ID
    if "cell_type" in adata.obs.columns:
        _ct_map = (
            adata.obs[[cluster_key, "cell_type"]]
            .drop_duplicates()
            .set_index(cluster_key)["cell_type"]
            .to_dict()
        )
        legend_handles = [
            mpatches.Patch(color=cluster_colour[c],
                           label=str(_ct_map.get(c, c))) for c in clusters
        ]
    else:
        legend_handles = [
            mpatches.Patch(color=cluster_colour[c], label=str(c)) for c in clusters
        ]

    # ── Helper: draw one spatial panel ───────────────────────────────────────
    def _draw_slide(ax, sub_adata, title):
        if "spatial" not in sub_adata.obsm or sub_adata.n_obs == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6, color="#888")
            ax.axis("off")
            return
        xy      = sub_adata.obsm["spatial"]
        colours = sub_adata.obs[cluster_key].astype(str).map(cluster_colour).fillna("#CCCCCC").values
        ax.scatter(xy[:, 0], xy[:, 1],
                   c=colours, s=spot_size, alpha=0.7,
                   linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=6.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)", fontsize=6)
        ax.set_ylabel("y (µm)", fontsize=6)
        ax.tick_params(labelsize=6)

    # ── Figure A: all slides (grid: rows=conditions, cols=slides per condition) ──
    if slide_col is not None:
        slides_per_cond = {
            cond: sorted(adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique())
            for cond in conditions
        }
        n_cols = max(len(v) for v in slides_per_cond.values())
        n_rows = len(conditions)

        fig_all, axes_all = plt.subplots(
            n_rows, n_cols,
            figsize=(SINGLE * n_cols, SINGLE * 1.05 * n_rows),
            squeeze=False,
        )

        panel_labels = iter("abcdefghijklmnopqrstuvwxyz")
        for row, cond in enumerate(conditions):
            slides = slides_per_cond[cond]
            for col in range(n_cols):
                ax = axes_all[row, col]
                if col < len(slides):
                    sid  = slides[col]
                    mask = (adata.obs[condition_key] == cond) & (adata.obs[slide_col] == sid)
                    _draw_slide(ax, adata[mask], f"{cond} · {sid}")
                    _panel_label(ax, next(panel_labels))
                else:
                    ax.set_visible(False)

        fig_all.legend(
            handles=legend_handles, title="Cluster",
            loc="lower center", frameon=False,
            fontsize=5.5, title_fontsize=6,
            ncol=min(len(clusters), 8),
            bbox_to_anchor=(0.5, -0.03),
        )
        fig_all.suptitle("Spatial distribution of Leiden clusters — all slides",
                         fontsize=8, y=1.01)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            fig_all.tight_layout(pad=0.5)
        _savefig(fig_all, output_dir / "fig3_spatial_clusters_all", fmt=fmt, dpi=dpi)
        plt.close(fig_all)

    # ── Figure B: representative slides only ─────────────────────────────────
    if representative_slides is None:
        # Default: first slide per condition (alphabetical)
        if slide_col is not None:
            representative_slides = {
                cond: slides_per_cond[cond][0]
                for cond in conditions
                if slides_per_cond[cond]
            }
        else:
            representative_slides = {}

    n_repr = len(representative_slides) if representative_slides else len(conditions)
    fig_repr, axes_repr = plt.subplots(
        1, max(n_repr, 1),
        figsize=(SINGLE * max(n_repr, 1) * 1.15, SINGLE * 1.1),
        squeeze=False,
    )
    axes_repr = axes_repr[0]

    panel_iter = iter("ab")
    for i, cond in enumerate(conditions):
        if i >= len(axes_repr):
            break
        ax = axes_repr[i]
        if representative_slides and cond in representative_slides and slide_col is not None:
            sid  = representative_slides[cond]
            mask = (adata.obs[condition_key] == cond) & (adata.obs[slide_col] == sid)
            _draw_slide(ax, adata[mask], f"{cond}\n({sid})")
        else:
            mask = adata.obs[condition_key] == cond
            _draw_slide(ax, adata[mask], cond)
        _panel_label(ax, next(panel_iter))

    fig_repr.legend(
        handles=legend_handles, title="Cluster",
        loc="center right", frameon=False,
        fontsize=6, title_fontsize=6,
        ncol=1,
        bbox_to_anchor=(1.0, 0.5),
    )
    fig_repr.suptitle("Spatial distribution of Leiden clusters (representative sections)",
                      fontsize=8, y=1.01)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        fig_repr.tight_layout(pad=0.5)

    out = _savefig(fig_repr, output_dir / "fig3_spatial_clusters_repr", fmt=fmt, dpi=dpi)
    plt.close(fig_repr)
    return out


# ===========================================================================
# Figure 4: Marker gene dot plot
# ===========================================================================

def plot_dotplot(
    adata: ad.AnnData,
    n_genes_per_cluster: int = 5,
    cluster_key: str = "leiden",
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Dot plot of top marker genes per cluster.

    Dot size = fraction of cells expressing the gene.
    Dot colour = mean normalised expression.
    """
    import scanpy as sc
    apply_nature_style()

    # Extract top marker genes
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon",
                             use_raw=False, key_added="rgg_dot", pts=True,
                             n_genes=n_genes_per_cluster)
    result = adata.uns["rgg_dot"]
    clusters = list(result["names"].dtype.names)

    genes_ordered = []
    cluster_labels = []
    for cl in clusters:
        g = list(result["names"][cl][:n_genes_per_cluster])
        genes_ordered += g
        cluster_labels += [cl] * len(g)

    # Deduplicate while preserving order
    seen = set()
    uniq_genes = []
    for g in genes_ordered:
        if g not in seen and g in adata.var_names:
            uniq_genes.append(g)
            seen.add(g)

    # Build mean expression and fraction matrices
    X = _get_lognorm(adata)
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    clust_ids = sorted(adata.obs[cluster_key].unique(), key=_safe_cluster_sort_key)

    mean_expr = np.zeros((len(clust_ids), len(uniq_genes)))
    frac_expr = np.zeros_like(mean_expr)

    for i, cl in enumerate(clust_ids):
        mask = adata.obs[cluster_key] == cl
        sub = X[mask]
        for j, g in enumerate(uniq_genes):
            gi = var_idx[g]
            vals = sub[:, gi]
            mean_expr[i, j] = vals.mean()
            frac_expr[i, j] = (vals > 0).mean()

    # Plot
    n_genes = len(uniq_genes)
    n_clust = len(clust_ids)
    fig_w = max(DOUBLE, n_genes * 0.22 + 0.8)
    fig_h = max(1.5, n_clust * 0.25 + 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    max_expr = mean_expr.max() or 1
    norm = mcolors.Normalize(vmin=0, vmax=max_expr)
    cmap = mpl.colormaps.get_cmap("Reds")

    for i, cl in enumerate(clust_ids):
        for j, g in enumerate(uniq_genes):
            size = frac_expr[i, j] * 8  # max dot radius in points
            colour = cmap(norm(mean_expr[i, j]))
            ax.scatter(j, i, s=size ** 2, c=[colour], linewidths=0.2,
                       edgecolors="0.5", zorder=2)

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(uniq_genes, rotation=90, fontsize=6, style="italic")
    ax.set_yticks(range(n_clust))
    # Use descriptive cell type labels if available, otherwise "Cluster N"
    if "cell_type" in adata.obs.columns:
        ct_map = (
            adata.obs[[cluster_key, "cell_type"]]
            .drop_duplicates()
            .set_index(cluster_key)["cell_type"]
            .to_dict()
        )
        y_labels = [str(ct_map.get(c, f"Cluster {c}")) for c in clust_ids]
    else:
        y_labels = [f"Cluster {c}" for c in clust_ids]
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlim(-0.7, n_genes - 0.3)
    ax.set_ylim(-0.7, n_clust - 0.3)
    ax.grid(True, color="0.92", linewidth=0.3, zorder=0)

    # Colour bar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=15, pad=0.02)
    cbar.set_label("Mean expression", fontsize=6)
    cbar.ax.tick_params(labelsize=6, width=0.4, length=2)

    # Size legend — placed below the x-axis so it never conflicts with the colorbar
    for pct, lab in [(0.25, "25%"), (0.5, "50%"), (1.0, "100%")]:
        ax.scatter([], [], s=(pct * 8) ** 2, c="0.5",
                   linewidths=0.2, edgecolors="0.5",
                   label=lab)
    ax.legend(
        title="Fraction expressing",
        title_fontsize=6, fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        scatterpoints=1,
    )
    ax.set_title("Marker genes per cluster", fontsize=8)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig4_dotplot", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Figure 5: Volcano plot
# ===========================================================================

def plot_volcano(
    dge_results: pd.DataFrame,
    condition_a: str = "Control",
    condition_b: str = "Treatment",
    log2fc_col: str = "log2fc",
    padj_col: str = "pval_adj",
    log2fc_thresh: float = 1.0,  # aligned with stringent_wilcoxon
    pval_thresh: float = 0.01,  # aligned with stringent_wilcoxon
    n_label: int = 20,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Publication-quality volcano plot.

    Significant up-regulated genes (cond B vs A) in orange,
    down-regulated in blue, non-significant in grey.
    Top N genes by abs(log2FC) * -log10(padj) labelled.
    """
    apply_nature_style()
    df = dge_results.dropna(subset=[log2fc_col, padj_col]).copy()
    df["-log10_padj"] = -np.log10(df[padj_col].clip(1e-300))

    # Significance categories
    df["category"] = "ns"
    df.loc[
        (df[log2fc_col] > log2fc_thresh) & (df[padj_col] < pval_thresh),
        "category"
    ] = "up"
    df.loc[
        (df[log2fc_col] < -log2fc_thresh) & (df[padj_col] < pval_thresh),
        "category"
    ] = "down"

    cat_colour = {"ns": "#CCCCCC", "up": "#D55E00", "down": "#0072B2"}
    cat_alpha  = {"ns": 0.3,       "up": 0.8,       "down": 0.8}
    cat_size   = {"ns": 1,         "up": 2,          "down": 2}

    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))

    for cat in ["ns", "down", "up"]:
        sub = df[df["category"] == cat]
        ax.scatter(
            sub[log2fc_col], sub["-log10_padj"],
            c=cat_colour[cat], s=cat_size[cat],
            alpha=cat_alpha[cat], linewidths=0,
            rasterized=True, label=cat,
        )

    # Threshold lines
    ax.axhline(-np.log10(pval_thresh), color="0.4", lw=0.5, ls="--", zorder=0)
    ax.axvline(log2fc_thresh,  color="#D55E00", lw=0.5, ls="--", zorder=0, alpha=0.6)
    ax.axvline(-log2fc_thresh, color="#0072B2", lw=0.5, ls="--", zorder=0, alpha=0.6)

    # Labels for top genes
    label_score = df["-log10_padj"] * df[log2fc_col].abs()
    top_genes = df.loc[label_score.nlargest(n_label).index]
    _label_points(ax, top_genes, log2fc_col, "-log10_padj", "gene")

    ax.set_xlabel(f"log$_2$FC ({condition_b} / {condition_a})")
    ax.set_ylabel("-log$_{10}$(adj. p-value)")
    ax.set_title("Differential gene expression", fontsize=8)

    # Count annotations
    n_up   = (df["category"] == "up").sum()
    n_down = (df["category"] == "down").sum()
    ax.text(0.98, 0.97, f"Up: {n_up}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="#D55E00")
    ax.text(0.02, 0.97, f"Down: {n_down}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color="#0072B2")

    fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig5_volcano", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


def _label_points(ax, df, xcol, ycol, gene_col):
    """Add non-overlapping gene labels to scatter points."""
    from matplotlib import patheffects
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in df.iterrows():
            t = ax.text(
                row[xcol], row[ycol], row[gene_col],
                fontsize=6, style="italic",
                va="bottom", ha="center",
            )
            texts.append(t)
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.3),
            ax=ax,
        )
    except ImportError:
        # Fallback: draw without adjustment
        for _, row in df.iterrows():
            ax.text(
                row[xcol] + 0.03, row[ycol] + 0.1,
                row[gene_col],
                fontsize=6, style="italic",
                va="bottom",
                path_effects=[
                    patheffects.withStroke(linewidth=0.6, foreground="white")
                ],
            )


# ===========================================================================
# Figure 6: DGE heatmap
# ===========================================================================

def plot_dge_heatmap(
    adata: ad.AnnData,
    dge_results: pd.DataFrame,
    condition_key: str = "condition",
    log2fc_col: str = "log2fc",
    padj_col: str = "pval_adj",
    log2fc_thresh: float = 1.0,  # aligned with stringent_wilcoxon
    pval_thresh: float = 0.01,  # aligned with stringent_wilcoxon
    n_top: int = 40,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Heatmap of top DGE genes (rows) across cells (columns),
    annotated by condition.
    """
    apply_nature_style()
    df = dge_results.dropna(subset=[padj_col, log2fc_col])
    sig = df[(df[padj_col] < pval_thresh) & (df[log2fc_col].abs() > log2fc_thresh)]
    top_up   = sig[sig[log2fc_col] > 0].nlargest(n_top // 2, log2fc_col)
    top_down = sig[sig[log2fc_col] < 0].nsmallest(n_top // 2, log2fc_col)
    gene_col = "gene" if "gene" in dge_results.columns else dge_results.columns[0]
    top_genes = (
        list(top_up[gene_col].values) + list(top_down[gene_col].values)
    )
    top_genes = [g for g in top_genes if g in adata.var_names]

    if not top_genes:
        logger.warning("No significant genes for heatmap; skipping fig6.")
        fig, ax = plt.subplots(figsize=(SINGLE, 1.5))
        ax.text(0.5, 0.5, "No significant DEGs found\n(try relaxing thresholds)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#888")
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig6_heatmap", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out

    # Sample up to 300 cells per condition for display
    rng = np.random.default_rng(42)
    conditions = adata.obs[condition_key].astype("category").cat.categories.tolist()
    cell_idx = []
    for cond in conditions:
        idx = np.where(adata.obs[condition_key] == cond)[0]
        sel = rng.choice(idx, size=min(300, len(idx)), replace=False)
        cell_idx.extend(sorted(sel))

    sub = adata[cell_idx, top_genes]
    X = _get_lognorm(sub)

    # Z-score per gene — densify first (sub is already cell-sampled so manageable size)
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    Z = np.clip(Z, -3, 3)

    # Condition annotation colours
    # imshow() needs a numeric (N,1,4) RGBA array -- hex strings will crash.
    # Convert each cell's condition label to an RGBA tuple via mcolors.to_rgba.
    cond_labels = sub.obs[condition_key].values
    hex_map = {c: CONDITION_COLOURS.get(c, WONG[i]) for i, c in enumerate(conditions)}
    rgba_bar = np.array(
        [mcolors.to_rgba(hex_map[c]) for c in cond_labels],
        dtype=np.float32,
    ).reshape(1, -1, 4)   # shape (1, n_cells, 4) -- correct for imshow

    fig_w = max(DOUBLE, len(top_genes) * 0.14 + 1.2)
    fig_h = 3.2
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [0.08, 1], "hspace": 0.01},
    )

    # Condition bar
    axes[0].imshow(
        rgba_bar,
        aspect="auto",
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(["Condition"], fontsize=6)
    axes[0].spines[:].set_visible(False)

    # Heatmap
    im = axes[1].imshow(
        Z.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-3, vmax=3,
        interpolation="nearest",
    )
    axes[1].set_yticks(range(len(top_genes)))
    axes[1].set_yticklabels(top_genes, fontsize=6, style="italic")
    axes[1].set_xticks([])
    axes[1].set_xlabel(f"Cells (n = {len(cell_idx)})", fontsize=6)

    cbar = fig.colorbar(im, ax=axes[1], shrink=0.4, aspect=15, pad=0.02)
    cbar.set_label("Z-score", fontsize=6)
    cbar.ax.tick_params(labelsize=5, width=0.4, length=1.5)

    # Legend
    handles = [mpatches.Patch(color=CONDITION_COLOURS.get(c, WONG[i]), label=c)
               for i, c in enumerate(conditions)]
    fig.legend(handles=handles, loc="upper right", fontsize=6,
               frameon=False, bbox_to_anchor=(1.0, 0.98))

    fig.suptitle("Top DGE genes", fontsize=8, y=1.01)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=0.3)
    out = _savefig(fig, output_dir / "fig6_heatmap", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Figure 7: Spatial expression maps for top DGE genes
# ===========================================================================

def plot_spatial_expression(
    adata: ad.AnnData,
    genes: Sequence[str],
    condition_key: str = "condition",
    n_genes: int = 6,
    spot_size: float = 3.0,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
    representative_slides: dict | None = None,
) -> Path:
    """
    Grid of spatial expression maps for the top DGE genes.

    Produces two figures:
      fig7_spatial_expr          — rows=genes, columns=representative slide per condition
      fig7_spatial_expr_all      — rows=genes, columns=all individual slides

    Parameters
    ----------
    representative_slides:
        Dict mapping condition to slide_id, e.g. {"ADULT": "ADULT_1", "AGED": "AGED_3"}.
        If None, first slide per condition is used.
    """
    apply_nature_style()
    conditions     = adata.obs[condition_key].astype("category").cat.categories.tolist()
    slide_col      = "slide_id" if "slide_id" in adata.obs.columns else None
    genes_to_plot  = [g for g in genes if g in adata.var_names][:n_genes]
    n_genes_plot   = len(genes_to_plot)

    if not genes_to_plot:
        logger.warning("None of the requested genes found in adata.var_names.")
        fig, ax = plt.subplots(figsize=(SINGLE, 1.5))
        ax.text(0.5, 0.5, "No DEG genes found in spatial data",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#888")
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig7_spatial_expr", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out

    X       = _get_lognorm(adata)
    var_idx = {g: i for i, g in enumerate(adata.var_names)}

    # Resolve representative slides
    if slide_col is not None:
        slides_per_cond = {
            cond: sorted(adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique())
            for cond in conditions
        }
    else:
        slides_per_cond = {cond: [] for cond in conditions}

    if representative_slides is None:
        representative_slides = {
            cond: slides_per_cond[cond][0]
            for cond in conditions
            if slides_per_cond.get(cond)
        }

    # Custom colormap: dark grey (zero/background) → deep red (high expression)
    # Much cleaner than magma which starts near-black for both zero and low expr.
    _EXPR_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "grey_red",
        ["#2A2A2A", "#7B1010", "#CC2222", "#FF6B35", "#FFD166"],
        N=256,
    )

    # ── Helper: draw one spatial expression panel ─────────────────────────────
    def _draw_expr(ax, mask, gi, vmax, row, col, title):
        sub = adata[mask]
        if "spatial" not in sub.obsm or sub.n_obs == 0:
            ax.set_visible(False)
            return None
        xy  = sub.obsm["spatial"]
        _e  = X[mask.values if hasattr(mask, 'values') else mask, :][:, gi]
        e   = np.array(_e.todense()).ravel() if hasattr(_e, "todense") else np.array(_e).ravel()
        sc  = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=e, cmap=_EXPR_CMAP, vmin=0, vmax=vmax,
            s=spot_size, alpha=0.85,
            linewidths=0, rasterized=True,
        )
        ax.set_facecolor("#1A1A1A")          # dark background so grey cells read clearly
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=6, color="#EEEEEE")
        return sc

    def _add_cbar(fig, sc, ax_row_list, gene, fontsize_label=5, fontsize_tick=4.5):
        """Add a colorbar just outside the rightmost axes using inset_axes.

        Using ax.inset_axes() with axes-fraction coordinates means the colorbar
        moves with the axes after tight_layout — it never overlaps the data.
        [1.02, 0.1, 0.04, 0.8] = x=1.02 (just right of the axes edge),
        y=0.1, width=0.04 axes-widths, height=0.8 axes-heights.
        """
        ax_right = ax_row_list[-1]
        cax = ax_right.inset_axes([1.02, 0.1, 0.04, 0.8])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("log-norm", fontsize=fontsize_label, labelpad=2)
        cbar.ax.tick_params(labelsize=fontsize_tick, width=0.4, length=2)
        cbar.outline.set_linewidth(0.4)

    # ── Figure A: representative slides ──────────────────────────────────────
    cols_repr   = [(cond, representative_slides.get(cond)) for cond in conditions]
    n_cols_repr = len(cols_repr)
    fig_r, axes_r = plt.subplots(
        n_genes_plot, n_cols_repr,
        figsize=(SINGLE * 1.15 * n_cols_repr, SINGLE * 0.85 * n_genes_plot),
        squeeze=False,
    )
    fig_r.patch.set_facecolor("#111111")

    for row, gene in enumerate(genes_to_plot):
        gi   = var_idx[gene]
        _xi  = X[:, gi]
        expr = np.array(_xi.todense()).ravel() if hasattr(_xi, "todense") else np.array(_xi).ravel()
        vmax = float(np.percentile(expr[expr > 0], 99)) if (expr > 0).any() else 1.0
        last_sc = None
        for col, (cond, sid) in enumerate(cols_repr):
            ax = axes_r[row, col]
            if sid is not None and slide_col is not None:
                mask  = (adata.obs[condition_key] == cond) & (adata.obs[slide_col] == sid)
                title = f"{cond}\n({sid})"
            else:
                mask  = adata.obs[condition_key] == cond
                title = cond
            sc = _draw_expr(ax, mask, gi, vmax, row, col, title)
            if sc is not None:
                last_sc = sc
            if col == 0:
                ax.set_ylabel(gene, fontsize=6, style="italic", rotation=0,
                              labelpad=40, va="center", color="#DDDDDD")
        if last_sc is not None:
            _add_cbar(fig_r, last_sc, list(axes_r[row, :]), gene)

    fig_r.suptitle("Spatial expression of top DGE genes (representative sections)",
                   fontsize=8, color="#EEEEEE")
    fig_r.tight_layout(pad=0.4)  # leave room for cbars on right
    out = _savefig(fig_r, output_dir / "fig7_spatial_expr", fmt=fmt, dpi=dpi)
    plt.close(fig_r)

    # ── Figure B: all slides ──────────────────────────────────────────────────
    if slide_col is not None:
        all_cols   = [(cond, sid)
                      for cond in conditions
                      for sid in slides_per_cond[cond]]
        n_cols_all = len(all_cols)
        fig_a, axes_a = plt.subplots(
            n_genes_plot, n_cols_all,
            figsize=(SINGLE * 0.95 * n_cols_all, SINGLE * 0.80 * n_genes_plot),
            squeeze=False,
        )
        fig_a.patch.set_facecolor("#111111")

        for row, gene in enumerate(genes_to_plot):
            gi   = var_idx[gene]
            _xi  = X[:, gi]
            expr = np.array(_xi.todense()).ravel() if hasattr(_xi, "todense") else np.array(_xi).ravel()
            vmax = float(np.percentile(expr[expr > 0], 99)) if (expr > 0).any() else 1.0
            last_sc = None
            for col, (cond, sid) in enumerate(all_cols):
                ax   = axes_a[row, col]
                mask = (adata.obs[condition_key] == cond) & (adata.obs[slide_col] == sid)
                sc   = _draw_expr(ax, mask, gi, vmax, row, col, f"{cond}\n{sid}")
                if sc is not None:
                    last_sc = sc
                if col == 0:
                    ax.set_ylabel(gene, fontsize=6, style="italic", rotation=0,
                                  labelpad=35, va="center", color="#DDDDDD")
            if last_sc is not None:
                _add_cbar(fig_a, last_sc, list(axes_a[row, :]), gene,
                          fontsize_label=5.5, fontsize_tick=5)

        fig_a.suptitle("Spatial expression of top DGE genes — all slides",
                       fontsize=8, color="#EEEEEE")
        fig_a.tight_layout(pad=0.3)
        _savefig(fig_a, output_dir / "fig7_spatial_expr_all", fmt=fmt, dpi=dpi)
        plt.close(fig_a)

    return out


# ===========================================================================
# Figure 8: Summary multi-panel composite
# ===========================================================================

def plot_summary_panel(
    adata: ad.AnnData,
    dge_results: pd.DataFrame,
    condition_key: str = "condition",
    cluster_key: str = "leiden",
    representative_slides: dict = None,
    log2fc_col: str = "log2fc",
    padj_col: str = "pval_adj",
    log2fc_thresh: float = 1.0,
    pval_thresh: float = 0.01,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Composite 4-panel summary figure (Nature main-figure style).

      a: UMAP coloured by Leiden cluster
      b: Cell-type composition stacked bar (AGED vs ADULT)
      c: Representative ADULT spatial map (cluster colours)
      d: Representative AGED spatial map (cluster colours)

    All cluster-coloured panels (a, b, c, d) share exactly the same colour
    dictionary, built once via get_cluster_colours().
    """
    apply_nature_style()

    # ── Build ONE canonical cluster colour dict used by every panel ───────────
    cl_col    = get_cluster_colours(adata, cluster_key)
    clusters  = sorted(cl_col.keys(), key=_safe_cluster_sort_key)
    conditions = adata.obs[condition_key].astype("category").cat.categories.tolist()

    # Resolve representative slides (one per condition)
    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
    if representative_slides is None and slide_col is not None:
        representative_slides = {
            cond: sorted(
                adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique()
            )[0]
            for cond in conditions
            if len(adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique()) > 0
        }
    elif representative_slides is None:
        representative_slides = {}

    # ── Figure layout: 2 rows × 2 cols ───────────────────────────────────────
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.72))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        wspace=0.45, hspace=0.52,
    )
    ax_a = fig.add_subplot(gs[0, 0])   # UMAP by cluster
    ax_b = fig.add_subplot(gs[0, 1])   # Composition bar
    ax_c = fig.add_subplot(gs[1, 0])   # ADULT spatial
    ax_d = fig.add_subplot(gs[1, 1])   # AGED spatial

    umap    = adata.obsm.get("X_umap")
    _umap_ok = umap is not None
    if not _umap_ok:
        umap = np.zeros((adata.n_obs, 2))
        logger.warning("X_umap not found; panel a in fig8 will be empty.")

    # ── Panel a: UMAP by cluster ──────────────────────────────────────────────
    for cl in clusters:
        m = adata.obs[cluster_key] == cl
        ax_a.scatter(umap[m, 0], umap[m, 1],
                     c=cl_col[cl], s=0.5, alpha=0.5,
                     linewidths=0, label=str(cl), rasterized=True)
    _clean_umap_ax(ax_a)
    ax_a.set_title("Leiden clusters", fontsize=8)
    # Compact legend: cluster ID only (cell-type labels go on the composition bar)
    leg_a = ax_a.legend(
        markerscale=4, frameon=False, fontsize=6,
        ncol=2, title="Cluster", title_fontsize=6,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    for h in leg_a.legend_handles:
        h.set_alpha(1)
    _panel_label(ax_a, "a")

    # ── Panel b: cell-type composition stacked bar ────────────────────────────
    # Group by condition × cluster; use cell_type labels if available.
    if "cell_type" in adata.obs.columns:
        # Build cluster→cell_type mapping for display labels
        ct_map = (
            adata.obs[[cluster_key, "cell_type"]]
            .drop_duplicates()
            .set_index(cluster_key)["cell_type"]
            .to_dict()
        )
    else:
        ct_map = {cl: str(cl) for cl in clusters}

    props = (
        adata.obs.groupby([condition_key, cluster_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    props = props.div(props.sum(axis=1), axis=0) * 100
    # Reindex columns to match the canonical sorted cluster order
    cols_ordered = [cl for cl in clusters if cl in props.columns]
    props = props.reindex(columns=cols_ordered, fill_value=0)

    x_pos  = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    for cl in cols_ordered:
        vals = props.reindex(conditions)[cl].fillna(0).values
        ax_b.bar(x_pos, vals, bottom=bottom,
                 color=cl_col[cl],
                 width=0.55, linewidth=0,
                 label=ct_map.get(cl, str(cl)))
        bottom += vals

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(conditions, rotation=30, ha="right", fontsize=7)
    ax_b.set_ylabel("Proportion (%)", fontsize=7)
    ax_b.set_ylim(0, 105)
    ax_b.set_title("Cell-type composition", fontsize=8)
    ax_b.spines[["top", "right"]].set_visible(False)
    # Compact side legend listing cell-type names
    handles_b = [
        mpatches.Patch(color=cl_col[cl], label=ct_map.get(cl, str(cl)))
        for cl in cols_ordered
    ]
    leg_b = ax_b.legend(
        handles=handles_b,
        frameon=False, fontsize=6,
        ncol=1, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    _panel_label(ax_b, "b")

    # ── Panels c & d: representative spatial maps ─────────────────────────────
    def _draw_repr(ax, cond, title):
        """Draw a spatial scatter for one representative slide."""
        sid      = representative_slides.get(cond) if representative_slides else None
        if sid is not None and slide_col is not None:
            mask = (adata.obs[condition_key] == cond) & (adata.obs[slide_col] == sid)
            sub  = adata[mask]
            subtitle = f"{cond}  ·  {sid}"
        else:
            sub      = adata[adata.obs[condition_key] == cond]
            subtitle = cond

        ax.set_title(subtitle, fontsize=6.5)

        if "spatial" not in sub.obsm or sub.n_obs == 0:
            ax.text(0.5, 0.5, "No spatial data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="#888")
            ax.axis("off")
            return

        xy      = sub.obsm["spatial"]
        colours = sub.obs[cluster_key].astype(str).map(cl_col).fillna("#CCCCCC").values
        ax.scatter(xy[:, 0], xy[:, 1],
                   c=colours, s=1.2, alpha=0.75,
                   linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)

    # Determine condition order: put ADULT first, AGED second (c then d)
    cond_order = []
    for preferred in ["ADULT", "Control", conditions[0]]:
        if preferred in conditions:
            cond_order.append(preferred)
            break
    for c in conditions:
        if c not in cond_order:
            cond_order.append(c)

    cond_c = cond_order[0] if len(cond_order) > 0 else conditions[0]
    cond_d = cond_order[1] if len(cond_order) > 1 else conditions[-1]

    _draw_repr(ax_c, cond_c, cond_c)
    _panel_label(ax_c, "c")

    _draw_repr(ax_d, cond_d, cond_d)
    _panel_label(ax_d, "d")

    fig.suptitle("MBH Xenium — cluster overview", fontsize=9, y=1.02)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=0.5)
    out = _savefig(fig, output_dir / "fig8_summary", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


def _clean_umap_ax(ax):
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.spines[["left", "bottom"]].set_visible(False)
