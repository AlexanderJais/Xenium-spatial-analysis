"""
figures_extended.py
-------------------
Nature-grade figures for cell type annotation, spatial statistics,
and per-cluster DGE. Extends figures.py.

Figures:
  Fig 9  -- Cell type annotation panel
             A: UMAP coloured by cell type
             B: Spatial map of cell types
             C: Cell type proportions per condition (stacked bar)
             D: Confidence score distribution

  Fig 10 -- Spatial statistics panel
             A: Moran's I Manhattan-style plot (top spatially variable genes)
             B: Spatial co-expression heatmap
             C: Neighbourhood enrichment heatmap

  Fig 11 -- Per-cluster DGE panel
             A: DEG count heatmap (clusters x up/down)
             B: Dot plot of top DEGs per cluster coloured by condition
             C: Cluster-level log2FC comparison (bubble chart)
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

# Reuse style helpers from figures.py
from src.figures import (
    apply_nature_style,
    _savefig,
    _panel_label,
    _cluster_palette,
    _safe_cluster_sort_key,
    _get_lognorm,
    DOUBLE, SINGLE, WONG,
    CONDITION_COLOURS,
)

logger = logging.getLogger(__name__)

# Categorical colour palette for cell types (12 distinguishable colours)
CELL_TYPE_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#D37295", "#A0CBE8",
]


# ===========================================================================
# Figure 9: Cell type annotation panel
# ===========================================================================

def plot_cell_type_panel(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    output_dir: Path = Path("figures_output"),
    spot_size: float = 3.0,
    fmt: str = "pdf",
    dpi: int = 300,
    representative_slides: dict | None = None,
) -> Path:
    """
    Three-panel cell type annotation figure.

    A: UMAP coloured by cell type (legend outside, right of panel)
    B: Representative ADULT section spatial map by cell type
    C: Cell type proportions per condition (stacked bar, %)
    D: Representative AGED section spatial map by cell type

    Panels B and D show one representative slide per condition rather than
    merging all slides (which produces uninterpretable overlapping coordinates).
    Panel D (confidence scores) removed — not populated by cluster-label method.
    """
    apply_nature_style()

    if cell_type_key not in adata.obs:
        raise KeyError(f"'{cell_type_key}' not in adata.obs. Run annotation first.")

    cell_types = sorted(adata.obs[cell_type_key].dropna().unique())
    n_ct = len(cell_types)
    ct_pal = dict(zip(cell_types, CELL_TYPE_PALETTE[:n_ct]))

    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
    conditions = adata.obs[condition_key].cat.categories.tolist()

    # Resolve representative slides (default: first slide per condition)
    if representative_slides is None:
        representative_slides = {}
        if slide_col is not None:
            for cond in conditions:
                slides = sorted(
                    adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique()
                )
                if slides:
                    representative_slides[cond] = slides[0]

    # Layout: 2x2 grid. A=top-left, B=top-right, C=bottom-left, D=bottom-right
    # Extra right margin to accommodate the legend placed outside panel A.
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.75))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    if "X_umap" not in adata.obsm:
        logger.warning("X_umap missing; skipping fig9 UMAP panel.")
        umap = None
    else:
        umap = adata.obsm["X_umap"]

    # ── A: UMAP by cell type — legend placed OUTSIDE to the right ──────────
    if umap is not None:
        for ct in cell_types:
            m = adata.obs[cell_type_key] == ct
            ax_a.scatter(
                umap[m, 0], umap[m, 1],
                c=ct_pal.get(ct, "#CCCCCC"),
                s=0.6, alpha=0.5, linewidths=0,
                label=ct, rasterized=True,
            )
        ax_a.set_title("Cell types (UMAP)", fontsize=7.5)
        _clean_ax(ax_a)
        leg = ax_a.legend(
            markerscale=6, frameon=False, fontsize=5.5,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),   # outside the axes, right side
            borderaxespad=0,
            handletextpad=0.3,
        )
        for h in leg.legend_handles:
            h.set_alpha(1)
    else:
        ax_a.text(0.5, 0.5, "UMAP not available", ha="center", va="center",
                  transform=ax_a.transAxes, fontsize=8, color="#888")
        ax_a.axis("off")
    _panel_label(ax_a, "a")

    # ── Helper: draw a single-slide spatial cell type map ───────────────────
    def _draw_spatial_ct(ax, cond, slide_id, title):
        if "spatial" not in adata.obsm:
            ax.text(0.5, 0.5, "No spatial", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6)
            ax.axis("off")
            return
        if slide_col is not None and slide_id is not None:
            mask = (adata.obs[condition_key] == cond) & (adata.obs[slide_col] == slide_id)
        else:
            mask = adata.obs[condition_key] == cond
        sub   = adata[mask]
        if sub.n_obs == 0:
            ax.text(0.5, 0.5, f"No cells for {cond}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6)
            ax.axis("off")
            return
        xy      = sub.obsm["spatial"]
        colours = sub.obs[cell_type_key].astype(str).map(ct_pal).fillna("#CCCCCC").values
        ax.scatter(xy[:, 0], xy[:, 1],
                   c=colours, s=spot_size, alpha=0.75,
                   linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)", fontsize=6)
        ax.set_ylabel("y (µm)", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=7)

    # ── B: Representative ADULT section ─────────────────────────────────────
    adult_slide = representative_slides.get("ADULT")
    adult_label = f"ADULT — {adult_slide}" if adult_slide else "ADULT"
    _draw_spatial_ct(ax_b, "ADULT", adult_slide, adult_label)
    _panel_label(ax_b, "b")

    # ── C: Proportion stacked bar ────────────────────────────────────────────
    props = (
        adata.obs.groupby([condition_key, cell_type_key], observed=True)
        .size().unstack(fill_value=0)
    )
    props = props.div(props.sum(axis=1), axis=0) * 100
    x_pos  = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    for ct in cell_types:
        if ct not in props.columns:
            continue
        vals = props.reindex(conditions)[ct].fillna(0).values
        ax_c.bar(x_pos, vals, bottom=bottom,
                 color=ct_pal.get(ct, "#CCCCCC"),
                 width=0.6, linewidth=0, label=ct)
        bottom += vals
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(conditions, rotation=30, ha="right", fontsize=6.5)
    ax_c.set_ylabel("Proportion (%)")
    ax_c.set_title("Cell type composition", fontsize=7.5)
    ax_c.set_ylim(0, 108)
    _panel_label(ax_c, "c")

    # ── D: Representative AGED section ──────────────────────────────────────
    aged_slide = representative_slides.get("AGED")
    aged_label = f"AGED — {aged_slide}" if aged_slide else "AGED"
    _draw_spatial_ct(ax_d, "AGED", aged_slide, aged_label)
    _panel_label(ax_d, "d")

    fig.suptitle("Cell type annotation", fontsize=9, y=1.01)
    # rect leaves the right margin free for the legend outside panel A
    fig.tight_layout(pad=0.4, rect=[0, 0, 0.88, 1])
    out = _savefig(fig, output_dir / "fig9_cell_types", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Figure 10: Spatial statistics panel
# ===========================================================================

def plot_spatial_stats(
    morans_df: pd.DataFrame,
    coexpr_matrix: Optional[pd.DataFrame] = None,
    neighborhood_result: Optional[dict] = None,
    n_top_genes: int = 30,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Three-panel spatial statistics figure.

    A: Moran's I lollipop for top spatially variable genes
    B: Spatial co-expression heatmap (if coexpr_matrix provided)
    C: Neighbourhood enrichment heatmap (if neighborhood_result provided)

    Parameters
    ----------
    morans_df:
        Output of spatial_stats.morans_i_scan.
    coexpr_matrix:
        Output of spatial_stats.spatial_coexpression (gene x gene DataFrame).
    neighborhood_result:
        Output of spatial_stats.neighborhood_enrichment.
    n_top_genes:
        Number of top SVGs to show in panel A.
    """
    apply_nature_style()

    # Determine layout based on available data
    n_panels = 1 + (coexpr_matrix is not None) + (neighborhood_result is not None)
    fig_w = DOUBLE if n_panels == 3 else (DOUBLE * 0.66 * n_panels)

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 2.8))
    if n_panels == 1:
        axes = [axes]
    axes = list(axes)
    panel_idx = 0

    # --- A: Moran's I lollipop ---
    ax = axes[panel_idx]
    df = morans_df.dropna(subset=["morans_i"]).head(n_top_genes).copy()
    df = df.sort_values("morans_i", ascending=True)
    y_pos = np.arange(len(df))
    sig = df["p_adj"] < 0.05
    colours = np.where(sig, "#D55E00", "#AAAAAA")

    ax.hlines(y_pos, 0, df["morans_i"], color=colours, linewidth=0.6, alpha=0.8)
    ax.scatter(df["morans_i"], y_pos, c=colours, s=8, zorder=3, linewidths=0)
    ax.axvline(0, color="black", lw=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["gene"].values, fontsize=6, style="italic")
    ax.set_xlabel("Moran's I")
    ax.set_title(f"Top {n_top_genes} spatially variable genes")

    # Legend
    handles = [
        mpatches.Patch(color="#D55E00", label="adj-p < 0.05"),
        mpatches.Patch(color="#AAAAAA", label="ns"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="lower right")
    _panel_label(ax, chr(ord("a") + panel_idx))
    panel_idx += 1

    # --- B: Spatial co-expression heatmap ---
    if coexpr_matrix is not None and panel_idx < len(axes):
        ax = axes[panel_idx]
        n_genes = len(coexpr_matrix)
        font_s = max(5, 7 - n_genes // 5)
        mask = np.eye(n_genes, dtype=bool)
        vmax = np.abs(coexpr_matrix.values[~mask]).max() or 1
        im = ax.imshow(
            coexpr_matrix.values,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )
        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(coexpr_matrix.columns, rotation=90,
                           fontsize=font_s, style="italic")
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(coexpr_matrix.index, fontsize=font_s, style="italic")
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=15, pad=0.02)
        cbar.set_label("Spatial lag\ncorrelation", fontsize=6)
        cbar.ax.tick_params(labelsize=5, width=0.4, length=1.5)
        ax.set_title("Spatial co-expression")
        _panel_label(ax, chr(ord("a") + panel_idx))
        panel_idx += 1

    # --- C: Neighbourhood enrichment ---
    if neighborhood_result is not None and panel_idx < len(axes):
        ax = axes[panel_idx]
        z = neighborhood_result["z_score"]
        p = neighborhood_result.get("p_adj", neighborhood_result["p_value"])
        labels = z.index.tolist()
        n_ct = len(labels)

        # Draw heatmap
        vmax = np.abs(z.values).max() or 1
        im = ax.imshow(
            z.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )

        # Mark significant cells with asterisk
        for i in range(n_ct):
            for j in range(n_ct):
                if p.iloc[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=6, color="black")

        font_s = max(5, 7 - n_ct // 4)
        ax.set_xticks(range(n_ct))
        ax.set_xticklabels(labels, rotation=90, fontsize=font_s)
        ax.set_yticks(range(n_ct))
        ax.set_yticklabels(labels, fontsize=font_s)
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=15, pad=0.02)
        cbar.set_label("z-score", fontsize=6)
        cbar.ax.tick_params(labelsize=5, width=0.4, length=1.5)
        ax.set_title("Neighbourhood enrichment\n(* p < 0.05)")
        _panel_label(ax, chr(ord("a") + panel_idx))

    fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig10_spatial_stats", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Figure 11: Per-cluster DGE panel
# ===========================================================================

def plot_cluster_dge(
    cluster_dge: pd.DataFrame,
    adata: Optional[ad.AnnData] = None,
    condition_key: str = "condition",
    n_top_per_cluster: int = 4,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Three-panel per-cluster DGE figure.

    A: DEG count bar chart (up / down per group)
    B: Bubble chart: top genes x clusters (size = -log10 padj, colour = log2fc)
    C: Heatmap of log2FC for top shared genes across clusters

    Parameters
    ----------
    cluster_dge:
        Output of cluster_dge.run_cluster_dge (long-format DataFrame).
    adata:
        Optional. Used only for cell count display in panel A.
    n_top_per_cluster:
        Top N up + N down genes to highlight in panel B.
    """
    apply_nature_style()

    if cluster_dge.empty:
        logger.warning("cluster_dge DataFrame is empty; skipping Fig 11.")
        fig, ax = plt.subplots(figsize=(SINGLE, 1.5))
        ax.text(0.5, 0.5, "No per-cluster DGE results",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#888")
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig11_cluster_dge", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out

    groups = cluster_dge["group"].unique().tolist()
    n_groups = len(groups)

    # Pre-compute top genes so we can scale figure height to both groups and genes.
    # Panel A is a narrow bar chart; panel B (bubble) gets 3× the width.
    # Figure stays at Nature double-column width (7.2") and scales taller instead.
    from src.cluster_dge import top_genes_per_group as _tgpg
    _top_pre = _tgpg(cluster_dge, n=n_top_per_cluster, direction="both")
    n_genes_est = len(_top_pre["gene"].unique()) if not _top_pre.empty else 1
    fig_w = DOUBLE
    fig_h = max(4.5, max(n_groups * 0.30, n_genes_est * 0.22) + 1.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    # GridSpec: A gets 1 unit, B gets 3 units width; tight gap between them
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[1, 3.2],
                           wspace=0.35)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # --- A: DEG count bar chart ---
    summary = (
        cluster_dge.groupby("group", observed=True)["direction"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(groups, fill_value=0)
    )
    y_pos = np.arange(n_groups)

    up_vals   = summary.get("up",   pd.Series(0, index=groups)).values.astype(float)
    down_vals = summary.get("down", pd.Series(0, index=groups)).values.astype(float)

    ax_a.barh(y_pos,  up_vals,   color="#D55E00", height=0.6, linewidth=0, label="Up")
    ax_a.barh(y_pos, -down_vals, color="#0072B2", height=0.6, linewidth=0, label="Down")
    ax_a.axvline(0, color="black", lw=0.4)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([str(g) for g in groups], fontsize=6)
    ax_a.set_xlabel("Number of DEGs")
    ax_a.set_title("DEGs per group")
    ax_a.legend(frameon=False, fontsize=6, loc="lower right")
    _panel_label(ax_a, "a")

    # --- B: Bubble chart (gene x cluster) — takes all remaining width ---
    top = _top_pre  # already computed above for figure sizing

    if not top.empty:
        genes_ordered = (
            top.groupby("gene", observed=True)["log2fc"]
            .mean().abs()
            .sort_values(ascending=False)
            .index.tolist()
        )
        genes_ordered = list(dict.fromkeys(genes_ordered))
        gene_y  = {g: i for i, g in enumerate(genes_ordered)}
        group_x = {g: i for i, g in enumerate(groups)}

        norm = mcolors.TwoSlopeNorm(
            vcenter=0,
            vmin=top["log2fc"].min() - 0.01,
            vmax=top["log2fc"].max() + 0.01,
        )
        cmap = mpl.colormaps.get_cmap("RdBu_r")

        for _, row in top.iterrows():
            if row["gene"] not in gene_y or str(row["group"]) not in group_x:
                continue
            xi = group_x[str(row["group"])]
            yi = gene_y[row["gene"]]
            neg_log_p = min(-np.log10(row["pval_adj"] + 1e-300), 8)
            size = (neg_log_p * 7) ** 1.35    # slightly larger dots now we have more space
            colour = cmap(norm(row["log2fc"]))
            ax_b.scatter(xi, yi, s=size, c=[colour], linewidths=0.25,
                         edgecolors="0.5", zorder=2)

        ax_b.set_xticks(range(len(groups)))
        ax_b.set_xticklabels([str(g) for g in groups], rotation=90, fontsize=6)
        ax_b.set_yticks(range(len(genes_ordered)))
        ax_b.set_yticklabels(genes_ordered, fontsize=6, style="italic")
        ax_b.set_xlim(-0.7, len(groups) - 0.3)
        ax_b.set_ylim(-0.7, len(genes_ordered) - 0.3)
        ax_b.grid(True, color="0.92", linewidth=0.35, zorder=0)
        ax_b.set_title("Top DEGs per group")

        # Colorbar anchored to panel B
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_b, shrink=0.5, pad=0.02, aspect=15)
        cbar.set_label("log$_2$FC", fontsize=6)
        cbar.ax.tick_params(labelsize=5.5, width=0.4, length=2)
        cbar.outline.set_linewidth(0.4)

    _panel_label(ax_b, "b")

    fig.suptitle("Per-cluster differential gene expression", fontsize=9, y=1.01)
    fig.tight_layout(pad=0.5, rect=[0, 0, 0.97, 1])
    out = _savefig(fig, output_dir / "fig11_cluster_dge", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Helper
# ===========================================================================

def _clean_ax(ax):
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.spines[["left", "bottom"]].set_visible(False)


# ===========================================================================
# Figure 14: Insulin / metabolic signalling panel
# ===========================================================================

# Complete insulin-related gene set confirmed in this panel
_INSULIN_GENES = [
    "Insr","Igf1","Igf2","Igfbp4","Igfbp5","Igfbp6","Slc2a1","Slc2a2","Gck",
    "Lepr","Ghsr","Glp1r","Gipr","Mc3r","Gpr50","Npy4r","Cckar","Crhr2",
    "Sstr5","Galr1","Galr3",
    "Npy","Agrp","Hcrt","Pnoc","Npvf","Tac2","Grp","Nmb",
    "Pomc","Cartpt","Adcyap1","Sst","Vip","Nts","Cck","Gal","Ucn3",
    "Oxt","Avp","Prlh","Tac1","Sct","Ghrh",
]

_INSULIN_GROUPS = {
    "Insulin/IGF\nreceptor & transport": [
        "Insr","Igf1","Igf2","Igfbp4","Igfbp5","Igfbp6","Slc2a1","Slc2a2","Gck",
    ],
    "Energy-sensing\nreceptors": [
        "Lepr","Ghsr","Glp1r","Gipr","Mc3r","Gpr50","Npy4r",
        "Cckar","Crhr2","Sstr5","Galr1","Galr3",
    ],
    "Orexigenic\nneuropeptides": [
        "Npy","Agrp","Hcrt","Pnoc","Npvf","Tac2","Grp","Nmb",
    ],
    "Anorexigenic\nneuropeptides": [
        "Pomc","Cartpt","Adcyap1","Sst","Vip","Nts","Cck",
        "Gal","Ucn3","Oxt","Avp","Prlh","Tac1","Sct","Ghrh",
    ],
}

_GROUP_COLOURS = {
    "Insulin/IGF\nreceptor & transport": "#0072B2",
    "Energy-sensing\nreceptors":         "#009E73",
    "Orexigenic\nneuropeptides":         "#D55E00",
    "Anorexigenic\nneuropeptides":       "#CC79A7",
}


def plot_insulin_panel(
    adata,
    dge_results: "pd.DataFrame",
    cluster_dge: "pd.DataFrame",
    condition_key: str = "condition",
    cluster_key: str = "leiden",
    padj_col: str = "pval_adj",
    log2fc_col: str = "log2fc",
    pval_thresh: float = 0.05,
    output_dir: "Path" = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> "Path":
    """
    Fig 14 — Insulin & metabolic signalling in the aged MBH.

    Layout (3 rows, 4 cols with merged cells):
      a (top, full width): Horizontal lollipop — global log2FC for all 44
                            insulin genes, coloured by pathway group,
                            significant genes marked.
      b (mid, left 2):     Dot plot — mean expression × fraction expressing
                            for significant insulin DEGs across all cell types.
      c (mid, right 2):    Heatmap — per-cluster log2FC for significant
                            insulin DEGs (rows=genes, cols=cell types).
      d (bottom, full):    Stacked bar — per-cell-type count of up/down
                            insulin DEGs.
    """
    import warnings as _w
    import matplotlib.patches as mpatches
    from pathlib import Path as _Path

    if output_dir is None:
        output_dir = _Path("figures_output")
    output_dir = _Path(output_dir)

    apply_nature_style()

    # ── Data preparation ──────────────────────────────────────────────────────
    # Global DGE for insulin genes
    gdge = dge_results.copy()
    gene_col = "gene" if "gene" in gdge.columns else gdge.columns[0]
    gdge = gdge[gdge[gene_col].isin(_INSULIN_GENES)].copy()
    gdge = gdge.set_index(gene_col)

    # Cluster DGE for insulin genes
    cdge = cluster_dge[cluster_dge["gene"].isin(_INSULIN_GENES)].copy()

    # Significant genes at global level
    sig_global = set(gdge[gdge[padj_col] < pval_thresh].index.tolist())

    # Gene order for panels: by group then by global log2fc
    gene_order = []
    group_of   = {}
    for grp, genes in _INSULIN_GROUPS.items():
        sub = [g for g in genes if g in gdge.index]
        sub_sorted = sorted(sub, key=lambda g: gdge.loc[g, log2fc_col])
        gene_order.extend(sub_sorted)
        for g in sub_sorted:
            group_of[g] = grp

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 1.5))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.6, 1.6, 0.7],
        wspace=0.45, hspace=0.55,
    )
    ax_a  = fig.add_subplot(gs[0, :])   # full-width lollipop
    ax_b  = fig.add_subplot(gs[1, 0])   # dot plot
    ax_c  = fig.add_subplot(gs[1, 1])   # heatmap
    ax_d  = fig.add_subplot(gs[2, :])   # per-cluster bar

    # ── Panel a: Lollipop — global log2FC ─────────────────────────────────────
    y_pos = np.arange(len(gene_order))
    for yi, gene in enumerate(gene_order):
        if gene not in gdge.index:
            continue
        lfc  = gdge.loc[gene, log2fc_col]
        sig  = gene in sig_global
        grp  = group_of.get(gene, list(_INSULIN_GROUPS.keys())[0])
        col  = _GROUP_COLOURS[grp]
        alpha = 1.0 if sig else 0.35
        lw    = 0.8 if sig else 0.4
        ax_a.hlines(yi, 0, lfc, colors=col, linewidth=lw, alpha=alpha)
        ax_a.scatter(lfc, yi, color=col, s=16 if sig else 6,
                     zorder=3, alpha=alpha,
                     edgecolors="none" if not sig else "black",
                     linewidths=0.3)

    ax_a.axvline(0, color="black", lw=0.5, zorder=2)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(gene_order, fontsize=6, style="italic")
    ax_a.set_xlabel("log₂FC (AGED vs ADULT)", fontsize=7)
    ax_a.set_title("Insulin & metabolic signalling genes — global AGED vs ADULT", fontsize=8)

    # Significance stars for sig genes
    for yi, gene in enumerate(gene_order):
        if gene in sig_global and gene in gdge.index:
            lfc = gdge.loc[gene, log2fc_col]
            ax_a.text(lfc + (0.02 if lfc >= 0 else -0.02), yi, "*",
                      ha="left" if lfc >= 0 else "right",
                      va="center", fontsize=6, color="black")

    # Group colour legend
    legend_handles = [
        mpatches.Patch(color=col, label=grp.replace("\n", " "))
        for grp, col in _GROUP_COLOURS.items()
    ]
    ax_a.legend(handles=legend_handles, frameon=False, fontsize=6,
                loc="lower right", ncol=2)

    # Group bracket lines on the y-axis
    cumulative = 0
    for grp, genes in _INSULIN_GROUPS.items():
        n = len([g for g in genes if g in gdge.index])
        if n == 0:
            continue
        y_start = cumulative - 0.4
        y_end   = cumulative + n - 0.6
        ax_a.annotate("", xy=(-0.22, y_end), xytext=(-0.22, y_start),
                      xycoords=("axes fraction", "data"),
                      textcoords=("axes fraction", "data"),
                      arrowprops=dict(arrowstyle="-", color=_GROUP_COLOURS[grp], lw=2))
        ax_a.text(-0.235, (y_start + y_end) / 2,
                  grp.replace("\n", " "), fontsize=5.5, color=_GROUP_COLOURS[grp],
                  ha="right", va="center", rotation=0,
                  transform=ax_a.get_yaxis_transform())
        cumulative += n

    _panel_label(ax_a, "a")

    # ── Panel b: Dot plot across cell types ───────────────────────────────────
    # Rows = significant insulin DEGs; cols = cell types
    # Dot colour = mean log-norm expression; dot size = fraction expressing
    sig_genes = sorted(sig_global, key=lambda g: gdge.loc[g, log2fc_col])

    # Prefer cell_type labels if available
    if cluster_key in adata.obs.columns:
        ct_key = cluster_key
    elif "cell_type" in adata.obs.columns:
        ct_key = "cell_type"
    else:
        ct_key = cluster_key
    if "cell_type" in adata.obs.columns:
        ct_key = "cell_type"
    cell_types = sorted(adata.obs[ct_key].unique(), key=_safe_cluster_sort_key)

    X    = _get_lognorm(adata)
    vn   = list(adata.var_names)
    valid_sig = [g for g in sig_genes if g in vn]

    dot_mean = np.zeros((len(valid_sig), len(cell_types)))
    dot_frac = np.zeros_like(dot_mean)
    for ci, ct in enumerate(cell_types):
        mask = adata.obs[ct_key] == ct
        sub  = X[mask.values]
        for gi, gene in enumerate(valid_sig):
            idx   = vn.index(gene)
            vals  = np.asarray(sub[:, idx].todense()).ravel() \
                    if sp.issparse(sub) else sub[:, idx]
            dot_mean[gi, ci] = float(vals.mean())
            dot_frac[gi, ci] = float((vals > 0).mean())

    vmax = np.percentile(dot_mean[dot_mean > 0], 95) if (dot_mean > 0).any() else 1.0
    cmap = plt.get_cmap("Reds")

    for gi in range(len(valid_sig)):
        for ci in range(len(cell_types)):
            m = dot_mean[gi, ci]
            f = dot_frac[gi, ci]
            if f < 0.01:
                continue
            colour = cmap(min(m / vmax, 1.0))
            size   = max(2, f * 60)
            ax_b.scatter(ci, gi, s=size, color=colour,
                         linewidths=0.2, edgecolors="#888888", zorder=2)

    ax_b.set_xticks(range(len(cell_types)))
    ax_b.set_xticklabels(
        [ct.replace("GABAergic neuron", "GABA").replace("Glutamatergic neuron", "Glut")
           .replace(" neuron", "").replace("(", "").replace(")","")
         for ct in cell_types],
        rotation=60, ha="right", fontsize=5.5,
    )
    ax_b.set_yticks(range(len(valid_sig)))
    ax_b.set_yticklabels(valid_sig, fontsize=6, style="italic")
    ax_b.set_xlim(-0.6, len(cell_types) - 0.4)
    ax_b.set_ylim(-0.6, len(valid_sig) - 0.4)
    ax_b.set_title("Expression of sig. insulin DEGs\nacross cell types", fontsize=7)
    ax_b.set_xlabel("Cell type", fontsize=6)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_b, shrink=0.4, pad=0.01, aspect=15)
    cb.set_label("Mean log-norm", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    # Size legend
    for frac, label in [(0.1, "10%"), (0.5, "50%"), (1.0, "100%")]:
        ax_b.scatter([], [], s=max(2, frac * 60), color="#AAAAAA",
                     edgecolors="#888888", linewidths=0.2, label=label)
    ax_b.legend(title="% expr.", frameon=False, fontsize=5.5,
                title_fontsize=6, loc="upper left",
                bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _panel_label(ax_b, "b")

    # ── Panel c: Heatmap — per-cluster log2FC ─────────────────────────────────
    if len(cdge) > 0:
        # Build pivot: significant hits only; fill non-sig as 0
        pivot = cdge[cdge[padj_col] < pval_thresh].pivot_table(
            index="gene", columns="group", values=log2fc_col, aggfunc="mean"
        )
        pivot = pivot.reindex(index=[g for g in sig_genes if g in pivot.index])
        pivot = pivot.fillna(0)

        # Shorten cell type column names
        short = {ct: ct.replace("GABAergic neuron", "GABA")
                          .replace("Glutamatergic neuron", "Glut")
                          .replace(" neuron", "")
                          .replace("(", "").replace(")","").strip()
                 for ct in pivot.columns}
        pivot.columns = [short.get(c, c) for c in pivot.columns]

        vlim = max(abs(pivot.values.max()), abs(pivot.values.min()), 0.5)
        im   = ax_c.imshow(pivot.values,
                            aspect="auto", cmap="RdBu_r",
                            vmin=-vlim, vmax=vlim,
                            interpolation="nearest")

        ax_c.set_yticks(range(len(pivot.index)))
        ax_c.set_yticklabels(pivot.index, fontsize=6, style="italic")
        ax_c.set_xticks(range(len(pivot.columns)))
        ax_c.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=5.5)
        ax_c.set_title("Per-cluster log₂FC\n(sig. hits only, adj-p < 0.05)", fontsize=7)

        # Overlay significance dots
        sig_pivot_full = cdge[cdge[padj_col] < pval_thresh].pivot_table(
            index="gene", columns="group", values=padj_col, aggfunc="min"
        )
        sig_pivot_full.columns = [short.get(c, c) for c in sig_pivot_full.columns]
        for ri, gene in enumerate(pivot.index):
            for ci2, ct in enumerate(pivot.columns):
                p = sig_pivot_full.loc[gene, ct] \
                    if gene in sig_pivot_full.index and ct in sig_pivot_full.columns \
                    else np.nan
                if pd.notna(p) and p < pval_thresh:
                    ax_c.text(ci2, ri, "·", ha="center", va="center",
                              fontsize=8, color="black", fontweight="bold")

        cb2 = fig.colorbar(im, ax=ax_c, shrink=0.5, pad=0.01, aspect=15)
        cb2.set_label("log₂FC", fontsize=6)
        cb2.ax.tick_params(labelsize=5)
    else:
        ax_c.text(0.5, 0.5, "No cluster DGE data", ha="center",
                  va="center", transform=ax_c.transAxes, fontsize=7)
        ax_c.axis("off")
    _panel_label(ax_c, "c")

    # ── Panel d: Per-cluster up/down count bar ────────────────────────────────
    if len(cdge) > 0:
        sum_ct = (
            cdge[cdge[padj_col] < pval_thresh]
            .groupby(["group", "direction"])
            .size()
            .unstack(fill_value=0)
        )
        ct_order = sum_ct.sum(axis=1).sort_values(ascending=False).index.tolist()
        sum_ct   = sum_ct.reindex(ct_order)

        x_pos2 = np.arange(len(ct_order))
        up_vals   = sum_ct.get("up",   pd.Series(0, index=ct_order)).values
        down_vals = sum_ct.get("down", pd.Series(0, index=ct_order)).values

        ax_d.bar(x_pos2, up_vals,   color="#D55E00", width=0.55,
                 label="Up in AGED",   linewidth=0)
        ax_d.bar(x_pos2, -down_vals, color="#0072B2", width=0.55,
                 label="Down in AGED", linewidth=0)
        ax_d.axhline(0, color="black", lw=0.5)

        short_ct = [ct.replace("GABAergic neuron", "GABA")
                       .replace("Glutamatergic neuron", "Glut")
                       .replace(" neuron", "")
                       .replace("(", "").replace(")","").strip()
                    for ct in ct_order]
        ax_d.set_xticks(x_pos2)
        ax_d.set_xticklabels(short_ct, rotation=50, ha="right", fontsize=6)
        ax_d.set_ylabel("No. sig. DEGs", fontsize=6)
        ax_d.set_title(
            "Insulin/metabolic signalling DEGs per cell type  (adj-p < 0.05)", fontsize=7
        )
        ax_d.legend(frameon=False, fontsize=6, loc="upper right")
        # Label totals
        for xi, (u, dw) in enumerate(zip(up_vals, down_vals)):
            tot = u + dw
            if tot > 0:
                ax_d.text(xi, max(u, 0.2) + 0.15, str(tot),
                          ha="center", va="bottom", fontsize=6)
    else:
        ax_d.axis("off")
    _panel_label(ax_d, "d")

    fig.suptitle(
        "Insulin & metabolic signalling — AGED vs ADULT MBH",
        fontsize=9, y=1.01,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.6)

    out = _savefig(fig, output_dir / "fig14_insulin_signalling", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out
