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
from typing import Optional

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
    get_cell_type_colours,
    DOUBLE, SINGLE, WONG,
    CONDITION_COLOURS,
    CELL_TYPE_PALETTE,
)

logger = logging.getLogger(__name__)


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

    ct_pal = get_cell_type_colours(adata, cell_type_key)
    cell_types = sorted(ct_pal.keys(), key=_safe_cluster_sort_key)

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

    # Layout: 2x2 grid. A=UMAP (top-left), B=composition (top-right),
    # C=spatial ADULT (bottom-left), D=spatial AGED (bottom-right).
    # Extra right margin to accommodate the legend placed outside panel A.
    fig = plt.figure(figsize=(DOUBLE * 1.25, DOUBLE * 0.75))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.50, hspace=0.48)
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
            markerscale=6, frameon=False, fontsize=6,
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

    # ── B: Proportion stacked bar ────────────────────────────────────────────
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
        ax_b.bar(x_pos, vals, bottom=bottom,
                 color=ct_pal.get(ct, "#CCCCCC"),
                 width=0.6, linewidth=0, label=ct)
        bottom += vals
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(conditions, rotation=30, ha="right", fontsize=6.5)
    ax_b.set_ylabel("Proportion (%)")
    ax_b.set_title("Cell type composition", fontsize=7.5)
    ax_b.set_ylim(0, 108)
    _panel_label(ax_b, "b")

    # ── C: Representative ADULT section ─────────────────────────────────────
    adult_slide = representative_slides.get("ADULT")
    adult_label = f"ADULT — {adult_slide}" if adult_slide else "ADULT"
    _draw_spatial_ct(ax_c, "ADULT", adult_slide, adult_label)
    _panel_label(ax_c, "c")

    # ── D: Representative AGED section ──────────────────────────────────────
    aged_slide = representative_slides.get("AGED")
    aged_label = f"AGED — {aged_slide}" if aged_slide else "AGED"
    _draw_spatial_ct(ax_d, "AGED", aged_slide, aged_label)
    _panel_label(ax_d, "d")

    fig.suptitle("Cell type annotation", fontsize=9, y=1.01)
    # rect leaves the right margin free for the legend outside panel A
    fig.tight_layout(pad=0.4, rect=[0, 0, 0.90, 1])
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

    fig_h = max(4.0, n_top_genes * 0.18)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
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
        cbar = fig.colorbar(sm, ax=ax_b, shrink=0.3, pad=0.02, aspect=20)
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
    fig = plt.figure(figsize=(DOUBLE * 1.25, DOUBLE * 1.5))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.6, 1.6, 0.7],
        wspace=0.40, hspace=0.55,
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
    ax_a.legend(handles=legend_handles, frameon=False, fontsize=5.5,
                loc="upper left", bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0, ncol=1)

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
                  grp.replace("\n", " "), fontsize=6, color=_GROUP_COLOURS[grp],
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
        rotation=60, ha="right", fontsize=6,
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
    ax_b.legend(title="% expr.", frameon=False, fontsize=6,
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
        ax_c.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=6)
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
        fig.tight_layout(pad=0.6, rect=[0.02, 0, 0.92, 1])

    out = _savefig(fig, output_dir / "fig14_insulin", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Figure 15: Galanin (Gal) expression changes in the ageing MBH
# ===========================================================================

_GAL_GENE = "Gal"


def _bh_correct(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction — scipy-version-independent."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    order    = np.argsort(pvalues)
    pv_sort  = np.asarray(pvalues)[order]
    adjusted = np.empty(n)
    cummin   = 1.0
    for i in range(n - 1, -1, -1):
        cummin          = min(cummin, pv_sort[i] * n / (i + 1))
        adjusted[order[i]] = cummin
    return np.clip(adjusted, 0.0, 1.0)


def plot_galanin_panel(
    adata: ad.AnnData,
    dge_results: Optional[pd.DataFrame] = None,
    condition_key: str = "condition",
    cluster_key: str = "leiden",
    cell_type_key: str = "cell_type",
    padj_col: str = "pval_adj",
    log2fc_col: str = "log2fc",
    pval_thresh: float = 0.05,
    representative_slides: Optional[dict] = None,
    spot_size: float = 3.0,
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 15 — Galanin (Gal) expression changes in the ageing MBH.

    A: Spatial Gal expression map — representative ADULT section (dark bg)
    B: Spatial Gal expression map — representative AGED section (dark bg)
    C: Split violin of Gal log-norm expression per cell type × condition
    D: Per-cell-type log₂FC (AGED/ADULT) lollipop with BH-corrected significance

    Per-cell-type fold changes are computed from lognorm means (difference of
    geometric-mean approximations in natural-log space, converted to log₂FC).
    Mann-Whitney U tests across cells with Benjamini-Hochberg correction across
    cell types provide significance markers — note this is cell-level
    pseudoreplication; interpret as exploratory, not confirmatory.

    Parameters
    ----------
    adata : AnnData
        Preprocessed and annotated AnnData (lognorm layer required).
    dge_results : pd.DataFrame, optional
        Global DGE table; used to overlay the bulk Gal effect size and p-value.
    cell_type_key : str
        obs column with cell type labels; falls back to cluster_key if absent.
    pval_thresh : float
        BH-adj-p threshold for significance asterisks in panel D.
    representative_slides : dict, optional
        {condition: slide_id} mapping for spatial panels A/B.
    """
    from scipy import stats as _stats

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    gene = _GAL_GENE

    # ── Guard: gene absent from panel ─────────────────────────────────────────
    if gene not in adata.var_names:
        logger.warning("'%s' not found in adata.var_names; skipping fig15.", gene)
        fig, ax = plt.subplots(figsize=(SINGLE, 1.5))
        ax.text(0.5, 0.5, f"Gene '{gene}' not in Xenium panel",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#888888")
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig15_galanin", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out

    # ── Data preparation ──────────────────────────────────────────────────────
    ct_key     = cell_type_key if cell_type_key in adata.obs.columns else cluster_key
    conditions = adata.obs[condition_key].astype("category").cat.categories.tolist()
    cell_types = sorted(
        adata.obs[ct_key].dropna().unique(), key=_safe_cluster_sort_key
    )

    # Dense lognorm matrix (_get_lognorm converts sparse → dense internally)
    X        = _get_lognorm(adata)
    gi       = list(adata.var_names).index(gene)
    expr_all = np.asarray(X[:, gi]).ravel()

    cond_a = conditions[0]                                     # ADULT / Control
    cond_b = conditions[1] if len(conditions) > 1 else cond_a  # AGED / Treatment

    # Per-cell-type, per-condition expression vectors
    cond_mask = {c: (adata.obs[condition_key] == c).values for c in conditions}
    ct_mask   = {ct: (adata.obs[ct_key] == ct).values        for ct in cell_types}
    ct_cond_expr: dict[str, dict[str, np.ndarray]] = {
        ct: {c: expr_all[ct_mask[ct] & cond_mask[c]] for c in conditions}
        for ct in cell_types
    }

    # Per-cell-type log₂FC (lognorm means → natural-log difference → /ln(2))
    # and Mann-Whitney U two-sided p-value
    lfc_per_ct:  dict[str, float] = {}
    pval_per_ct: dict[str, float] = {}
    for ct in cell_types:
        va = ct_cond_expr[ct][cond_a]
        vb = ct_cond_expr[ct][cond_b]
        mean_a = float(va.mean()) if len(va) > 0 else 0.0
        mean_b = float(vb.mean()) if len(vb) > 0 else 0.0
        lfc_per_ct[ct] = (mean_b - mean_a) / np.log(2)
        if len(va) >= 3 and len(vb) >= 3:
            _, p = _stats.mannwhitneyu(vb, va, alternative="two-sided")
        else:
            p = 1.0
        pval_per_ct[ct] = p

    # BH correction across all cell types
    ct_list  = list(cell_types)
    padj_per_ct = dict(
        zip(ct_list, _bh_correct(np.array([pval_per_ct[ct] for ct in ct_list])))
    )

    # Resolve representative slides for spatial panels
    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
    if representative_slides is None:
        representative_slides = {}
        if slide_col is not None:
            for cond in conditions:
                slides = sorted(
                    adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique()
                )
                if slides:
                    representative_slides[cond] = slides[0]

    # Expression colour scale (shared A + B, 99th percentile of positive cells)
    pos_vals = expr_all[expr_all > 0]
    vmax     = float(np.percentile(pos_vals, 99)) if len(pos_vals) > 0 else 1.0
    _EXPR_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "grey_red",
        ["#2A2A2A", "#7B1010", "#CC2222", "#FF6B35", "#FFD166"],
        N=256,
    )

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(DOUBLE * 1.2, DOUBLE * 1.10))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.05, 1.0],
        wspace=0.48, hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[0, 0])   # spatial: ADULT
    ax_b = fig.add_subplot(gs[0, 1])   # spatial: AGED
    ax_c = fig.add_subplot(gs[1, 0])   # split violin per cell type
    ax_d = fig.add_subplot(gs[1, 1])   # log₂FC lollipop per cell type

    # ── Panels A & B: Spatial Gal expression ──────────────────────────────────
    def _draw_spatial(ax, cond, title):
        if "spatial" not in adata.obsm:
            ax.text(0.5, 0.5, "No spatial data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#888888")
            ax.axis("off")
            return None
        sid = representative_slides.get(cond)
        if sid is not None and slide_col is not None:
            idx = np.where(cond_mask[cond] & (adata.obs[slide_col] == sid).values)[0]
        else:
            idx = np.where(cond_mask[cond])[0]
        if len(idx) == 0:
            ax.text(0.5, 0.5, f"No cells for {cond}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#888888")
            ax.axis("off")
            return None
        xy = adata.obsm["spatial"][idx]
        e  = expr_all[idx]
        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=e, cmap=_EXPR_CMAP, vmin=0, vmax=vmax,
            s=spot_size, alpha=0.85, linewidths=0, rasterized=True,
        )
        ax.set_facecolor("#1A1A1A")
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)
        ax.set_title(title, fontsize=7, color="#EEEEEE", pad=3)
        return sc

    sc_a = _draw_spatial(ax_a, cond_a, f"Galanin — {cond_a}")
    sc_b = _draw_spatial(ax_b, cond_b, f"Galanin — {cond_b}")

    # Single shared colorbar anchored to the right of panel B
    ref_sc = sc_b if sc_b is not None else sc_a
    if ref_sc is not None:
        cax  = ax_b.inset_axes([1.04, 0.05, 0.04, 0.90])
        cbar = fig.colorbar(ref_sc, cax=cax)
        cbar.set_label("log-norm", fontsize=6, color="#CCCCCC")
        cbar.ax.yaxis.set_tick_params(labelsize=6, colors="#CCCCCC")
        cbar.outline.set_linewidth(0.4)
        cbar.outline.set_edgecolor("#555555")

    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")

    # ── Panel C: Split violins per cell type × condition ──────────────────────
    n_ct        = len(cell_types)
    cond_colour = {
        c: CONDITION_COLOURS.get(c, WONG[i % len(WONG)])
        for i, c in enumerate(conditions)
    }
    offsets = {cond_a: -0.18, cond_b: 0.18}  # left/right split

    for ci, ct in enumerate(cell_types):
        for cond in conditions:
            vals = ct_cond_expr[ct][cond]
            if len(vals) < 5:
                continue
            col = cond_colour[cond]
            off = offsets.get(cond, 0.0)
            try:
                from scipy.stats import gaussian_kde as _kde
                kde     = _kde(vals, bw_method=0.35)
                x_range = np.linspace(float(vals.min()), float(vals.max()), 100)
                dens    = kde(x_range)
                dens    = dens / dens.max() * 0.30   # normalise half-width
                ax_c.fill_betweenx(
                    x_range,
                    ci + off - dens,
                    ci + off + dens,
                    color=col, alpha=0.55, linewidth=0,
                )
            except Exception:
                pass
            # Median marker
            ax_c.scatter(ci + off, float(np.median(vals)),
                         color=col, s=12, zorder=4,
                         linewidths=0.5, edgecolors="white")

    ax_c.set_xticks(range(n_ct))
    _short_c = [
        ct.replace("GABAergic neuron", "GABA")
          .replace("Glutamatergic neuron", "Glut")
          .replace(" neuron", "")
          .replace("(", "").replace(")", "").strip()
        for ct in cell_types
    ]
    ax_c.set_xticklabels(_short_c, rotation=50, ha="right", fontsize=6)
    ax_c.set_ylabel("Gal log-norm expression", fontsize=6)
    ax_c.set_title("Galanin expression by cell type", fontsize=7)
    ax_c.set_xlim(-0.6, n_ct - 0.4)
    ax_c.spines[["top", "right"]].set_visible(False)

    leg_c = [mpatches.Patch(color=cond_colour[c], label=c) for c in conditions]
    ax_c.legend(handles=leg_c, frameon=False, fontsize=6, loc="upper right")
    _panel_label(ax_c, "c")

    # ── Panel D: Per-cell-type log₂FC lollipop ────────────────────────────────
    # Sorted by absolute fold change (largest effect at top)
    ct_sorted = sorted(cell_types, key=lambda ct: abs(lfc_per_ct[ct]), reverse=True)
    y_pos   = np.arange(len(ct_sorted))
    lfcs    = np.array([lfc_per_ct[ct]  for ct in ct_sorted])
    sigs    = np.array([padj_per_ct[ct] < pval_thresh for ct in ct_sorted])
    colours = [cond_colour[cond_b] if lfc > 0 else cond_colour[cond_a] for lfc in lfcs]
    alphas  = np.where(sigs, 1.0, 0.28)

    for i, (lfc, col, alpha, sig) in enumerate(zip(lfcs, colours, alphas, sigs)):
        ax_d.hlines(i, 0, lfc, colors=col, linewidth=0.9, alpha=float(alpha))
        ax_d.scatter(lfc, i, color=col, s=22 if sig else 7,
                     zorder=3, alpha=float(alpha), linewidths=0)
        if sig:
            ha_  = "left"  if lfc >= 0 else "right"
            off_ = 0.012   if lfc >= 0 else -0.012
            ax_d.text(lfc + off_, i, "✱",
                      ha=ha_, va="center", fontsize=7, color="black")

    ax_d.axvline(0, color="black", lw=0.5, zorder=2)
    ax_d.set_yticks(y_pos)
    _short_d = [
        ct.replace("GABAergic neuron", "GABA")
          .replace("Glutamatergic neuron", "Glut")
          .replace(" neuron", "")
          .replace("(", "").replace(")", "").strip()
        for ct in ct_sorted
    ]
    ax_d.set_yticklabels(_short_d, fontsize=6)
    ax_d.set_xlabel(f"log₂FC  Gal  ({cond_b} / {cond_a})", fontsize=6)
    ax_d.set_title(
        f"Galanin fold change per cell type\n(✱ BH-adj-p < {pval_thresh})",
        fontsize=7,
    )
    ax_d.spines[["top", "right"]].set_visible(False)

    leg_d = [
        mpatches.Patch(color=cond_colour[cond_b], label=f"Higher in {cond_b}"),
        mpatches.Patch(color=cond_colour[cond_a], label=f"Higher in {cond_a}"),
        mpatches.Patch(color="#BBBBBB", alpha=0.45, label="ns"),
    ]
    ax_d.legend(handles=leg_d, frameon=False, fontsize=5.5,
                loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    _panel_label(ax_d, "d")

    # ── Global DGE annotation from bulk results ───────────────────────────────
    if dge_results is not None and not dge_results.empty:
        gene_col = "gene" if "gene" in dge_results.columns else dge_results.columns[0]
        gal_row  = dge_results[dge_results[gene_col] == gene]
        if (not gal_row.empty
                and padj_col in gal_row.columns
                and log2fc_col in gal_row.columns):
            gal_lfc  = float(gal_row[log2fc_col].iloc[0])
            gal_padj = float(gal_row[padj_col].iloc[0])
            sig_star = " ✱" if gal_padj < pval_thresh else ""
            fig.text(
                0.5, 0.005,
                (f"Global DGE (all cells):  log₂FC = {gal_lfc:+.2f},  "
                 f"BH-adj-p = {gal_padj:.3g}{sig_star}"),
                ha="center", va="bottom", fontsize=6.5, color="#444444",
                transform=fig.transFigure,
            )

    fig.suptitle(
        "Galanin (Gal) in the ageing hypothalamus (MBH)",
        fontsize=9, y=1.01,
    )
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fig.tight_layout(pad=0.5, rect=[0, 0, 0.88, 1])

    out = _savefig(fig, output_dir / "fig15_galanin", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Fig 16: Cell type composition (scCODA / CLR + t-test)
# ===========================================================================

def plot_composition_panel(
    composition_results: "pd.DataFrame",
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    replicate_key: str = "slide_id",
    output_dir=None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 16 — Cell type composition testing panel.

    Layout (double-column, 2 rows):
      A (left)  — Stacked proportion bar chart per biological replicate,
                  grouped by condition.
      B (right) — Forest-plot-style lollipop showing log₂FC per cell type
                  with 95% credible / confidence interval (from composition_results).
                  Significant cell types highlighted in the condition-B colour.
    """
    import matplotlib.gridspec as gridspec

    from src.figures import (
        apply_nature_style, DOUBLE, WONG, _savefig,
    )
    from src.composition_analysis import _build_composition_table

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)

    apply_nature_style()

    conditions = sorted(adata.obs[condition_key].unique())
    cond_a = conditions[0] if len(conditions) >= 1 else "A"
    cond_b = conditions[1] if len(conditions) >= 2 else "B"
    pal = {cond_a: WONG[5], cond_b: WONG[6]}   # blue / vermillion

    # ── Build proportion table ────────────────────────────────────────────────
    comp_df = _build_composition_table(adata, cell_type_key, replicate_key, condition_key)
    cell_types = [c for c in comp_df.columns if c not in (condition_key, "n_cells")]

    # Proportions per replicate
    counts_mat = comp_df[cell_types].values.astype(float)
    row_sums = counts_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    props_mat = counts_mat / row_sums

    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.90))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        wspace=0.45,
        left=0.10, right=0.98, top=0.90, bottom=0.22,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ── Panel A: stacked proportion bars ─────────────────────────────────────
    # Use the canonical cell type palette for consistent colours across figures
    n_ct = len(cell_types)
    ct_pal = get_cell_type_colours(adata, cell_type_key)
    ct_colors = [ct_pal.get(ct, "#CCCCCC") for ct in cell_types]

    # Sort replicates: condition_a first, then condition_b
    rep_order = (
        list(comp_df.index[comp_df[condition_key] == cond_a]) +
        list(comp_df.index[comp_df[condition_key] == cond_b])
    )
    rep_order = [r for r in rep_order if r in comp_df.index]
    props_sorted = pd.DataFrame(props_mat, index=comp_df.index, columns=cell_types).loc[rep_order]
    cond_of_rep = comp_df[condition_key].loc[rep_order]

    bottom = np.zeros(len(rep_order))
    for i, ct in enumerate(cell_types):
        heights = props_sorted[ct].values
        ax_a.bar(
            range(len(rep_order)), heights,
            bottom=bottom,
            color=ct_colors[i],
            width=0.7,
            linewidth=0,
            label=ct,
        )
        bottom += heights

    # Condition separator line
    n_a = (cond_of_rep == cond_a).sum()
    if 0 < n_a < len(rep_order):
        ax_a.axvline(n_a - 0.5, color="#444444", lw=0.8, ls="--")

    # X-ticks = replicate IDs
    tick_labels = [str(r).replace("ADULT_", "Ad").replace("AGED_", "Ag") for r in rep_order]
    ax_a.set_xticks(range(len(rep_order)))
    ax_a.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6)
    ax_a.set_ylabel("Cell type proportion", fontsize=7)
    ax_a.set_ylim(0, 1)
    ax_a.set_title("a", loc="left", fontsize=9, fontweight="bold", pad=3)

    # Condition labels above bars
    if n_a > 0:
        ax_a.text(
            (n_a - 1) / 2, 1.04, cond_a, ha="center", va="bottom",
            fontsize=6.5, color=pal[cond_a], transform=ax_a.transData,
        )
    if n_a < len(rep_order):
        ax_a.text(
            (n_a + len(rep_order) - 1) / 2, 1.04, cond_b, ha="center", va="bottom",
            fontsize=6.5, color=pal[cond_b], transform=ax_a.transData,
        )

    # Legend: place below panel A
    handles_a = [
        plt.Rectangle((0, 0), 1, 1, fc=ct_colors[i], linewidth=0)
        for i in range(n_ct)
    ]
    ax_a.legend(
        handles_a, cell_types,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=min(3, n_ct),
        fontsize=5,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    # ── Panel B: forest plot (lollipop with CI) ───────────────────────────────
    if composition_results is not None and not composition_results.empty:
        df_b = composition_results.sort_values("log2fc").reset_index(drop=True)
        y_pos = np.arange(len(df_b))

        # CI bars
        if "credible_interval_lo" in df_b.columns and "credible_interval_hi" in df_b.columns:
            for i, row in df_b.iterrows():
                lo = row.get("credible_interval_lo", np.nan)
                hi = row.get("credible_interval_hi", np.nan)
                if np.isfinite(lo) and np.isfinite(hi):
                    ax_b.plot(
                        [lo, hi], [y_pos[i], y_pos[i]],
                        color="#BBBBBB", lw=1.2, zorder=1, solid_capstyle="round",
                    )

        # Lollipop stems
        sig_mask = df_b["significant"].values
        for i, (lfc, sig) in enumerate(zip(df_b["log2fc"], sig_mask)):
            color = pal[cond_b] if sig else "#AAAAAA"
            ax_b.plot([0, lfc], [y_pos[i], y_pos[i]], color=color, lw=0.8, zorder=2)

        # Dots
        scatter_colors = [pal[cond_b] if s else "#AAAAAA" for s in sig_mask]
        ax_b.scatter(
            df_b["log2fc"], y_pos,
            c=scatter_colors, s=18, zorder=3, linewidths=0,
        )

        # Significance marker
        for i, sig in enumerate(sig_mask):
            if sig:
                lfc = df_b["log2fc"].iloc[i]
                ax_b.text(
                    lfc + 0.08 * np.sign(lfc), y_pos[i], "✱",
                    ha="center", va="center", fontsize=6, color=pal[cond_b],
                )

        # Y-axis labels
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(df_b["cell_type"].values, fontsize=6)

        ax_b.axvline(0, color="#444444", lw=0.7, ls="-")
        ax_b.set_xlabel(f"log₂FC ({cond_b} / {cond_a})", fontsize=7)

        # Method annotation
        method_str = df_b["method"].iloc[0] if "method" in df_b.columns else ""
        method_label = {
            "scCODA"  : "scCODA  (Dirichlet-multinomial, Büttner 2021)",
            "CLR_ttest": "CLR + Welch t-test  (Aitchison 1986; scCODA fallback)",
        }.get(method_str, method_str)
        ax_b.text(
            0.5, -0.16, method_label,
            ha="center", va="top", fontsize=5,
            color="#666666", transform=ax_b.transAxes,
        )

        n_sig = int(sig_mask.sum())
        ax_b.set_title(
            f"b   n={n_sig} significant cell type{'s' if n_sig != 1 else ''}",
            loc="left", fontsize=9, fontweight="bold", pad=3,
        )
    else:
        ax_b.text(
            0.5, 0.5, "No composition results\navailable",
            ha="center", va="center", fontsize=7, transform=ax_b.transAxes,
        )
        ax_b.set_title("b", loc="left", fontsize=9, fontweight="bold", pad=3)

    fig.suptitle(
        "Cell type composition — AGED vs ADULT (MBH)",
        fontsize=9, y=0.97,
    )

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fig.tight_layout(pad=0.5, rect=[0, 0.05, 1, 0.96])

    out = _savefig(fig, output_dir / "fig16_composition", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Figure 17: Neuropeptide co-expression modules
# ===========================================================================

# MBH neuropeptide systems — gene sets ordered by functional relevance.
# Genes are filtered at runtime to those present in the panel.
NEUROPEPTIDE_MODULES: dict[str, list[str]] = {
    "AgRP/NPY\n(orexigenic)":    ["Agrp", "Npy", "Npy1r", "Npy2r", "Npy5r", "Ghsr"],
    "POMC/CART\n(anorexigenic)": ["Pomc", "Cartpt", "Pcsk1", "Pcsk2", "Mc3r"],
    "KNDy\n(Kiss1/Tac2/Pdyn)":   ["Kiss1", "Tac2", "Pdyn", "Tacr3", "Kiss1r"],
    "Somatostatin":              ["Sst", "Sstr1", "Sstr2", "Sstr3", "Sstr4", "Sstr5"],
    "TRH/Dopamine":              ["Trh", "Trhr", "Th", "Slc6a3", "Drd1", "Drd2"],
    "Galanin\nsystem":           ["Gal", "Galr1", "Galr2", "Galr3"],
}

_MODULE_COLOURS: dict[str, str] = {
    "AgRP/NPY\n(orexigenic)":    "#D55E00",
    "POMC/CART\n(anorexigenic)": "#0072B2",
    "KNDy\n(Kiss1/Tac2/Pdyn)":   "#CC79A7",
    "Somatostatin":              "#009E73",
    "TRH/Dopamine":              "#E69F00",
    "Galanin\nsystem":           "#56B4E9",
}

_SCORE_THRESHOLD = 0.05   # cells below this max score are labelled "unassigned"


def _score_neuropeptide_modules(
    adata: ad.AnnData,
    modules: dict[str, list[str]] = NEUROPEPTIDE_MODULES,
) -> list[tuple[str, str, int]]:
    """
    Score cells for each neuropeptide module with scanpy's score_genes.

    Returns
    -------
    List of (module_name, obs_key, n_genes_in_panel) for modules with ≥2 genes.
    Module scores are added to adata.obs in-place.
    """
    import scanpy as sc

    scored: list[tuple[str, str, int]] = []
    for name, genes in modules.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 2:
            logger.info(
                "Neuropeptide module '%s': %d/%d genes in panel — skipping",
                name, len(present), len(genes),
            )
            continue
        # Build a safe obs-column key from the first line of the name
        safe = name.split("\n")[0].lower()
        for ch in "/()- ":
            safe = safe.replace(ch, "_")
        key = f"npmod_{safe}"
        sc.tl.score_genes(adata, gene_list=present, score_name=key, use_raw=False)
        scored.append((name, key, len(present)))
        logger.info(
            "Neuropeptide module '%s': scored %d/%d genes → obs['%s']",
            name, len(present), len(genes), key,
        )
    return scored


def plot_neuropeptide_modules(
    adata: ad.AnnData,
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    representative_slides: Optional[dict] = None,
    score_threshold: float = _SCORE_THRESHOLD,
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 17 — Neuropeptide co-expression modules in the ageing MBH.

    Panels
    ------
    A (top-left):
        UMAP coloured by dominant neuropeptide module (argmax across module
        scores). Cells where every score ≤ score_threshold are grey
        ("unassigned").
    B (top-right):
        Heatmap — mean module score per cell type × module. Rows = cell types,
        columns = modules; each column is z-scored for visual contrast.
    C (bottom-left):
        Grouped bar chart — AGED vs ADULT mean module score for each module
        across all cells. Mann-Whitney U p-values annotated above each pair.
    D (bottom-right):
        Spatial maps — dominant module assignment on one representative slide
        per condition (ADULT top, AGED bottom). Grey = unassigned.

    Parameters
    ----------
    adata:
        AnnData with obs[condition_key], obs[cell_type_key], obsm['X_umap'],
        obs['x_centroid'] / obs['y_centroid'] (spatial coordinates).
    condition_key:
        obs column for condition labels.
    cell_type_key:
        obs column for cell type labels (used for panel B).
    representative_slides:
        Dict mapping condition → slide_id for panel D.  Auto-inferred if None.
    score_threshold:
        Minimum per-cell maximum module score to assign a dominant module.
    output_dir:
        Directory to save the figure.
    fmt / dpi:
        Figure format and resolution.
    """
    import warnings as _w
    from scipy import stats as _stats

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)

    apply_nature_style()

    # ── Score modules ──────────────────────────────────────────────────────────
    scored = _score_neuropeptide_modules(adata)
    if not scored:
        logger.warning("Fig 17: no neuropeptide modules could be scored — skipping.")
        return output_dir / f"fig17_neuropeptide_modules.{fmt}"

    module_names = [s[0] for s in scored]
    score_keys   = [s[1] for s in scored]
    n_genes_list = [s[2] for s in scored]
    mod_colours  = [_MODULE_COLOURS.get(n, "#888888") for n in module_names]

    score_mat = adata.obs[score_keys].values.astype(float)  # (n_cells, n_modules)

    # Dominant module: argmax; grey if max ≤ threshold
    max_scores = score_mat.max(axis=1)
    dom_idx    = score_mat.argmax(axis=1)
    dom_idx    = np.where(max_scores > score_threshold, dom_idx, -1)  # -1 = unassigned

    adata.obs["npmod_dominant"] = pd.Categorical(
        [module_names[i] if i >= 0 else "unassigned" for i in dom_idx]
    )

    # ── Conditions ────────────────────────────────────────────────────────────
    conditions = sorted(adata.obs[condition_key].unique().tolist())
    cond_a, cond_b = conditions[0], conditions[-1]
    cond_pal = {c: CONDITION_COLOURS.get(c, WONG[i]) for i, c in enumerate(conditions)}

    # ── Representative slides ─────────────────────────────────────────────────
    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
    if representative_slides is None:
        representative_slides = {}
        if slide_col is not None:
            for cond in conditions:
                slides = sorted(
                    adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique()
                )
                if slides:
                    representative_slides[cond] = slides[0]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(DOUBLE * 1.3, DOUBLE * 0.95))
    gs_outer = gridspec.GridSpec(
        2, 2, figure=fig, wspace=0.50, hspace=0.52,
    )
    ax_a = fig.add_subplot(gs_outer[0, 0])   # UMAP
    ax_b = fig.add_subplot(gs_outer[0, 1])   # Heatmap
    ax_c = fig.add_subplot(gs_outer[1, 0])   # Bar chart
    # Panel D: two stacked spatial maps
    gs_d = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[1, 1], hspace=0.35
    )
    ax_d0 = fig.add_subplot(gs_d[0])   # spatial — cond_a (ADULT first)
    ax_d1 = fig.add_subplot(gs_d[1])   # spatial — cond_b (AGED second)

    # ── Panel A: UMAP coloured by dominant module ──────────────────────────────
    if "X_umap" in adata.obsm:
        umap = adata.obsm["X_umap"]
        # Unassigned cells first (background)
        mask_un = dom_idx < 0
        ax_a.scatter(
            umap[mask_un, 0], umap[mask_un, 1],
            c="#CCCCCC", s=0.4, alpha=0.3, linewidths=0, rasterized=True,
        )
        legend_handles = []
        for mi, (name, key, col) in enumerate(zip(module_names, score_keys, mod_colours)):
            mask = dom_idx == mi
            if mask.sum() == 0:
                continue
            ax_a.scatter(
                umap[mask, 0], umap[mask, 1],
                c=col, s=0.8, alpha=0.7, linewidths=0,
                label=name, rasterized=True,
            )
            legend_handles.append(mpatches.Patch(color=col, label=name.replace("\n", " ")))
        # Unassigned in legend
        legend_handles.append(mpatches.Patch(color="#CCCCCC", label="unassigned"))
        ax_a.legend(
            handles=legend_handles, loc="upper left",
            bbox_to_anchor=(1.04, 1.0), borderaxespad=0,
            frameon=False, fontsize=5, handlelength=0.8,
        )
    else:
        ax_a.text(0.5, 0.5, "UMAP not available", ha="center", va="center",
                  fontsize=7, transform=ax_a.transAxes)

    _panel_label(ax_a, "a")
    ax_a.set_title("Dominant neuropeptide module", fontsize=7.5)
    _clean_ax(ax_a)

    # ── Panel B: Heatmap — mean score per cell type × module ──────────────────
    use_ct = cell_type_key if cell_type_key in adata.obs.columns else condition_key
    cell_types_present = sorted(adata.obs[use_ct].dropna().unique().tolist())

    hm_data = np.zeros((len(cell_types_present), len(score_keys)))
    for ci, ct in enumerate(cell_types_present):
        ct_mask = adata.obs[use_ct] == ct
        for mi, key in enumerate(score_keys):
            hm_data[ci, mi] = adata.obs.loc[ct_mask, key].mean()

    # Z-score each module column so cell types with high overall expression
    # don't dominate the colour scale
    with np.errstate(invalid="ignore", divide="ignore"):
        col_std = hm_data.std(axis=0)
        col_std[col_std == 0] = 1.0
        hm_z = (hm_data - hm_data.mean(axis=0)) / col_std

    # Short cell type labels for y-axis
    short_ct = [ct[:18] for ct in cell_types_present]
    short_mod = [n.split("\n")[0] for n in module_names]

    im = ax_b.imshow(
        hm_z, aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-2, vmax=2,
    )
    ax_b.set_xticks(np.arange(len(score_keys)))
    ax_b.set_xticklabels(short_mod, rotation=35, ha="right", fontsize=5.5)
    ax_b.set_yticks(np.arange(len(cell_types_present)))
    ax_b.set_yticklabels(short_ct, fontsize=5.5)
    ax_b.tick_params(left=False, bottom=False)

    # Module colour tick marks on x-axis
    for mi, col in enumerate(mod_colours):
        ax_b.get_xticklabels()[mi].set_color(col)

    cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04, shrink=0.7)
    cbar.set_label("z-score", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)

    _panel_label(ax_b, "b")
    ax_b.set_title("Module score per cell type", fontsize=7.5)

    # ── Panel C: AGED vs ADULT mean score per module ───────────────────────────
    n_mod = len(score_keys)
    x_pos = np.arange(n_mod)
    bar_w = 0.35

    # Pre-compute per-condition means/sems for bar plotting and ymax calculation
    means_by_cond: dict[str, list[float]] = {}
    sems_by_cond:  dict[str, list[float]] = {}
    for cond in conditions:
        cond_mask = adata.obs[condition_key] == cond
        means_by_cond[cond] = [adata.obs.loc[cond_mask, key].mean() for key in score_keys]
        sems_by_cond[cond]  = [
            adata.obs.loc[cond_mask, key].sem() if cond_mask.sum() > 1 else 0.0
            for key in score_keys
        ]

    for ci, cond in enumerate(conditions):
        offset = (ci - 0.5) * bar_w
        ax_c.bar(
            x_pos + offset, means_by_cond[cond], bar_w,
            color=cond_pal.get(cond, WONG[ci]),
            yerr=sems_by_cond[cond], capsize=2, error_kw={"linewidth": 0.7},
            label=cond, linewidth=0,
        )

    # Mann-Whitney U significance stars with BH correction across modules.
    # Cell-level test — exploratory; acknowledged in docstring.
    raw_pvals: list[float] = []
    test_indices: list[int] = []
    for mi, key in enumerate(score_keys):
        vals_a = adata.obs.loc[adata.obs[condition_key] == cond_a, key].dropna().values
        vals_b = adata.obs.loc[adata.obs[condition_key] == cond_b, key].dropna().values
        if len(vals_a) < 3 or len(vals_b) < 3:
            continue
        _, p = _stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        raw_pvals.append(p)
        test_indices.append(mi)

    if raw_pvals:
        # Benjamini-Hochberg correction across tested modules
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(raw_pvals, method="fdr_bh")
        for mi, p_adj in zip(test_indices, padj):
            star = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
            if star:
                # Place star just above the tallest bar-top (mean + sem) for this module
                bar_tops = [means_by_cond[c][mi] + sems_by_cond[c][mi] for c in conditions]
                ymax = max(max(bar_tops), 0) + 0.01
                ax_c.text(x_pos[mi], ymax, star, ha="center", va="bottom",
                          fontsize=6, color="#333333")

    ax_c.axhline(0, color="#AAAAAA", lw=0.5, ls="--")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(short_mod, rotation=30, ha="right", fontsize=5.5)
    ax_c.set_ylabel("Mean module score (±SEM)", fontsize=6.5)
    ax_c.legend(frameon=False, fontsize=6, loc="upper right")
    _panel_label(ax_c, "c")
    ax_c.set_title("Condition comparison per module", fontsize=7.5)
    ax_c.tick_params(labelsize=5.5)

    # ── Panel D: Spatial dominant-module maps ──────────────────────────────────
    # Support both obs columns (x_centroid/y_centroid) and obsm["spatial"]
    has_spatial_obs = ("x_centroid" in adata.obs.columns and
                       "y_centroid" in adata.obs.columns)
    has_spatial_obsm = "spatial" in adata.obsm
    has_spatial = has_spatial_obs or has_spatial_obsm

    all_dom_labels = [module_names[i] if i >= 0 else "unassigned" for i in dom_idx]
    all_dom_colours = [
        _MODULE_COLOURS.get(lb, "#CCCCCC") for lb in all_dom_labels
    ]

    for ax_d, cond in [(ax_d0, cond_a), (ax_d1, cond_b)]:
        slide_id = representative_slides.get(cond) if representative_slides else None
        if slide_id and slide_col in adata.obs.columns:
            slide_mask = (adata.obs[slide_col] == slide_id).values
        else:
            slide_mask = (adata.obs[condition_key] == cond).values

        if not has_spatial or slide_mask.sum() == 0:
            ax_d.text(0.5, 0.5, f"No spatial data\n({cond})",
                      ha="center", va="center", fontsize=6,
                      transform=ax_d.transAxes)
            ax_d.set_aspect("equal")
            ax_d.axis("off")
            ax_d.set_title(cond, fontsize=6)
            continue

        if has_spatial_obs:
            x = adata.obs.loc[slide_mask, "x_centroid"].values
            y = adata.obs.loc[slide_mask, "y_centroid"].values
        else:
            xy = adata.obsm["spatial"][slide_mask]
            x = xy[:, 0]
            y = xy[:, 1]
        colours_slide = np.array(all_dom_colours)[slide_mask]

        ax_d.scatter(x, -y, c=colours_slide, s=1.2, alpha=0.7,
                     linewidths=0, rasterized=True)
        ax_d.set_aspect("equal")
        ax_d.tick_params(left=False, bottom=False,
                         labelleft=False, labelbottom=False)
        ax_d.spines[["left", "bottom", "top", "right"]].set_visible(False)
        ax_d.set_title(cond, fontsize=6, pad=2)

    _panel_label(ax_d0, "d")

    # Gene-count annotation per module
    gene_note = "  |  ".join(
        f"{n.split(chr(10))[0]}: {k}g" for n, k in zip(module_names, n_genes_list)
    )
    fig.text(
        0.5, 0.01, gene_note, ha="center", va="bottom",
        fontsize=4.5, color="#666666",
    )

    fig.suptitle(
        "Neuropeptide co-expression modules — AGED vs ADULT (MBH)",
        fontsize=9, y=0.97,
    )

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fig.tight_layout(pad=0.5, rect=[0, 0.04, 0.88, 0.96])

    out = _savefig(fig, output_dir / "fig17_neuropeptide_modules", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out
