"""
figures_panel.py
----------------
Nature-grade panel QC figures for multi-slide Xenium experiments
with mixed base + custom gene panels.

Figures produced
----------------
Fig 13a  Panel composition bar chart
         Stacked bar per slide: base | custom_shared | custom_unique

Fig 13b  Custom gene presence heatmap
         Genes (rows) x slides (columns), colour = present / absent / zero-filled
         Rows sorted by number of slides; column annotations show condition.

Fig 13c  UpSet-style slide-count histogram
         How many custom genes appear in exactly k slides (k = 1 … N)?
         Bars coloured by partial_union threshold decision.

Fig 13d  Zero-fill impact per slide
         After partial_union harmonisation: how many custom gene columns
         were zero-filled per slide, broken down by gene category.

All four panels are combined into a single Fig 13 composite.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from src.figures import (
    apply_nature_style, _savefig, _panel_label,
    DOUBLE, SINGLE,
)

logger = logging.getLogger(__name__)

# Palette for panel types
PANEL_TYPE_COLOURS = {
    "base"           : "#0072B2",   # blue
    "custom_shared"  : "#009E73",   # green
    "custom_unique"  : "#E69F00",   # amber
    "zero_filled"    : "#CCCCCC",   # grey
}

# Wong 2011 colour-blind-safe palette (same as figures.py)
_WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
         "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def _condition_palette(conditions) -> dict:
    """
    Build a condition → hex-colour mapping from the actual condition labels.

    Uses Wong 2011 palette so colours are colour-blind safe and consistent
    with the rest of the pipeline.  The first sorted condition gets blue
    (#0072B2), the second gets vermillion (#D55E00), matching the AGED/ADULT
    convention but working for any study design.
    """
    unique = sorted(set(conditions))
    # Assign from index 5 (blue) onwards in the Wong palette
    palette_pool = [_WONG[5], _WONG[6], _WONG[1], _WONG[3], _WONG[7], _WONG[2]]
    return {c: palette_pool[i % len(palette_pool)] for i, c in enumerate(unique)}


# ===========================================================================
# Fig 13: Full panel QC composite
# ===========================================================================

def plot_panel_overview(
    adatas_raw: list[ad.AnnData],
    slide_ids: list[str],
    conditions: list[str],
    registry,                       # PanelRegistry instance
    harmonised: Optional[list[ad.AnnData]] = None,
    min_slides_threshold: int = 2,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Four-panel composite figure summarising the panel composition and
    overlap structure across all slides.

    Parameters
    ----------
    adatas_raw:
        Per-slide AnnData objects BEFORE harmonisation (raw gene sets).
    slide_ids:
        Slide identifiers (same order as adatas_raw).
    conditions:
        Condition label per slide (same order).
    registry:
        PanelRegistry instance (already loaded from base_panel_csv).
    harmonised:
        Per-slide AnnData objects AFTER harmonisation (optional).
        If provided, panel D shows zero-fill counts post-harmonisation.
    min_slides_threshold:
        The min_slides value used for partial_union (visualised in panel C).
    output_dir:
        Directory for saving.
    """
    apply_nature_style()

    overlap_df = registry.custom_gene_counts(adatas_raw, slide_ids)
    matrix     = registry.custom_overlap_matrix(adatas_raw, slide_ids)

    fig = plt.figure(figsize=(DOUBLE * 1.3, DOUBLE * 0.9))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        wspace=0.50, hspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _plot_composition_bars(ax_a, adatas_raw, slide_ids, conditions, registry, overlap_df)
    _panel_label(ax_a, "a")

    ax_b_ann, _ = _plot_presence_heatmap(ax_b, matrix, slide_ids, conditions, fig)
    # ax_b was removed inside _plot_presence_heatmap and replaced with two
    # stacked axes; use ax_b_ann (the condition bar, top axis) for the panel label.
    _panel_label(ax_b_ann, "b")

    _plot_upset_histogram(ax_c, overlap_df, len(slide_ids), min_slides_threshold)
    _panel_label(ax_c, "c")

    _plot_zerofill_bars(ax_d, harmonised or adatas_raw, slide_ids, conditions,
                        used_harmonised=harmonised is not None)
    _panel_label(ax_d, "d")

    fig.suptitle("Gene panel composition and overlap across slides", fontsize=9, y=1.01)
    fig.tight_layout(pad=0.5, rect=[0, 0, 0.92, 1])
    out = _savefig(fig, output_dir / "fig13_panel_qc", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===========================================================================
# Panel A: stacked bar of gene type composition per slide
# ===========================================================================

def _plot_composition_bars(ax, adatas, slide_ids, conditions, registry, overlap_df):
    cond_unique = sorted(set(conditions))
    cond_colours = _condition_palette(conditions)

    shared_all_genes  = set(overlap_df[overlap_df["category"] == "shared_all"]["gene"])
    shared_part_genes = set(overlap_df[overlap_df["category"] == "shared_partial"]["gene"])

    x = np.arange(len(slide_ids))
    bar_base   = []
    bar_shared = []
    bar_unique = []

    for adata in adatas:
        gs = set(adata.var_names)
        bar_base.append(len(gs & registry.base_gene_set))
        bar_shared.append(len((gs - registry.base_gene_set) & (shared_all_genes | shared_part_genes)))
        bar_unique.append(len((gs - registry.base_gene_set) - shared_all_genes - shared_part_genes))

    bar_base   = np.array(bar_base)
    bar_shared = np.array(bar_shared)
    bar_unique = np.array(bar_unique)

    ax.bar(x, bar_base,
           color=PANEL_TYPE_COLOURS["base"], width=0.6,
           linewidth=0, label="Base panel")
    ax.bar(x, bar_shared, bottom=bar_base,
           color=PANEL_TYPE_COLOURS["custom_shared"], width=0.6,
           linewidth=0, label="Custom shared")
    ax.bar(x, bar_unique, bottom=bar_base + bar_shared,
           color=PANEL_TYPE_COLOURS["custom_unique"], width=0.6,
           linewidth=0, label="Custom unique")

    # Condition tick colours
    ax.set_xticks(x)
    ax.set_xticklabels(slide_ids, rotation=45, ha="right", fontsize=6)
    for tick, cond in zip(ax.get_xticklabels(), conditions):
        tick.set_color(cond_colours.get(cond, "black"))

    ax.set_ylabel("Number of genes")
    ax.set_title("Panel composition per slide")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Total annotation on top of bars
    for xi, total in enumerate(bar_base + bar_shared + bar_unique):
        ax.text(xi, total + 1, str(int(total)),
                ha="center", va="bottom", fontsize=6, color="#444444")


# ===========================================================================
# Panel B: custom gene presence heatmap
# ===========================================================================

def _plot_presence_heatmap(ax, matrix, slide_ids, conditions, fig):
    """
    Rows = custom genes (sorted by slide count, desc).
    Cols = slides.
    Colour: dark = present, light = absent.
    Top annotation bar shows condition.
    """
    n_genes, n_slides = matrix.shape
    Z = matrix.values.astype(float)   # 1 = present, 0 = absent

    cond_colours_map = _condition_palette(conditions)

    # Split into condition annotation + heatmap
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    subplot_spec = ax.get_subplotspec()   # works regardless of parent GridSpec shape
    ax.remove()

    # Re-create as two stacked axes sharing the same subplot cell
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=subplot_spec,
        height_ratios=[0.05, 1],
        hspace=0.02,
    )
    ax_ann  = fig.add_subplot(sub_gs[0])
    ax_heat = fig.add_subplot(sub_gs[1])

    # Condition annotation bar
    cond_colours = [cond_colours_map.get(c, "#999") for c in conditions]
    ax_ann.imshow(
        np.array([[mcolors.to_rgba(c) for c in cond_colours]]),
        aspect="auto",
    )
    ax_ann.set_xticks([])
    ax_ann.set_yticks([0])
    ax_ann.set_yticklabels(["Condition"], fontsize=6)
    ax_ann.spines[:].set_visible(False)

    # Heatmap: custom two-colour map (present = teal, absent = near-white)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pres", ["#F0F0F0", "#009E73"]
    )
    ax_heat.imshow(Z, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    ax_heat.set_xticks(range(n_slides))
    ax_heat.set_xticklabels(slide_ids, rotation=45, ha="right", fontsize=6)
    for tick, cond in zip(ax_heat.get_xticklabels(), conditions):
        tick.set_color(cond_colours_map.get(cond, "black"))

    # Only show every Nth gene label if many genes
    step = max(1, n_genes // 30)
    yticks = list(range(0, n_genes, step))
    ax_heat.set_yticks(yticks)
    ax_heat.set_yticklabels(
        [matrix.index[i] for i in yticks],
        fontsize=max(6, 7 - n_genes // 15),
        style="italic",
    )
    ax_heat.set_title("Custom gene presence across slides", fontsize=7, pad=14)

    # Legend patches
    handles = [
        mpatches.Patch(color="#009E73", label="Present"),
        mpatches.Patch(color="#F0F0F0", label="Absent"),
    ]
    ax_heat.legend(handles=handles, frameon=False, fontsize=6,
                   loc="lower right", bbox_to_anchor=(1.0, 0))

    # Slide count annotations on right
    slide_counts = Z.sum(axis=1).astype(int)
    ax2 = ax_heat.twinx()
    ax2.set_ylim(ax_heat.get_ylim())
    ax2.set_yticks(range(n_genes))
    ax2.set_yticklabels(
        [str(c) for c in slide_counts],
        fontsize=max(6, 7 - n_genes // 20),
        color="#555555",
    )
    ax2.set_ylabel("n slides", fontsize=6, color="#555555")
    ax2.spines["right"].set_color("#CCCCCC")
    ax2.spines["right"].set_linewidth(0.4)

    return ax_ann, ax_heat


# ===========================================================================
# Panel C: UpSet-style slide-count histogram
# ===========================================================================

def _plot_upset_histogram(ax, overlap_df, n_slides, min_slides_threshold):
    """
    Bar chart: x = number of slides a custom gene appears in (1 to n_slides),
    y = number of custom genes with that slide count.
    Bars left of threshold are grey (will be dropped), right are green (kept).
    """
    counts = overlap_df["n_slides"].value_counts().sort_index()

    for k, count in counts.items():
        colour = (PANEL_TYPE_COLOURS["custom_shared"]
                  if k >= min_slides_threshold
                  else PANEL_TYPE_COLOURS["zero_filled"])
        ax.bar(k, count, color=colour, width=0.7, linewidth=0)
        ax.text(k, count + 0.3, str(int(count)),
                ha="center", va="bottom", fontsize=6)

    # Threshold line
    ax.axvline(min_slides_threshold - 0.5, color="#D55E00",
               lw=1.0, ls="--", label=f"min_slides = {min_slides_threshold}")
    ax.set_xlabel("Number of slides carrying the gene")
    ax.set_ylabel("Number of custom genes")
    ax.set_title("Custom gene slide-count distribution")
    ax.set_xticks(range(1, n_slides + 1))
    ax.legend(frameon=False, fontsize=6)

    # Annotation: kept vs dropped
    n_kept    = (overlap_df["n_slides"] >= min_slides_threshold).sum()
    n_dropped = (overlap_df["n_slides"] <  min_slides_threshold).sum()
    ax.text(0.97, 0.96,
            f"Kept: {n_kept}\nDropped: {n_dropped}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="#444444",
            bbox=dict(fc="white", alpha=0.7, pad=2, ec="none"))


# ===========================================================================
# Panel D: zero-fill count per slide after harmonisation
# ===========================================================================

def _plot_zerofill_bars(ax, adatas, slide_ids, conditions, used_harmonised: bool):
    """
    After partial_union harmonisation: number of zero-filled custom gene
    columns per slide, broken down by shared vs unique.
    """
    cond_colours_map = _condition_palette(conditions)
    x = np.arange(len(slide_ids))

    if not adatas or not used_harmonised or "zero_filled" not in adatas[0].var.columns:
        ax.text(0.5, 0.5, "Run harmonise() first\nto see zero-fill counts",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="#888888")
        ax.set_title("Zero-filled columns per slide")
        return

    zf_shared  = []
    zf_unique  = []
    for adata in adatas:
        zf_mask = adata.var["zero_filled"] if "zero_filled" in adata.var.columns else pd.Series(False, index=adata.var_names)
        pt      = adata.var["panel_type"]  if "panel_type"  in adata.var.columns else pd.Series("base",  index=adata.var_names)
        zf_shared.append(int((zf_mask & (pt == "custom_shared")).sum()))
        zf_unique.append(int((zf_mask & (pt == "custom_unique")).sum()))

    ax.bar(x, zf_shared,
           color=PANEL_TYPE_COLOURS["custom_shared"], width=0.6,
           linewidth=0, label="custom_shared (zero-filled)")
    ax.bar(x, zf_unique, bottom=zf_shared,
           color=PANEL_TYPE_COLOURS["custom_unique"], width=0.6,
           linewidth=0, label="custom_unique (zero-filled)")

    ax.set_xticks(x)
    ax.set_xticklabels(slide_ids, rotation=45, ha="right", fontsize=6)
    for tick, cond in zip(ax.get_xticklabels(), conditions):
        tick.set_color(cond_colours_map.get(cond, "black"))

    ax.set_ylabel("Zero-filled gene columns")
    ax.set_title("Zero-fill impact after harmonisation")
    ax.legend(frameon=False, fontsize=6, loc="upper right")

    # Ideal is zero — add a note
    max_zf = max((s + u for s, u in zip(zf_shared, zf_unique)), default=1)
    max_zf = max(max_zf, 1)
    ax.set_ylim(0, max_zf * 1.25)


# ===========================================================================
# Standalone convenience wrappers
# ===========================================================================

def plot_custom_gene_heatmap(
    adatas: list[ad.AnnData],
    slide_ids: list[str],
    conditions: list[str],
    registry,
    output_dir: Path = Path("figures_output"),
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """Standalone version of panel B only (custom gene presence heatmap)."""
    apply_nature_style()
    matrix = registry.custom_overlap_matrix(adatas, slide_ids)

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.6, max(3.0, len(matrix) * 0.18)))
    _plot_presence_heatmap(ax, matrix, slide_ids, conditions, fig)
    fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig13b_custom_gene_heatmap", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out
