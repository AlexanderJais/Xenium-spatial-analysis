"""
figures_galanin_resistance.py
-----------------------------
Nature-grade figures for galanin resistance analysis (Fig 19-25).
Extends the existing figure pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from src.figures import (
    apply_nature_style, _savefig, _panel_label,
    DOUBLE, SINGLE, WONG, CONDITION_COLOURS,
)
from src.galanin_resistance import (
    GAL_GENE, GALR1_GENE, GALR3_GENE, GAL_SYSTEM_GENES,
    _get_expr_vector,
)

logger = logging.getLogger(__name__)

# Shared colour map for expression intensity (dark background)
_EXPR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "grey_red", ["#2A2A2A", "#7B1010", "#CC2222", "#FF6B35", "#FFD166"], N=256,
)


def _resolve_slides(adata, condition_key, representative_slides):
    """Pick one representative slide per condition."""
    slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
    if representative_slides is None:
        representative_slides = {}
        if slide_col:
            for cond in sorted(adata.obs[condition_key].unique()):
                slides = sorted(
                    adata.obs.loc[adata.obs[condition_key] == cond, slide_col].unique()
                )
                if slides:
                    representative_slides[cond] = slides[0]
    return representative_slides, slide_col


# ===================================================================
# Fig 19: Spatial expression maps — Gal, Galr1, Galr3 x condition
# ===================================================================

def plot_gal_spatial_maps(
    adata: ad.AnnData,
    condition_key: str = "condition",
    representative_slides: Optional[dict] = None,
    spot_size: float = 3.0,
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 19 — Spatial transcript maps of Gal, Galr1, Galr3.

    Layout: 2 rows (ADULT, AGED) x 3 columns (Gal, Galr1, Galr3).
    Each panel shows cell positions colour-coded by expression level.
    """
    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    rep_slides, slide_col = _resolve_slides(
        adata, condition_key, representative_slides
    )
    conditions = sorted(adata.obs[condition_key].unique())
    genes = GAL_SYSTEM_GENES

    fig, axes = plt.subplots(
        len(conditions), len(genes),
        figsize=(DOUBLE * 1.1, DOUBLE * 0.75),
        squeeze=False,
    )

    for ri, cond in enumerate(conditions):
        cond_mask = adata.obs[condition_key] == cond
        sid = rep_slides.get(cond)
        if sid and slide_col:
            idx = np.where(cond_mask & (adata.obs[slide_col] == sid))[0]
        else:
            idx = np.where(cond_mask)[0]

        xy = adata.obsm["spatial"][idx] if "spatial" in adata.obsm else None

        for ci, gene in enumerate(genes):
            ax = axes[ri, ci]
            if xy is None or len(idx) == 0 or gene not in adata.var_names:
                ax.text(
                    0.5, 0.5,
                    f"{'No spatial' if xy is None else 'No cells'}"
                    if gene in adata.var_names else f"{gene} not in panel",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="#888",
                )
                ax.axis("off")
                continue

            expr = _get_expr_vector(adata, gene)[idx]
            pos = expr > 0
            vmax = float(np.percentile(expr[pos], 99)) if pos.sum() > 10 else 1.0

            # Plot zero-expression cells as grey background
            ax.scatter(
                xy[~pos, 0], xy[~pos, 1],
                c="#333333", s=spot_size * 0.3, alpha=0.3,
                linewidths=0, rasterized=True,
            )
            sc = ax.scatter(
                xy[pos, 0], xy[pos, 1],
                c=expr[pos], cmap=_EXPR_CMAP, vmin=0, vmax=vmax,
                s=spot_size, alpha=0.85, linewidths=0, rasterized=True,
            )

            ax.set_facecolor("#1A1A1A")
            ax.set_aspect("equal")
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False,
            )
            ax.spines[:].set_visible(False)

            if ri == 0:
                ax.set_title(gene, fontsize=8, fontweight="bold", color="#EEE", pad=4)
            if ci == 0:
                ax.set_ylabel(cond, fontsize=7, color="#EEE", labelpad=6)

    # Panel labels
    labels = "abcdefghijklmnopqrstuvwxyz"
    for i, ax in enumerate(axes.ravel()):
        if i < len(labels):
            _panel_label(ax, labels[i])

    fig.suptitle(
        "Galanin system spatial expression (Gal / Galr1 / Galr3)",
        fontsize=9, y=1.02, color="black",
    )
    fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig19_gal_spatial_maps", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===================================================================
# Fig 20: Expression quantification + galanin resistance index
# ===================================================================

def plot_gal_expression_and_resistance(
    adata: ad.AnnData,
    condition_key: str = "condition",
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 20 — Expression quantification and resistance index.

    A: Violin plots of Gal, Galr1, Galr3 per-cell expression (adult vs aged)
       with Wilcoxon rank-sum p-values.
    B: Violin plot of galanin resistance index (GRI) per condition.
    """
    from scipy.stats import ranksums

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    from src.galanin_resistance import compute_resistance_index
    if "galanin_resistance_index" not in adata.obs.columns:
        compute_resistance_index(adata, store_in_obs=True)

    conditions = sorted(adata.obs[condition_key].unique())
    cond_a, cond_b = conditions[0], conditions[-1]
    cond_col = {
        c: CONDITION_COLOURS.get(c, WONG[i % len(WONG)])
        for i, c in enumerate(conditions)
    }

    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE * 1.05, SINGLE * 0.85))

    # --- Panels A-C: per-gene violins ---
    genes = GAL_SYSTEM_GENES
    for gi, gene in enumerate(genes):
        ax = axes[gi]
        expr = _get_expr_vector(adata, gene)
        data_by_cond = {}
        for cond in conditions:
            mask = adata.obs[condition_key] == cond
            data_by_cond[cond] = expr[mask.values]

        positions = list(range(len(conditions)))
        for pi, cond in enumerate(conditions):
            vals = data_by_cond[cond]
            if len(vals) < 5:
                continue
            parts = ax.violinplot(
                vals[vals > 0] if (vals > 0).sum() > 5 else vals,
                positions=[pi], showmeans=False, showmedians=True, widths=0.7,
            )
            for pc in parts.get("bodies", []):
                pc.set_facecolor(cond_col[cond])
                pc.set_alpha(0.6)
            for key in ("cmins", "cmaxes", "cmedians", "cbars"):
                if key in parts:
                    parts[key].set_color(cond_col[cond])
                    parts[key].set_linewidth(0.8)

        # Wilcoxon test
        va, vb = data_by_cond[cond_a], data_by_cond[cond_b]
        if len(va) >= 3 and len(vb) >= 3:
            _, p = ranksums(vb, va)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(0.5, 0.97, sig, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, fontweight="bold")

        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, fontsize=6)
        ax.set_title(gene, fontsize=8, fontweight="bold")
        ax.set_ylabel("log-norm expr" if gi == 0 else "", fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        _panel_label(ax, chr(ord("a") + gi))

    # --- Panel D: Resistance index violin ---
    ax = axes[3]
    gri = adata.obs["galanin_resistance_index"].values
    for pi, cond in enumerate(conditions):
        mask = (adata.obs[condition_key] == cond).values
        vals = gri[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 5:
            continue
        parts = ax.violinplot(vals, positions=[pi], showmedians=True, widths=0.7)
        for pc in parts.get("bodies", []):
            pc.set_facecolor(cond_col[cond])
            pc.set_alpha(0.6)
        for key in ("cmins", "cmaxes", "cmedians", "cbars"):
            if key in parts:
                parts[key].set_color(cond_col[cond])
                parts[key].set_linewidth(0.8)

    va = gri[(adata.obs[condition_key] == cond_a).values]
    vb = gri[(adata.obs[condition_key] == cond_b).values]
    va, vb = va[~np.isnan(va)], vb[~np.isnan(vb)]
    if len(va) >= 3 and len(vb) >= 3:
        _, p = ranksums(vb, va)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(0.5, 0.97, sig, transform=ax.transAxes,
                ha="center", va="top", fontsize=7, fontweight="bold")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=6)
    ax.set_title("Resistance index", fontsize=8, fontweight="bold")
    ax.set_ylabel("GRI", fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "d")

    # Legend
    handles = [mpatches.Patch(color=cond_col[c], label=c) for c in conditions]
    axes[0].legend(handles=handles, frameon=False, fontsize=6, loc="upper left")

    fig.suptitle(
        "Galanin system expression & resistance index (adult vs aged)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(pad=0.5)
    out = _savefig(fig, output_dir / "fig20_gal_expression_resistance", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===================================================================
# Fig 21: Spatial co-expression maps + proportions
# ===================================================================

# Co-expression category colours
_COEXPR_COLOURS = {
    "Gal+Galr1+": "#E69F00",   # orange
    "Gal+Galr3+": "#009E73",   # green
    "Gal+only":   "#D55E00",   # vermillion
    "Galr1+only": "#56B4E9",   # sky blue
    "Galr3+only": "#0072B2",   # blue
    "Galr1+Galr3+": "#CC79A7", # pink
    "Negative":   "#444444",   # dark grey
}


def plot_gal_coexpression(
    adata: ad.AnnData,
    condition_key: str = "condition",
    representative_slides: Optional[dict] = None,
    spot_size: float = 3.0,
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 21 — Spatial co-expression of Gal with receptors.

    A-B: Spatial maps coloured by co-expression status (adult / aged).
    C:   Stacked bar showing proportions of Gal+ cells co-expressing receptors.
    """
    from src.galanin_resistance import classify_coexpression, coexpression_proportions

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    if "gal_coexpr_status" not in adata.obs.columns:
        classify_coexpression(adata, store_in_obs=True)

    rep_slides, slide_col = _resolve_slides(
        adata, condition_key, representative_slides
    )
    conditions = sorted(adata.obs[condition_key].unique())
    # Spatial panels limited to first 2 conditions (adult/aged)
    spatial_conds = conditions[:2]

    fig = plt.figure(figsize=(DOUBLE * 1.1, DOUBLE * 0.55))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.8], wspace=0.35)

    # --- Panels A-B: spatial co-expression maps ---
    for ci, cond in enumerate(spatial_conds):
        ax = fig.add_subplot(gs[ci])
        cond_mask = adata.obs[condition_key] == cond
        sid = rep_slides.get(cond)
        if sid and slide_col:
            idx = np.where(cond_mask & (adata.obs[slide_col] == sid))[0]
        else:
            idx = np.where(cond_mask)[0]

        if "spatial" not in adata.obsm or len(idx) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", fontsize=7, color="#888")
            ax.axis("off")
            continue

        xy = adata.obsm["spatial"][idx]
        labels = adata.obs["gal_coexpr_status"].values[idx]

        # Plot negative cells first (background), then positives on top
        for cat in ["Negative", "Galr1+only", "Galr3+only", "Galr1+Galr3+",
                     "Gal+only", "Gal+Galr3+", "Gal+Galr1+"]:
            mask = labels == cat
            if mask.sum() == 0:
                continue
            s = spot_size * 0.3 if cat == "Negative" else spot_size
            a = 0.2 if cat == "Negative" else 0.8
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                c=_COEXPR_COLOURS.get(cat, "#888"),
                s=s, alpha=a, linewidths=0, rasterized=True,
            )

        ax.set_facecolor("#1A1A1A")
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)
        ax.set_title(cond, fontsize=8, color="#EEE", pad=3)
        _panel_label(ax, chr(ord("a") + ci))

    # --- Panel C: stacked bar proportions ---
    ax_c = fig.add_subplot(gs[2])
    coexpr_df = coexpression_proportions(adata, condition_key=condition_key)
    cdf = coexpr_df.set_index("condition")

    # Use pct_any_receptor (non-overlapping with Gal+only) for stacked bar
    x = np.arange(len(conditions))
    pct_rec = cdf.loc[conditions, "pct_any_receptor"].values
    pct_only = 100.0 - pct_rec

    ax_c.bar(x, pct_rec, color=_COEXPR_COLOURS["Gal+Galr1+"],
             width=0.5, label="Gal+receptor+")
    ax_c.bar(x, pct_only, bottom=pct_rec, color=_COEXPR_COLOURS["Gal+only"],
             width=0.5, label="Gal+only")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(conditions, fontsize=6)
    ax_c.set_ylabel("% of Gal+ cells", fontsize=6)
    ax_c.set_title("Receptor co-expression", fontsize=8)
    ax_c.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax_c.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_c, "c")

    # Shared legend for spatial panels
    legend_handles = [
        mpatches.Patch(color=_COEXPR_COLOURS[c], label=c)
        for c in ["Gal+Galr1+", "Gal+Galr3+", "Gal+only",
                   "Galr1+only", "Galr3+only", "Negative"]
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=6,
        fontsize=5.5, frameon=False, bbox_to_anchor=(0.45, -0.06),
    )

    fig.suptitle(
        "Galanin-receptor co-expression (adult vs aged)", fontsize=9, y=1.02,
    )
    fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig21_gal_coexpression", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===================================================================
# Fig 22: Regional / cluster breakdown of galanin system
# ===================================================================

def plot_gal_regional(
    adata: ad.AnnData,
    condition_key: str = "condition",
    region_key: str = "cell_type",
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 22 — Regional breakdown of Gal / Galr1 / Galr3 expression.

    A: Grouped dot plot (size = % expressing, colour = mean expression)
       per region x gene, split by condition.
    B: Mean resistance index per region, adult vs aged (horizontal bar).

    Falls back to leiden clusters if cell_type not available.
    """
    from src.galanin_resistance import regional_expression_summary, compute_resistance_index

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    if region_key not in adata.obs.columns:
        region_key = "leiden" if "leiden" in adata.obs.columns else None
    if region_key is None:
        logger.warning("No region key for fig22; generating placeholder.")
        fig, ax = plt.subplots(figsize=(SINGLE, 2))
        ax.text(0.5, 0.5, "No region annotations available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig22_gal_regional", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out

    if "galanin_resistance_index" not in adata.obs.columns:
        compute_resistance_index(adata, store_in_obs=True)

    df = regional_expression_summary(adata, region_key=region_key,
                                     condition_key=condition_key)
    if df.empty:
        fig, ax = plt.subplots(figsize=(SINGLE, 2))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        out = _savefig(fig, output_dir / "fig22_gal_regional", fmt=fmt, dpi=dpi)
        plt.close(fig)
        return out

    conditions = sorted(df["condition"].unique())
    regions = sorted(df["region"].unique())
    genes = [g for g in GAL_SYSTEM_GENES if g in df["gene"].unique()]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(DOUBLE * 1.15, max(3.5, len(regions) * 0.35)),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    # --- Panel A: dot plot ---
    cond_col = {
        c: CONDITION_COLOURS.get(c, WONG[i % len(WONG)])
        for i, c in enumerate(conditions)
    }
    y_labels = []
    y_idx = 0
    for ri, region in enumerate(regions):
        for gi, gene in enumerate(genes):
            for ci, cond in enumerate(conditions):
                row = df[(df["region"] == region) & (df["gene"] == gene)
                         & (df["condition"] == cond)]
                if row.empty:
                    continue
                pct = float(row["pct_expressing"].iloc[0])
                x_pos = gi * len(conditions) + ci
                ax_a.scatter(
                    x_pos, y_idx,
                    s=max(pct * 1.5, 3),
                    c=cond_col[cond], alpha=0.8,
                    edgecolors="white", linewidths=0.3,
                )
        y_labels.append(str(region))
        y_idx += 1

    ax_a.set_yticks(range(len(regions)))
    ax_a.set_yticklabels(y_labels, fontsize=6)
    x_ticks = []
    x_labels = []
    for gi, gene in enumerate(genes):
        for ci, cond in enumerate(conditions):
            x_ticks.append(gi * len(conditions) + ci)
            x_labels.append(f"{gene}\n{cond[:3]}")
    ax_a.set_xticks(x_ticks)
    ax_a.set_xticklabels(x_labels, fontsize=5, rotation=45, ha="right")
    ax_a.set_title("Expression by region", fontsize=8)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.invert_yaxis()
    _panel_label(ax_a, "a")

    # --- Panel B: resistance index per region ---
    gri_data = []
    for region in regions:
        for cond in conditions:
            mask = ((adata.obs[region_key] == region)
                    & (adata.obs[condition_key] == cond))
            vals = adata.obs.loc[mask, "galanin_resistance_index"].dropna()
            gri_data.append({
                "region": region, "condition": cond, "mean_gri": float(vals.mean()) if len(vals) > 0 else 0.0,
            })
    gri_df = pd.DataFrame(gri_data)

    y_pos = np.arange(len(regions))
    bar_h = 0.35
    for ci, cond in enumerate(conditions):
        sub = gri_df[gri_df["condition"] == cond].set_index("region")
        vals = [sub.loc[r, "mean_gri"] if r in sub.index else 0 for r in regions]
        offset = -bar_h / 2 if ci == 0 else bar_h / 2
        ax_b.barh(
            y_pos + offset, vals, height=bar_h,
            color=cond_col[cond], alpha=0.8, label=cond,
        )

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(regions, fontsize=6)
    ax_b.set_xlabel("Mean GRI", fontsize=6)
    ax_b.set_title("Resistance index", fontsize=8)
    ax_b.legend(fontsize=5.5, frameon=False)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.invert_yaxis()
    _panel_label(ax_b, "b")

    fig.suptitle("Galanin system: regional analysis", fontsize=9, y=1.02)
    fig.tight_layout(pad=0.5)
    out = _savefig(fig, output_dir / "fig22_gal_regional", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===================================================================
# Fig 23: Neighborhood / niche receptor availability
# ===================================================================

def plot_gal_niche(
    adata: ad.AnnData,
    condition_key: str = "condition",
    k: int = 15,
    representative_slides: Optional[dict] = None,
    spot_size: float = 4.0,
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 23 — Niche receptor availability around Gal+ cells.

    A: Box plot of niche receptor score by condition.
    B-C: Spatial heatmaps of niche score on tissue (adult / aged).
    """
    from scipy.stats import ranksums
    from src.galanin_resistance import niche_receptor_score

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    if "niche_receptor_score" not in adata.obs.columns:
        niche_receptor_score(adata, k=k, store_in_obs=True)

    conditions = sorted(adata.obs[condition_key].unique())
    cond_col = {
        c: CONDITION_COLOURS.get(c, WONG[i % len(WONG)])
        for i, c in enumerate(conditions)
    }

    fig = plt.figure(figsize=(DOUBLE, SINGLE * 0.9))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.7, 1, 1], wspace=0.3)

    # --- Panel A: box plot ---
    ax_a = fig.add_subplot(gs[0])
    box_data, box_labels, box_colors = [], [], []
    for ci, cond in enumerate(conditions):
        mask = (adata.obs[condition_key] == cond).values
        vals = adata.obs["niche_receptor_score"].values[mask]
        vals = vals[~np.isnan(vals)]
        box_data.append(vals)
        box_labels.append(cond)
        box_colors.append(cond_col[cond])

    bp = ax_a.boxplot(
        box_data, positions=range(len(conditions)), widths=0.5,
        patch_artist=True, showfliers=False,
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color("#333")
            line.set_linewidth(0.8)

    # Wilcoxon test
    if len(box_data) >= 2 and len(box_data[0]) >= 3 and len(box_data[1]) >= 3:
        _, p = ranksums(box_data[1], box_data[0])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ymax = max(np.percentile(d, 95) for d in box_data if len(d) > 0)
        ax_a.text(0.5, ymax * 1.05, sig, ha="center", fontsize=8, fontweight="bold")

    ax_a.set_xticks(range(len(conditions)))
    ax_a.set_xticklabels(conditions, fontsize=6)
    ax_a.set_ylabel("Niche receptor score", fontsize=6)
    ax_a.set_title("Receptor availability\naround Gal+ cells", fontsize=7)
    ax_a.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_a, "a")

    # --- Panels B-C: spatial heatmaps ---
    rep_slides, slide_col = _resolve_slides(
        adata, condition_key, representative_slides
    )
    niche_cmap = mcolors.LinearSegmentedColormap.from_list(
        "niche", ["#2A2A2A", "#0072B2", "#56B4E9", "#FFD166"], N=256,
    )

    # Spatial panels limited to first 2 conditions (GridSpec has 3 columns)
    spatial_conds = conditions[:2]
    for ci, cond in enumerate(spatial_conds):
        ax = fig.add_subplot(gs[1 + ci])
        cond_mask = adata.obs[condition_key] == cond
        sid = rep_slides.get(cond)
        if sid and slide_col:
            idx = np.where(cond_mask & (adata.obs[slide_col] == sid))[0]
        else:
            idx = np.where(cond_mask)[0]

        if "spatial" not in adata.obsm or len(idx) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", fontsize=7, color="#888")
            ax.axis("off")
            continue

        xy = adata.obsm["spatial"][idx]
        scores = adata.obs["niche_receptor_score"].values[idx]
        valid = ~np.isnan(scores)

        # Background: non-Gal+ cells
        ax.scatter(
            xy[~valid, 0], xy[~valid, 1],
            c="#333", s=spot_size * 0.2, alpha=0.15,
            linewidths=0, rasterized=True,
        )
        # Gal+ cells coloured by niche score
        if valid.sum() > 0:
            sc = ax.scatter(
                xy[valid, 0], xy[valid, 1],
                c=scores[valid], cmap=niche_cmap, vmin=0, vmax=1,
                s=spot_size, alpha=0.85, linewidths=0, rasterized=True,
            )
            cax = ax.inset_axes([1.02, 0.1, 0.04, 0.8])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label("Score", fontsize=5, color="#CCC")
            cb.ax.tick_params(labelsize=5, colors="#CCC")

        ax.set_facecolor("#1A1A1A")
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)
        ax.set_title(f"Niche score — {cond}", fontsize=7, color="#EEE", pad=3)
        _panel_label(ax, chr(ord("b") + ci))

    fig.suptitle(
        f"Receptor availability in Gal+ cell neighborhood (k={k})",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig23_gal_niche", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===================================================================
# Fig 24: Ligand-receptor spatial proximity
# ===================================================================

def plot_gal_proximity(
    adata: ad.AnnData,
    condition_key: str = "condition",
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 24 — Nearest-neighbour distance from Gal+ to receptor+ cells.

    A: KDE of Gal-to-Galr1 distances, adult vs aged.
    B: KDE of Gal-to-Galr3 distances, adult vs aged.
    C: Combined box plot summary.
    """
    from scipy.stats import ranksums, gaussian_kde
    from src.galanin_resistance import ligand_receptor_distances

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    dist_data = ligand_receptor_distances(adata, condition_key=condition_key)
    conditions = sorted(set(dist_data["condition"]))
    cond_col = {
        c: CONDITION_COLOURS.get(c, WONG[i % len(WONG)])
        for i, c in enumerate(conditions)
    }

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, SINGLE * 0.85))

    # --- Panels A-B: KDE distance distributions ---
    receptors = [("gal_to_galr1", "Galr1"), ("gal_to_galr3", "Galr3")]
    for pi, (dist_key, receptor_name) in enumerate(receptors):
        ax = axes[pi]
        dists = dist_data[dist_key]
        conds = dist_data["condition"]

        for cond in conditions:
            mask = conds == cond
            vals = dists[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 5:
                continue
            kde = gaussian_kde(vals, bw_method=0.3)
            x = np.linspace(0, float(np.percentile(vals, 99)), 200)
            ax.fill_between(x, kde(x), alpha=0.4, color=cond_col[cond], label=cond)
            ax.plot(x, kde(x), color=cond_col[cond], lw=0.8)

        # Wilcoxon test
        cond_list = list(conditions)
        if len(cond_list) >= 2:
            va = dists[conds == cond_list[0]]
            vb = dists[conds == cond_list[1]]
            va, vb = va[~np.isnan(va)], vb[~np.isnan(vb)]
            if len(va) >= 3 and len(vb) >= 3:
                _, p = ranksums(vb, va)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.text(0.95, 0.95, f"p={p:.2g} {sig}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6)

        ax.set_xlabel(f"Distance to nearest {receptor_name}+ (um)", fontsize=6)
        ax.set_ylabel("Density" if pi == 0 else "", fontsize=6)
        ax.set_title(f"Gal+ -> {receptor_name}+", fontsize=8)
        ax.legend(fontsize=5.5, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        _panel_label(ax, chr(ord("a") + pi))

    # --- Panel C: combined box plot ---
    ax_c = axes[2]
    box_positions = []
    box_data_list = []
    box_colors_list = []
    tick_labels = []
    pos = 0

    for dist_key, receptor_name in receptors:
        dists = dist_data[dist_key]
        conds = dist_data["condition"]
        for cond in conditions:
            mask = conds == cond
            vals = dists[mask]
            vals = vals[~np.isnan(vals)]
            box_data_list.append(vals)
            box_positions.append(pos)
            box_colors_list.append(cond_col[cond])
            tick_labels.append(f"{receptor_name}\n{cond[:3]}")
            pos += 1
        pos += 0.5  # gap between receptor groups

    if box_data_list:
        bp = ax_c.boxplot(
            box_data_list, positions=box_positions, widths=0.45,
            patch_artist=True, showfliers=False,
        )
        for patch, col in zip(bp["boxes"], box_colors_list):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        for el in ("whiskers", "caps", "medians"):
            for line in bp[el]:
                line.set_color("#333")
                line.set_linewidth(0.8)

    ax_c.set_xticks(box_positions)
    ax_c.set_xticklabels(tick_labels, fontsize=5)
    ax_c.set_ylabel("Distance (um)", fontsize=6)
    ax_c.set_title("Summary", fontsize=8)
    ax_c.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_c, "c")

    fig.suptitle(
        "Ligand-receptor spatial proximity (Gal+ to receptor+ cells)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(pad=0.5)
    out = _savefig(fig, output_dir / "fig24_gal_proximity", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out


# ===================================================================
# Fig 25: Composite "galanin resistance" summary figure
# ===================================================================

def plot_gal_resistance_summary(
    adata: ad.AnnData,
    condition_key: str = "condition",
    k: int = 15,
    representative_slides: Optional[dict] = None,
    spot_size: float = 3.0,
    output_dir: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Fig 25 — Composite summary of galanin resistance in ageing.

    6-panel figure:
      A: Spatial Gal expression (ADULT)
      B: Spatial Gal expression (AGED)
      C: Violin — Gal, Galr1, Galr3 expression by condition
      D: Violin — Resistance index by condition
      E: Box — Niche receptor score by condition
      F: KDE — Gal-to-receptor distance by condition
    """
    from scipy.stats import ranksums, gaussian_kde
    from src.galanin_resistance import (
        compute_resistance_index, niche_receptor_score,
        ligand_receptor_distances,
    )

    if output_dir is None:
        output_dir = Path("figures_output")
    output_dir = Path(output_dir)
    apply_nature_style()

    # Ensure computed columns exist
    if "galanin_resistance_index" not in adata.obs.columns:
        compute_resistance_index(adata, store_in_obs=True)
    if "niche_receptor_score" not in adata.obs.columns:
        niche_receptor_score(adata, k=k, store_in_obs=True)

    conditions = sorted(adata.obs[condition_key].unique())
    cond_a, cond_b = conditions[0], conditions[-1]
    cond_col = {
        c: CONDITION_COLOURS.get(c, WONG[i % len(WONG)])
        for i, c in enumerate(conditions)
    }
    rep_slides, slide_col = _resolve_slides(
        adata, condition_key, representative_slides
    )

    fig = plt.figure(figsize=(DOUBLE * 1.2, DOUBLE * 0.95))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # --- A & B: Spatial Gal maps ---
    expr_cmap = _EXPR_CMAP
    gal_expr = _get_expr_vector(adata, GAL_GENE)
    pos_vals = gal_expr[gal_expr > 0]
    vmax = float(np.percentile(pos_vals, 99)) if len(pos_vals) > 10 else 1.0

    for ci, cond in enumerate(conditions[:2]):
        ax = fig.add_subplot(gs[0, ci])
        cond_mask = adata.obs[condition_key] == cond
        sid = rep_slides.get(cond)
        if sid and slide_col:
            idx = np.where(cond_mask & (adata.obs[slide_col] == sid))[0]
        else:
            idx = np.where(cond_mask)[0]

        if "spatial" in adata.obsm and len(idx) > 0:
            xy = adata.obsm["spatial"][idx]
            e = gal_expr[idx]
            ax.scatter(
                xy[:, 0], xy[:, 1], c=e, cmap=expr_cmap,
                vmin=0, vmax=vmax, s=spot_size, alpha=0.85,
                linewidths=0, rasterized=True,
            )
        ax.set_facecolor("#1A1A1A")
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)
        ax.set_title(f"Gal — {cond}", fontsize=7, color="#EEE", pad=3)
        _panel_label(ax, chr(ord("a") + ci))

    # --- C: per-gene violin (compact) ---
    ax_c = fig.add_subplot(gs[0, 2])
    genes = GAL_SYSTEM_GENES
    for gi, gene in enumerate(genes):
        expr = _get_expr_vector(adata, gene)
        for ci, cond in enumerate(conditions[:2]):
            mask = (adata.obs[condition_key] == cond).values
            vals = expr[mask]
            vals = vals[vals > 0] if (vals > 0).sum() > 5 else vals
            if len(vals) < 5:
                continue
            x_pos = gi * 2.5 + ci
            parts = ax_c.violinplot(
                vals, positions=[x_pos], showmedians=True, widths=0.8,
            )
            for pc in parts.get("bodies", []):
                pc.set_facecolor(cond_col[cond])
                pc.set_alpha(0.55)
            for key in ("cmins", "cmaxes", "cmedians", "cbars"):
                if key in parts:
                    parts[key].set_color(cond_col[cond])
                    parts[key].set_linewidth(0.6)

    xticks = [gi * 2.5 + 0.5 for gi in range(len(genes))]
    ax_c.set_xticks(xticks)
    ax_c.set_xticklabels(genes, fontsize=6)
    ax_c.set_ylabel("log-norm", fontsize=6)
    ax_c.set_title("Expression", fontsize=7)
    ax_c.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_c, "c")

    # --- D: Resistance index violin ---
    ax_d = fig.add_subplot(gs[1, 0])
    gri = adata.obs["galanin_resistance_index"].values
    for ci, cond in enumerate(conditions[:2]):
        mask = (adata.obs[condition_key] == cond).values
        vals = gri[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 5:
            continue
        parts = ax_d.violinplot(vals, positions=[ci], showmedians=True, widths=0.65)
        for pc in parts.get("bodies", []):
            pc.set_facecolor(cond_col[cond])
            pc.set_alpha(0.55)
        for key in ("cmins", "cmaxes", "cmedians", "cbars"):
            if key in parts:
                parts[key].set_color(cond_col[cond])
                parts[key].set_linewidth(0.6)

    va = gri[(adata.obs[condition_key] == cond_a).values]
    vb = gri[(adata.obs[condition_key] == cond_b).values]
    va, vb = va[~np.isnan(va)], vb[~np.isnan(vb)]
    if len(va) >= 3 and len(vb) >= 3:
        _, p = ranksums(vb, va)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax_d.text(0.5, 0.97, sig, transform=ax_d.transAxes,
                  ha="center", va="top", fontsize=7, fontweight="bold")
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(conditions[:2], fontsize=6)
    ax_d.set_ylabel("GRI", fontsize=6)
    ax_d.set_title("Resistance index", fontsize=7)
    ax_d.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_d, "d")

    # --- E: Niche receptor box plot ---
    ax_e = fig.add_subplot(gs[1, 1])
    box_data = []
    for cond in conditions[:2]:
        mask = (adata.obs[condition_key] == cond).values
        vals = adata.obs["niche_receptor_score"].values[mask]
        box_data.append(vals[~np.isnan(vals)])

    if all(len(d) > 0 for d in box_data):
        bp = ax_e.boxplot(
            box_data, positions=[0, 1], widths=0.5,
            patch_artist=True, showfliers=False,
        )
        for patch, cond in zip(bp["boxes"], conditions[:2]):
            patch.set_facecolor(cond_col[cond])
            patch.set_alpha(0.6)
        for el in ("whiskers", "caps", "medians"):
            for line in bp[el]:
                line.set_color("#333")
                line.set_linewidth(0.7)

    ax_e.set_xticks([0, 1])
    ax_e.set_xticklabels(conditions[:2], fontsize=6)
    ax_e.set_ylabel("Niche score", fontsize=6)
    ax_e.set_title("Receptor in niche", fontsize=7)
    ax_e.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_e, "e")

    # --- F: Distance KDE ---
    ax_f = fig.add_subplot(gs[1, 2])
    dist_data = ligand_receptor_distances(adata, condition_key=condition_key)
    dists = dist_data["gal_to_galr1"]
    conds = dist_data["condition"]

    for cond in conditions[:2]:
        mask = conds == cond
        vals = dists[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals, bw_method=0.3)
        x = np.linspace(0, float(np.percentile(vals, 99)), 200)
        ax_f.fill_between(x, kde(x), alpha=0.35, color=cond_col[cond], label=cond)
        ax_f.plot(x, kde(x), color=cond_col[cond], lw=0.8)

    ax_f.set_xlabel("Dist to Galr1+ (um)", fontsize=6)
    ax_f.set_ylabel("Density", fontsize=6)
    ax_f.set_title("L-R proximity", fontsize=7)
    ax_f.legend(fontsize=5.5, frameon=False)
    ax_f.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_f, "f")

    fig.suptitle(
        "Galanin resistance in the ageing hypothalamus — summary",
        fontsize=9, fontweight="bold", y=1.02,
    )
    fig.tight_layout(pad=0.4)
    out = _savefig(fig, output_dir / "fig25_gal_resistance_summary", fmt=fmt, dpi=dpi)
    plt.close(fig)
    return out
