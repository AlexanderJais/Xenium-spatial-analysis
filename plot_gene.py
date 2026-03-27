#!/usr/bin/env python3
"""
plot_gene.py — Spatial expression map for any gene across all 8 MBH slides.

Usage
-----
    python plot_gene.py Wfs1
    python plot_gene.py Wfs1 --out ~/Desktop/Wfs1_spatial.pdf
    python plot_gene.py Wfs1 Cd68 Trem2          # multiple genes, one figure each
    python plot_gene.py Wfs1 --fmt png --dpi 150  # raster output

Loads the preprocessed AnnData from cache — no rerunning of the full pipeline.
Produces one PDF (or PNG) per gene: rows = gene (1 row), columns = all 8 slides.
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("plot_gene")

# ── Default cache and output paths ───────────────────────────────────────────
_HERE        = Path(__file__).parent
_DEFAULT_CACHE = Path.home() / "xenium_dge_output_cache" / "adata_mbh_preprocessed.h5ad"
_DEFAULT_OUT   = Path.home() / "xenium_dge_output"

# ── Custom grey → red colormap (same as pipeline fig7) ───────────────────────
GREY_RED = mcolors.LinearSegmentedColormap.from_list(
    "grey_red",
    ["#2A2A2A", "#7B1010", "#CC2222", "#FF6B35", "#FFD166"],
    N=256,
)


def _get_lognorm(adata):
    import scipy.sparse as sp
    if "lognorm" in adata.layers:
        X = adata.layers["lognorm"]
    else:
        X = adata.X
    if sp.issparse(X):
        return X
    return np.asarray(X)


def plot_gene(
    gene: str,
    adata,
    output_path: Path,
    fmt: str = "pdf",
    dpi: int = 200,
    spot_size: float = 2.5,
    condition_key: str = "condition",
    slide_key: str = "slide_id",
):
    """Generate spatial expression map for one gene across all slides."""
    if gene not in adata.var_names:
        # Case-insensitive search
        hits = [g for g in adata.var_names if g.lower() == gene.lower()]
        if hits:
            gene = hits[0]
            logger.info("Case-corrected gene name to '%s'", gene)
        else:
            close = [g for g in adata.var_names if gene.lower() in g.lower()]
            msg = f"Gene '{gene}' not found in dataset."
            if close:
                msg += f" Did you mean: {', '.join(close[:5])}?"
            raise ValueError(msg)

    gi   = list(adata.var_names).index(gene)
    X    = _get_lognorm(adata)
    _xi  = X[:, gi]
    expr = np.array(_xi.todense()).ravel() if hasattr(_xi, "todense") else np.array(_xi).ravel()
    vmax = float(np.percentile(expr[expr > 0], 99)) if (expr > 0).any() else 1.0

    conditions = adata.obs[condition_key].cat.categories.tolist()
    if slide_key not in adata.obs.columns:
        logger.warning("No '%s' column — plotting by condition only.", slide_key)
        slides_per_cond = {c: [c] for c in conditions}
        slide_key = condition_key
    else:
        slides_per_cond = {
            cond: sorted(adata.obs.loc[adata.obs[condition_key] == cond, slide_key].unique())
            for cond in conditions
        }

    all_cols = [(cond, sid) for cond in conditions for sid in slides_per_cond[cond]]
    n_cols   = len(all_cols)

    # Figure: single row, n_cols panels + colorbar
    panel_w = 2.2
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(panel_w * n_cols, panel_w * 0.92),
        squeeze=False,
    )
    fig.patch.set_facecolor("#111111")

    sc = None  # initialise so colorbar guard works even if all panels have no data
    for col, (cond, sid) in enumerate(all_cols):
        ax   = axes[0, col]
        mask = (adata.obs[condition_key] == cond) & (adata.obs[slide_key] == sid)
        sub  = adata[mask]

        if sub.n_obs == 0 or "spatial" not in sub.obsm:
            ax.set_facecolor("#111111")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="#888")
            ax.axis("off")
            continue

        xy = sub.obsm["spatial"]
        _e = X[mask, :][:, gi]
        e  = np.array(_e.todense()).ravel() if hasattr(_e, "todense") else np.array(_e).ravel()

        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=e, cmap=GREY_RED,
            vmin=0, vmax=vmax,
            s=spot_size, alpha=0.85,
            linewidths=0, rasterized=True,
        )
        ax.set_facecolor("#1A1A1A")
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)

        # Slide label at top
        label = f"{cond}\n{sid}"
        ax.set_title(label, fontsize=5.5, color="#DDDDDD", pad=3)

    # Colorbar anchored to right of last panel
    # sc is only assigned when at least one panel had data; check before use.
    if sc is not None:
        cax  = axes[0, -1].inset_axes([1.03, 0.1, 0.06, 0.8])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("log-norm", fontsize=6, color="#CCCCCC", labelpad=3)
        cbar.ax.tick_params(labelsize=5, colors="#CCCCCC", width=0.4, length=2)
        cbar.outline.set_edgecolor("#444444")
        cbar.outline.set_linewidth(0.4)

    fig.suptitle(
        f"{gene}  —  spatial expression across all MBH slides",
        fontsize=9, color="#EEEEEE", y=1.02,
    )
    fig.tight_layout(pad=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved: %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Spatial expression map for any gene across all 8 MBH slides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "genes", nargs="+",
        help="Gene name(s) to plot (e.g. Wfs1  or  Wfs1 Cd68 Trem2)",
    )
    parser.add_argument(
        "--cache", default=str(_DEFAULT_CACHE),
        help=f"Path to preprocessed AnnData h5ad (default: {_DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Output path for a single gene.  "
            "For multiple genes, pass a directory instead (one file per gene). "
            f"Default: {_DEFAULT_OUT}/<gene>_spatial.pdf"
        ),
    )
    parser.add_argument(
        "--fmt", default="pdf", choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Resolution for raster outputs (default: 200)",
    )
    parser.add_argument(
        "--spot-size", type=float, default=2.5,
        help="Scatter dot size (default: 2.5)",
    )
    parser.add_argument(
        "--condition-key", default="condition",
        help="obs column for condition (default: condition)",
    )
    parser.add_argument(
        "--slide-key", default="slide_id",
        help="obs column for slide identity (default: slide_id)",
    )
    args = parser.parse_args()

    # ── Load AnnData ─────────────────────────────────────────────────────────
    cache = Path(args.cache)
    if not cache.exists():
        # Try raw AnnData as fallback
        raw_cache = cache.parent / "adata_mbh_raw.h5ad"
        if raw_cache.exists():
            logger.warning(
                "Preprocessed cache not found; using raw AnnData (%s). "
                "Spatial coordinates will be available but expression is raw counts.",
                raw_cache,
            )
            cache = raw_cache
        else:
            sys.exit(
                f"ERROR: No AnnData cache found at:\n"
                f"  {cache}\n"
                f"  {raw_cache}\n"
                f"Run the full pipeline first, or pass --cache <path>."
            )

    logger.info("Loading AnnData from %s …", cache)
    import anndata as ad
    adata = ad.read_h5ad(cache)
    logger.info("Loaded: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # ── Resolve output path(s) ────────────────────────────────────────────────
    out_arg = Path(args.out) if args.out else None
    if len(args.genes) == 1 and out_arg is not None and out_arg.suffix:
        # Single gene → single explicit file path (has a file extension)
        out_paths = [out_arg]
    else:
        # Multiple genes, or --out is a directory (existing or to-be-created)
        if out_arg is not None and not out_arg.suffix:
            # --out points to a directory
            out_dir = out_arg
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = _DEFAULT_OUT
        out_paths = [
            out_dir / f"{g}_spatial.{args.fmt}"
            for g in args.genes
        ]

    # ── Plot ──────────────────────────────────────────────────────────────────
    failed = []
    for gene, out_path in zip(args.genes, out_paths):
        try:
            plot_gene(
                gene        = gene,
                adata       = adata,
                output_path = out_path,
                fmt         = args.fmt,
                dpi         = args.dpi,
                spot_size   = args.spot_size,
                condition_key = args.condition_key,
                slide_key     = args.slide_key,
            )
        except ValueError as e:
            logger.error("%s", e)
            failed.append(gene)

    if failed:
        sys.exit(f"\nFailed genes: {', '.join(failed)}")


if __name__ == "__main__":
    main()
