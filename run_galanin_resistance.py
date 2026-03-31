#!/usr/bin/env python3
"""
run_galanin_resistance.py
-------------------------
Entry point for the galanin resistance analysis (Fig 19-25).

Reads a preprocessed AnnData (.h5ad) produced by the main pipeline
and generates all galanin resistance figures.

Usage:
    python run_galanin_resistance.py --input adata_mbh_final.h5ad [--output figures_output_gal]

Requires: Gal (and ideally Galr1, Galr3) in the gene panel.
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("galanin_resistance")


def main():
    parser = argparse.ArgumentParser(
        description="Galanin resistance analysis (Fig 19-25)",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to preprocessed AnnData (.h5ad) from the main pipeline.",
    )
    parser.add_argument(
        "--output", "-o", default="figures_output_gal",
        help="Output directory for figures and tables (default: figures_output_gal).",
    )
    parser.add_argument("--fmt", default="pdf", help="Figure format (pdf/png/svg).")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--k-niche", type=int, default=15,
                        help="k for spatial neighbour niche analysis.")
    parser.add_argument("--condition-key", default="condition",
                        help="obs column with condition labels.")
    parser.add_argument("--region-key", default="cell_type",
                        help="obs column for regional breakdown (cell_type or leiden).")
    parser.add_argument("--spot-size", type=float, default=3.0,
                        help="Dot size for spatial plots.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading AnnData from %s ...", args.input)
    adata = ad.read_h5ad(args.input)
    logger.info("Loaded: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # Check gene availability
    from src.galanin_resistance import GAL_SYSTEM_GENES, _check_genes
    present = _check_genes(adata)
    for gene, found in present.items():
        status = "OK" if found else "MISSING (will be treated as zero)"
        logger.info("  %s: %s", gene, status)

    if not present["Gal"]:
        logger.error("Gal is not in the panel. Cannot run galanin analysis.")
        sys.exit(1)

    # ── Run analyses ─────────────────────────────────────────────────────
    from src.galanin_resistance import (
        compute_resistance_index,
        classify_coexpression,
        niche_receptor_score,
    )

    logger.info("Computing resistance index ...")
    compute_resistance_index(adata, store_in_obs=True)

    logger.info("Classifying co-expression ...")
    classify_coexpression(adata, store_in_obs=True)

    logger.info("Computing niche receptor scores (k=%d) ...", args.k_niche)
    niche_receptor_score(adata, k=args.k_niche, store_in_obs=True)

    # ── Generate figures ─────────────────────────────────────────────────
    from src.figures_galanin_resistance import (
        plot_gal_spatial_maps,
        plot_gal_expression_and_resistance,
        plot_gal_coexpression,
        plot_gal_regional,
        plot_gal_niche,
        plot_gal_proximity,
        plot_gal_resistance_summary,
    )

    fig_kwargs = dict(
        condition_key=args.condition_key,
        output_dir=output_dir,
        fmt=args.fmt,
        dpi=args.dpi,
    )

    logger.info("Generating Fig 18: spatial expression maps ...")
    plot_gal_spatial_maps(adata, spot_size=args.spot_size, **fig_kwargs)

    logger.info("Generating Fig 19: expression & resistance index ...")
    plot_gal_expression_and_resistance(adata, **fig_kwargs)

    logger.info("Generating Fig 20: co-expression maps ...")
    plot_gal_coexpression(adata, spot_size=args.spot_size, **fig_kwargs)

    logger.info("Generating Fig 21: regional breakdown ...")
    plot_gal_regional(adata, region_key=args.region_key, **fig_kwargs)

    logger.info("Generating Fig 22: niche analysis ...")
    plot_gal_niche(adata, k=args.k_niche, spot_size=args.spot_size, **fig_kwargs)

    logger.info("Generating Fig 23: ligand-receptor proximity ...")
    plot_gal_proximity(adata, **fig_kwargs)

    logger.info("Generating Fig 24: composite summary ...")
    plot_gal_resistance_summary(
        adata, k=args.k_niche, spot_size=args.spot_size, **fig_kwargs,
    )

    # ── Export analysis tables ───────────────────────────────────────────
    from src.galanin_resistance import (
        coexpression_proportions,
        regional_expression_summary,
        ligand_receptor_distances,
        test_coexpression_proportion,
    )

    logger.info("Exporting analysis tables ...")
    coexpr_df = coexpression_proportions(adata, condition_key=args.condition_key)
    coexpr_df.to_csv(output_dir / "gal_coexpression_proportions.csv", index=False)

    regional_df = regional_expression_summary(
        adata, region_key=args.region_key, condition_key=args.condition_key,
    )
    if not regional_df.empty:
        regional_df.to_csv(output_dir / "gal_regional_expression.csv", index=False)

    chi2_result = test_coexpression_proportion(
        adata, condition_key=args.condition_key,
    )
    logger.info(
        "Co-expression chi2 test: chi2=%.2f, p=%.4g",
        chi2_result["chi2"], chi2_result["pval"],
    )

    logger.info("Done. All outputs saved to %s/", output_dir)


if __name__ == "__main__":
    main()
