"""
pipeline.py
-----------
Top-level orchestrator for the Xenium DGE pipeline.

Usage
-----
    from src.config import PipelineConfig
    from src.pipeline import XeniumDGEPipeline

    cfg = PipelineConfig(
        condition_a_dir = "data/control_run",
        condition_b_dir = "data/treatment_run",
        condition_a_label = "Control",
        condition_b_label = "Treatment",
        output_dir = "figures_output",
    )
    pipe = XeniumDGEPipeline(cfg)
    pipe.run()

The pipeline saves:
    - Preprocessed AnnData (.h5ad) for reproducibility
    - All 8 Nature-grade figures
    - DGE results table (CSV)
    - A plain-text run summary
"""

import logging
import time
from pathlib import Path

import anndata as ad
import pandas as pd

from src.config import PipelineConfig
from src.xenium_loader import load_two_conditions
from src.preprocessing import full_preprocessing_pipeline
from src.dge_analysis import run_dge
from src import figures as fig_module
from src import figures_galanin_resistance as fig_gal_module
from src import figures_spatial_domains as fig_sd_module
from src import galanin_resistance as gal_analysis
from src import spatial_domain_detection as sd_module

# Library code should never call basicConfig() -- that is the caller's
# responsibility.  A NullHandler prevents "No handler found" warnings when the
# library is used without any logging configuration.
logging.getLogger("XeniumDGEPipeline").addHandler(logging.NullHandler())
logger = logging.getLogger("XeniumDGEPipeline")


def setup_log_file(output_dir: Path, log_name: str = "pipeline_run.log") -> Path:
    """
    Attach a FileHandler to the root logger so that every module's log
    messages (at INFO level and above) are mirrored to a log file in output_dir.

    Noisy third-party loggers (fontTools.subset, harmonypy, numba, PIL) are
    silenced to WARNING in the file to keep it readable -- they each emit
    dozens to hundreds of low-value INFO lines per run.

    Safe to call multiple times -- duplicate FileHandlers are not added.
    Returns the path of the log file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_name

    root_logger = logging.getLogger()
    # Avoid adding a second handler if already attached to this file
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve():
            return log_path

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    # Silence high-volume third-party loggers in the file only.
    # fontTools.subset: ~50 INFO lines per saved PDF (Arial font subsetting).
    # harmonypy: verbose iteration logs redundant with our own summary message.
    # anndata: "storing '...' as categorical" FutureWarning on every h5ad write.
    # numba/PIL/matplotlib: can emit thousands of lines during JIT / font caching.
    _quiet = [
        "fontTools", "fontTools.subset", "fontTools.otlLib",
        "harmonypy",
        "anndata", "anndata._io",
        "numba", "numba.core",
        "PIL", "PIL.Image",
        "matplotlib", "matplotlib.font_manager",
    ]
    for _name in _quiet:
        logging.getLogger(_name).setLevel(logging.WARNING)

    # Session-start marker so multiple runs in one file are easy to distinguish.
    import datetime
    file_handler.stream.write(
        f"\n{'='*70}\n"
        f"  SESSION START  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*70}\n"
    )
    file_handler.stream.flush()

    logger.info("Log file: %s", log_path)
    return log_path


class XeniumDGEPipeline:
    """
    End-to-end Xenium spatial DGE pipeline.

    Parameters
    ----------
    cfg:
        PipelineConfig dataclass with all analysis parameters.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.adata: ad.AnnData | None = None
        self.dge_results: pd.DataFrame | None = None
        self.domain_degs: pd.DataFrame | None = None
        self._t0 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "XeniumDGEPipeline":
        """Execute the full pipeline and save outputs."""
        self._t0 = time.time()

        # Mirror all log output to a file in the output directory so users
        # can review the complete run after the fact.
        setup_log_file(self.cfg.output_dir)

        logger.info("=" * 60)
        logger.info("Xenium DGE Pipeline  |  %s vs %s",
                    self.cfg.condition_a_label, self.cfg.condition_b_label)
        logger.info("=" * 60)

        self.load_data()
        self.preprocess()
        self.run_spatial_domains()
        self.run_dge()
        self.make_figures()
        self.save_results()
        self._log_summary()
        return self

    def load_data(self) -> "XeniumDGEPipeline":
        """Step 1: Load and concatenate both Xenium runs."""
        logger.info("Step 1/4: Loading data …")
        self.adata = load_two_conditions(
            dir_a=self.cfg.condition_a_dir,
            dir_b=self.cfg.condition_b_dir,
            label_a=self.cfg.condition_a_label,
            label_b=self.cfg.condition_b_label,
        )
        return self

    def preprocess(self) -> "XeniumDGEPipeline":
        """Step 2: QC, normalisation, PCA, Harmony, UMAP, clustering."""
        logger.info("Step 2/4: Preprocessing …")

        cache_path = self.cfg.cache_dir / "adata_preprocessed.h5ad"
        if cache_path.exists():
            logger.info("Loading cached preprocessed AnnData from %s", cache_path)
            self.adata = ad.read_h5ad(cache_path)
        else:
            if self.adata is None:
                self.load_data()
            self.adata = full_preprocessing_pipeline(self.adata, self.cfg)
            self.adata.write_h5ad(cache_path)
            logger.info("Preprocessed AnnData cached to %s", cache_path)

        return self

    def run_spatial_domains(self) -> "XeniumDGEPipeline":
        """Step 2b: Spatial domain detection (optional)."""
        if not self.cfg.run_spatial_domains:
            logger.info("Spatial domain detection skipped (run_spatial_domains=False).")
            return self

        if self.adata is None:
            self.preprocess()

        logger.info(
            "Step 2b: Spatial domain detection (lambda=%.2f, res=%.2f) …",
            self.cfg.lambda_spatial, self.cfg.spatial_domain_resolution,
        )
        try:
            self.adata, self.domain_degs = sd_module.run_spatial_domain_pipeline(
                self.adata,
                lambda_spatial=self.cfg.lambda_spatial,
                resolution=self.cfg.spatial_domain_resolution,
                n_spatial_neighbors=self.cfg.n_spatial_neighbors,
                min_fragment_cells=self.cfg.spatial_domain_min_cells,
                domain_key=self.cfg.spatial_domain_key,
                run_degs=self.cfg.spatial_domain_degs,
                random_state=self.cfg.random_state,
            )

            # Export domain DEGs
            if self.domain_degs is not None:
                csv_path = self.cfg.output_dir / "spatial_domain_degs.csv"
                self.domain_degs.to_csv(csv_path, index=False)
                logger.info("Spatial domain DEGs saved to %s", csv_path)
        except Exception:
            logger.exception(
                "Spatial domain detection failed; continuing without domains."
            )
            self.domain_degs = None

        return self

    def run_dge(self) -> "XeniumDGEPipeline":
        """Step 3: Differential gene expression."""
        logger.info("Step 3/4: Running DGE (%s) …", self.cfg.dge_method)

        if self.adata is None:
            self.preprocess()

        dge_kwargs = {}
        if self.cfg.dge_method == "pydeseq2":
            dge_kwargs["min_cells"] = self.cfg.dge_min_cells
        elif self.cfg.dge_method == "stringent_wilcoxon":
            # Auto-detect number of replicates and adjust consistency threshold
            rep_key = "slide_id"
            if rep_key in self.adata.obs.columns:
                for cond in [self.cfg.condition_a_label, self.cfg.condition_b_label]:
                    n_reps = self.adata.obs.loc[
                        self.adata.obs[self.cfg.dge_group_key] == cond, rep_key
                    ].nunique()
                    if n_reps < 3:
                        dge_kwargs["min_consistent_replicates"] = max(1, n_reps)
                        logger.warning(
                            "Only %d replicate(s) for '%s'; reducing "
                            "min_consistent_replicates to %d.",
                            n_reps, cond, dge_kwargs["min_consistent_replicates"],
                        )
                        break

        self.dge_results = run_dge(
            self.adata,
            method        = self.cfg.dge_method,
            condition_key = self.cfg.dge_group_key,
            condition_a   = self.cfg.condition_a_label,
            condition_b   = self.cfg.condition_b_label,
            **dge_kwargs,
        )

        # Save raw results
        csv_path = self.cfg.output_dir / "dge_results.csv"
        self.dge_results.to_csv(csv_path, index=False)
        logger.info("DGE results saved to %s", csv_path)
        return self

    def make_figures(self) -> "XeniumDGEPipeline":
        """Step 4: Generate all 8 Nature-grade figures."""
        logger.info("Step 4/4: Generating figures …")

        if self.adata is None or self.dge_results is None:
            raise RuntimeError("Call preprocess() and run_dge() before make_figures().")

        out = self.cfg.output_dir
        fmt = self.cfg.figure_format
        dpi = self.cfg.dpi
        ck = self.cfg.dge_group_key
        clk = self.cfg.cluster_key

        # run_dge() guarantees canonical column names: 'log2fc' and 'pval_adj'.
        lfc_col = "log2fc"
        p_col   = "pval_adj"
        g_col   = "gene" if "gene" in self.dge_results.columns else self.dge_results.columns[0]

        # Figure 1: QC
        fig_module.plot_qc(
            self.adata, condition_key=ck, output_dir=out, fmt=fmt, dpi=dpi
        )

        # Figure 2: UMAP
        fig_module.plot_umap(
            self.adata, condition_key=ck, cluster_key=clk,
            output_dir=out, fmt=fmt, dpi=dpi,
        )

        # Figure 3: Spatial clusters
        fig_module.plot_spatial_clusters(
            self.adata, cluster_key=clk, condition_key=ck,
            output_dir=out, spot_size=self.cfg.spot_size, fmt=fmt, dpi=dpi,
            representative_slides=self.cfg.representative_slides,
        )

        # Figure 4: Marker gene dot plot
        fig_module.plot_dotplot(
            self.adata, cluster_key=clk, output_dir=out, fmt=fmt, dpi=dpi
        )

        # Figure 5: Volcano
        fig_module.plot_volcano(
            self.dge_results,
            condition_a=self.cfg.condition_a_label,
            condition_b=self.cfg.condition_b_label,
            log2fc_col=lfc_col,
            padj_col=p_col,
            log2fc_thresh=self.cfg.dge_log2fc_threshold,
            pval_thresh=self.cfg.dge_pval_threshold,
            n_label=self.cfg.n_top_dge_genes,
            output_dir=out, fmt=fmt, dpi=dpi,
        )

        # Figure 6: DGE heatmap
        fig_module.plot_dge_heatmap(
            self.adata, self.dge_results,
            condition_key=ck,
            log2fc_col=lfc_col, padj_col=p_col,
            log2fc_thresh=self.cfg.dge_log2fc_threshold,
            pval_thresh=self.cfg.dge_pval_threshold,
            output_dir=out, fmt=fmt, dpi=dpi,
        )

        # Figure 7: Spatial expression of top DGE genes
        sig = self.dge_results.dropna(subset=[p_col, lfc_col])
        sig = sig[
            (sig[p_col] < self.cfg.dge_pval_threshold)
            & (sig[lfc_col].abs() > self.cfg.dge_log2fc_threshold)
        ]
        top_genes = sig.nlargest(6, lfc_col)[g_col].tolist()
        if not top_genes:
            logger.warning(
                "Figure 7 skipped — no genes pass the DGE thresholds "
                "(pval_adj < %.2g, |log2FC| > %.2g). "
                "Try relaxing dge_pval_threshold or dge_log2fc_threshold.",
                self.cfg.dge_pval_threshold, self.cfg.dge_log2fc_threshold,
            )
        else:
            fig_module.plot_spatial_expression(
                self.adata, genes=top_genes, condition_key=ck,
                spot_size=self.cfg.spot_size, output_dir=out, fmt=fmt, dpi=dpi,
                representative_slides=self.cfg.representative_slides,
            )

        # Figure 8: Summary composite (a=UMAP cluster, b=composition, c=ADULT spatial, d=AGED spatial)
        fig_module.plot_summary_panel(
            self.adata, self.dge_results,
            condition_key=ck, cluster_key=clk,
            representative_slides=self.cfg.representative_slides,
            log2fc_col=lfc_col, padj_col=p_col,
            log2fc_thresh=self.cfg.dge_log2fc_threshold,
            pval_thresh=self.cfg.dge_pval_threshold,
            output_dir=out, fmt=fmt, dpi=dpi,
        )

        # ── Spatial domain figures (Fig SD1-SD5) ──────────────────────────
        self._make_spatial_domain_figures(out, fmt, dpi, ck, clk)

        # ── Galanin resistance analysis (Fig 19-25) ─────────────────────
        self._make_galanin_figures(out, fmt, dpi, ck)

        logger.info("All figures saved to %s/", out)
        return self

    def _make_spatial_domain_figures(self, out, fmt, dpi, ck, clk):
        """Generate spatial domain figures (Fig SD1-SD5) if domains were computed."""
        sdk = self.cfg.spatial_domain_key
        if sdk not in self.adata.obs.columns:
            logger.info("Spatial domain figures skipped — no spatial domains computed.")
            return

        logger.info("Generating spatial domain figures (Fig SD1-SD5) …")
        rep = self.cfg.representative_slides
        spot = self.cfg.spot_size

        fig_sd_module.plot_spatial_domains(
            self.adata, domain_key=sdk, condition_key=ck,
            output_dir=out, spot_size=spot, fmt=fmt, dpi=dpi,
            representative_slides=rep,
        )
        fig_sd_module.plot_domain_composition(
            self.adata, domain_key=sdk, condition_key=ck,
            output_dir=out, fmt=fmt, dpi=dpi,
        )
        if self.domain_degs is not None:
            fig_sd_module.plot_domain_markers(
                self.adata, self.domain_degs, domain_key=sdk,
                output_dir=out, fmt=fmt, dpi=dpi,
            )
        fig_sd_module.plot_domain_vs_leiden(
            self.adata, domain_key=sdk, cluster_key=clk,
            condition_key=ck, output_dir=out, spot_size=spot,
            fmt=fmt, dpi=dpi, representative_slides=rep,
        )
        logger.info("Spatial domain figures saved.")

    def _make_galanin_figures(self, out, fmt, dpi, ck):
        """Generate galanin resistance figures (Fig 19-25) if Gal is in panel."""
        gene_status = gal_analysis._check_genes(self.adata)
        if not gene_status.get("Gal", False):
            logger.info(
                "Galanin resistance figures skipped — Gal not in gene panel."
            )
            return

        logger.info("Generating galanin resistance figures (Fig 19-25) …")

        # Pre-compute analysis columns
        gal_analysis.compute_resistance_index(self.adata, store_in_obs=True)
        gal_analysis.classify_coexpression(self.adata, store_in_obs=True)
        gal_analysis.niche_receptor_score(self.adata, k=15, store_in_obs=True)

        common = dict(condition_key=ck, output_dir=out, fmt=fmt, dpi=dpi)
        rep = self.cfg.representative_slides
        spot = self.cfg.spot_size

        fig_gal_module.plot_gal_spatial_maps(
            self.adata, spot_size=spot, representative_slides=rep, **common,
        )
        fig_gal_module.plot_gal_expression_and_resistance(
            self.adata, **common,
        )
        fig_gal_module.plot_gal_coexpression(
            self.adata, spot_size=spot, representative_slides=rep, **common,
        )
        fig_gal_module.plot_gal_regional(
            self.adata, **common,
        )
        fig_gal_module.plot_gal_niche(
            self.adata, k=15, spot_size=spot, representative_slides=rep, **common,
        )
        fig_gal_module.plot_gal_proximity(
            self.adata, **common,
        )
        fig_gal_module.plot_gal_resistance_summary(
            self.adata, k=15, spot_size=spot, representative_slides=rep, **common,
        )

        # Export galanin analysis tables
        coexpr_df = gal_analysis.coexpression_proportions(
            self.adata, condition_key=ck,
        )
        coexpr_df.to_csv(out / "gal_coexpression_proportions.csv", index=False)

        regional_df = gal_analysis.regional_expression_summary(
            self.adata, condition_key=ck,
        )
        if not regional_df.empty:
            regional_df.to_csv(out / "gal_regional_expression.csv", index=False)

        logger.info("Galanin resistance figures and tables saved.")

    def save_results(self) -> "XeniumDGEPipeline":
        """Write the final AnnData to disk."""
        out_h5ad = self.cfg.output_dir / "adata_final.h5ad"
        if self.adata is not None:
            self.adata.write_h5ad(out_h5ad)
            logger.info("Final AnnData saved to %s", out_h5ad)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_summary(self):
        elapsed = time.time() - self._t0 if self._t0 is not None else 0
        n_cells = self.adata.n_obs if self.adata is not None else "N/A"
        n_genes = self.adata.n_vars if self.adata is not None else "N/A"
        n_clusters = (
            self.adata.obs[self.cfg.cluster_key].nunique()
            if self.adata is not None and self.cfg.cluster_key in self.adata.obs
            else "N/A"
        )

        p_col   = "pval_adj"
        lfc_col = "log2fc"
        n_sig = (
            (
                (self.dge_results[p_col] < self.cfg.dge_pval_threshold)
                & (self.dge_results[lfc_col].abs() > self.cfg.dge_log2fc_threshold)
            ).sum()
            if self.dge_results is not None
            else "N/A"
        )

        # Spatial domain info (if computed)
        sdk = self.cfg.spatial_domain_key
        n_domains = "N/A"
        domain_coherence = "N/A"
        if self.adata is not None and sdk in self.adata.obs.columns:
            n_domains = self.adata.obs[sdk].nunique()
            domain_coherence = self.adata.uns.get("spatial_domain_coherence", "N/A")
            if isinstance(domain_coherence, float):
                domain_coherence = f"{domain_coherence:.3f}"

        summary = f"""
========================================================
XENIUM DGE PIPELINE SUMMARY
========================================================
Conditions     : {self.cfg.condition_a_label} vs {self.cfg.condition_b_label}
Cells retained : {n_cells}
Genes retained : {n_genes}
Clusters       : {n_clusters}
Spatial domains: {n_domains} (coherence: {domain_coherence})
DGE method     : {self.cfg.dge_method}
Significant    : {n_sig} genes (|log2FC| > {self.cfg.dge_log2fc_threshold}, adj-p < {self.cfg.dge_pval_threshold})
Elapsed        : {elapsed:.0f} s
Output         : {self.cfg.output_dir}/
========================================================
"""
        logger.info(summary)
        summary_path = self.cfg.output_dir / "run_summary.txt"
        summary_path.write_text(summary)
