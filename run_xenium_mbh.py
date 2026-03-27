"""
run_xenium_mbh.py
-----------------
End-to-end pipeline for the AGED vs ADULT mediobasal hypothalamus study.
This is the single production entry point — run directly, via the
Streamlit web app, or via the Tkinter launcher (launcher.py).

Study design
------------
  - 4 AGED brain sections  (biological replicates)
  - 4 ADULT brain sections (biological replicates)
  - Base panel: Xenium_mBrain_v1_1 (247 genes, all slides)
  - Custom genes: 7-12 per slide (vary between slides)
  - ROI: mediobasal hypothalamus (MBH), drawn interactively per slide
  - DGE: PyDESeq2 pseudobulk (proper 4 vs 4 replicates)

Workflow
--------
  Step 1: Define slide manifest (paths + conditions)
  Step 2: Interactive ROI drawing (one window per slide)
          -- skipped if roi_cache/ already contains all 8 ROIs
  Step 3: Load all slides, harmonise panels, apply MBH ROI
  Step 4: QC, normalisation, Harmony integration, UMAP, Leiden clustering
  Step 5: Cell type annotation (base panel marker genes)
  Step 6: Global DGE: AGED vs ADULT (PyDESeq2 pseudobulk)
  Step 7: Per-cluster DGE
  Step 8: Spatial statistics (Moran's I, neighbourhood enrichment)
  Step 9: Generate all 11 Nature-grade figures

Usage
-----
  # Subsequent runs (ROI is saved; will not re-open GUI)
  python run_xenium_mbh.py

  # Force ROI redraw
  python run_xenium_mbh.py --redraw-roi

  # Skip interactive ROI (use preset ellipse from data generation)
  python run_xenium_mbh.py --no-roi-gui
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Suppress third-party FutureWarnings that appear on every run and carry no
# actionable information for the pipeline user.
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", message=".*palette.*without.*hue.*")
warnings.filterwarnings("ignore", message=".*flavor.*igraph.*")
warnings.filterwarnings("ignore", message=".*default backend.*leiden.*")
# tight_layout UserWarning is expected when external legends are placed outside axes
warnings.filterwarnings("ignore", message=".*tight_layout.*Axes that are not compatible.*")

from src.config import PipelineConfig
from src.multislide_loader import MultiSlideLoader, SlideManifest, load_aged_adult_study
from src.panel_registry import PanelRegistry
from src.roi_selector import ROISelector, interactive_roi_session
from src.preprocessing import full_preprocessing_pipeline
from src.cell_type_annotation import annotate_cell_types, assign_labels_from_markers
from src.dge_analysis import run_dge
from src.cluster_dge import run_cluster_dge
from src.spatial_stats import morans_i_scan, neighborhood_enrichment
from src.pipeline import XeniumDGEPipeline
from src import figures as fig_module
from src import figures_extended as fe
from src import figures_panel as fp

logger = logging.getLogger("AgedAdultPipeline")


# ===========================================================================
# Study configuration
# ===========================================================================

ROOT_DATA    = Path("data")
OUTPUT_DIR   = Path("figures_output_mbh")
CACHE_DIR    = Path(".cache_mbh")
ROI_CACHE    = Path("roi_cache")
BASE_PANEL   = Path("data/Xenium_mBrain_v1_1_metadata.csv")

AGED_SLIDES = [
    {"slide_id": "AGED_1",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_1"},
    {"slide_id": "AGED_2",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_2"},
    {"slide_id": "AGED_3",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_3"},
    {"slide_id": "AGED_4",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_4"},
]
ADULT_SLIDES = [
    {"slide_id": "ADULT_1", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_1"},
    {"slide_id": "ADULT_2", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_2"},
    {"slide_id": "ADULT_3", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_3"},
    {"slide_id": "ADULT_4", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_4"},
]
ALL_SLIDES = AGED_SLIDES + ADULT_SLIDES

# PipelineConfig for preprocessing / DGE parameters
CFG = PipelineConfig(
    condition_a_dir   = ROOT_DATA / "ADULT_1",   # placeholder (not used by multislide loader)
    condition_b_dir   = ROOT_DATA / "AGED_1",
    condition_a_label = "ADULT",
    condition_b_label = "AGED",
    output_dir        = OUTPUT_DIR,
    cache_dir         = CACHE_DIR,

    # QC
    min_counts         = 10,
    max_counts         = 2_000,
    min_genes          = 10,   # raised from 5; removes low-quality cells robustly
    max_genes          = 300,
    min_cells_per_gene = 5,

    # Preprocessing
    target_sum    = 100.0,   # Xenium-specific (Salas 2025 Nature Methods benchmark)
    n_top_genes   = 0,     # 0 = use all genes (recommended for Xenium targeted panels)
    filter_control_probes   = True,   # remove cells with control_probe_counts > 0
    filter_control_codewords = True,  # remove cells with control_codeword_counts > 0
    normalize_by_cell_area  = False,  # optional: correct for cell size variation
    n_pcs         = 30,

    # Integration
    harmony_key      = "slide_id",   # correct: batch = slide, not condition
    harmony_max_iter = 30,

    # UMAP / Clustering
    n_neighbors       = 12,    # lowered from 15 per Xenium benchmark recommendation
    umap_min_dist     = 0.3,
    leiden_resolution = 0.6,   # raised from 0.5 per Xenium benchmark recommendation

    # DGE: PyDESeq2 pseudobulk with real replicates
    dge_method           = "stringent_wilcoxon",  # see dge_analysis.stringent_wilcoxon_dge
    dge_group_key        = "condition",
    dge_log2fc_threshold = 1.0,    # matches stringent_wilcoxon internal lfc filter
    dge_pval_threshold   = 0.01,   # matches stringent_wilcoxon internal p filter
    n_top_dge_genes      = 25,

    # Figures
    dpi           = 300,
    figure_format = "pdf",
    spot_size     = 5.0,
)


# ===========================================================================
# Pipeline
# ===========================================================================

def main(redraw_roi: bool = False, no_roi_gui: bool = False, panel_mode: str = "partial_union", min_slides: int = 2):
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Reconfigure logging for main() so file handler captures everything.
    # basicConfig is a no-op if handlers already exist, so clear first.
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    from src.pipeline import setup_log_file
    log_path = setup_log_file(OUTPUT_DIR)
    logger.info("Log file: %s", log_path)

    # ------------------------------------------------------------------
    # Step 1: Build manifest
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("AGED vs ADULT MBH Xenium DGE Pipeline")
    logger.info("=" * 65)

    manifest = SlideManifest.from_dict(ALL_SLIDES)
    logger.info("Manifest:\n%s", manifest.summary().to_string(index=False))

    # ------------------------------------------------------------------
    # Step 2: ROI selection (interactive or preset)
    # ------------------------------------------------------------------
    registry     = PanelRegistry(BASE_PANEL)
    roi_selector = ROISelector(cache_dir=ROI_CACHE)

    missing_rois = [s for s in manifest.slide_ids if not roi_selector.has_roi(s)]

    if missing_rois and not no_roi_gui:
        logger.info(
            "Step 2: Interactive ROI selection for %d slides: %s",
            len(missing_rois), missing_rois,
        )
        logger.info(
            "A matplotlib window will open for each slide.\n"
            "  Polygon mode: left-click to add vertices, "
            "right-click or Enter to close, Backspace to undo last point.\n"
            "  The dashed ellipse is an anatomical MBH hint — "
            "draw your polygon to match the section anatomy.\n"
            "  ROIs are saved automatically to roi_cache/ after drawing."
        )
        # Load just the slides needing ROIs for the GUI
        from src.xenium_loader import load_xenium_run
        for entry in manifest:
            sid = entry["slide_id"]
            if sid not in missing_rois:
                continue
            tmp = load_xenium_run(entry["run_dir"], condition_label=entry["condition"])
            roi_selector.draw(
                adata=tmp, slide_id=sid,
                colour_key=None,
                mode="polygon",
                show_mbh_hint=True,
                force_redraw=redraw_roi,
                roi_name="MBH",
            )
    elif missing_rois and no_roi_gui:
        logger.info(
            "Step 2: Skipping ROI GUI (--no-roi-gui). "
            "Slides without ROI will be returned in full."
        )
    else:
        logger.info(
            "Step 2: All ROIs already saved in %s/. "
            "Pass --redraw-roi to redo.", ROI_CACHE,
        )

    # Summarise saved ROIs
    roi_table = roi_selector.list_rois()
    if not roi_table.empty:
        logger.info("Saved ROIs:\n%s", roi_table.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 3: Multi-slide load + panel harmonisation + ROI filter
    # ------------------------------------------------------------------
    logger.info("Step 3: Loading slides, harmonising panels, applying MBH ROI …")

    h5_cache = CACHE_DIR / "adata_mbh_raw.h5ad"
    loader   = None   # kept for Fig 13; None when loading from cache

    # Validate cache: a 0-cell AnnData means the ROIs were wrong when the
    # cache was written.  Delete it so the pipeline re-runs the full load.
    if h5_cache.exists():
        import anndata as ad
        _tmp = ad.read_h5ad(h5_cache)
        if _tmp.n_obs == 0:
            logger.warning(
                "Cached AnnData at %s has 0 cells — cache was written with "
                "invalid ROIs. Deleting cache and re-running the full load.",
                h5_cache,
            )
            h5_cache.unlink()

    if h5_cache.exists():
        import anndata as ad
        logger.info("Loading cached raw AnnData from %s", h5_cache)
        adata = ad.read_h5ad(h5_cache)
        if not adata.uns.get("roi_applied", True):
            logger.warning(
                "Cached AnnData was built WITHOUT ROI filtering (ROIs were invalid). "
                "Redraw ROIs in the app, then delete %s to rerun with MBH region.",
                h5_cache,
            )
    else:
        loader = MultiSlideLoader(
            manifest       = manifest,
            panel_registry = registry,
            roi_selector   = roi_selector,
            panel_mode     = panel_mode,
            min_slides     = min_slides,
            apply_roi      = True,
            output_dir     = OUTPUT_DIR,   # panel_validation.csv saved here
        )
        adata = loader.load_all()

        # Guard: if all ROIs selected 0 cells, stop immediately.
        # Silently falling back to the full tissue is wrong for MBH analysis —
        # it would mix in cortex, hippocampus, etc.
        if adata.n_obs == 0:
            raise ValueError(
                "\n"
                + "=" * 65 + "\n"
                "ROI ERROR: All slides returned 0 cells after ROI filtering.\n"
                "The saved ROI polygons do not overlap the tissue coordinates.\n\n"
                "HOW TO FIX:\n"
                "  1. Open the app -> ROI Manager\n"
                "  2. Click 'Delete ALL ROIs and start over'\n"
                "  3. For each slide, use the SLIDERS to frame the MBH region.\n"
                "     The scatter shows orange cells = what will be selected.\n"
                "     The Save button is disabled until cells > 0.\n"
                "  4. Delete the pipeline cache:\n"
                "        rm -rf " + str(CACHE_DIR) + "\n"
                "  5. Re-run the pipeline\n"
                + "=" * 65
            )
        adata.uns["roi_applied"] = True

        adata.write_h5ad(h5_cache)
        logger.info("Raw MBH AnnData cached to %s", h5_cache)

    logger.info("MBH AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)
    _log_cell_counts(adata)

    # ------------------------------------------------------------------
    # Step 4: Preprocessing (QC, norm, PCA, Harmony, UMAP, Leiden)
    # ------------------------------------------------------------------
    # Default representative slides for spatial figures.
    # These match the user's preferred sections for AGED (G_073_1 = AGED_3)
    # and ADULT (Q378_2 = ADULT_1).
    if CFG.representative_slides is None:
        CFG.representative_slides = {"AGED": "AGED_3", "ADULT": "ADULT_1"}

    logger.info("Step 4: Preprocessing …")

    pre_cache = CACHE_DIR / "adata_mbh_preprocessed.h5ad"

    # Validate and optionally invalidate the preprocessed cache
    if pre_cache.exists():
        import anndata as ad
        _tmp2 = ad.read_h5ad(pre_cache)
        if _tmp2.n_obs == 0:
            logger.warning("Preprocessed cache has 0 cells — deleting and reprocessing.")
            pre_cache.unlink()
        else:
            logger.info("Loading cached preprocessed AnnData from %s", pre_cache)
            adata = _tmp2

    # Run full preprocessing if no valid cache exists
    if not pre_cache.exists():
        adata = full_preprocessing_pipeline(adata, CFG)
        adata.write_h5ad(pre_cache)
        logger.info("Preprocessed AnnData cached to %s", pre_cache)

    # ------------------------------------------------------------------
    # Step 5: Cell type annotation
    # ------------------------------------------------------------------
    logger.info("Step 5: Cell type annotation …")
    try:
        # Data-driven annotation: scores each cluster's mean expression
        # against MBH_MARKERS (broad types) then MBH_SUBTYPE_MARKERS
        # (neuronal subtypes). Labels are derived from the actual data of
        # THIS run — not from a hardcoded cluster-number lookup that would
        # silently mislabel clusters whenever resolution, QC, or cell count
        # changes between runs.
        from src.cell_type_annotation import assign_labels_from_markers
        adata = assign_labels_from_markers(
            adata,
            cluster_key = CFG.cluster_key,
        )
    except Exception as e:
        logger.warning(
            "Data-driven annotation failed: %s — falling back to marker scoring.", e
        )
        try:
            adata = annotate_cell_types(
                adata,
                method          = "marker_scoring",
                min_score_delta = 0.03,
            )
        except Exception as e2:
            logger.warning("Marker scoring also failed: %s", e2)

    # Generate Supplementary Table 1 CSV (cluster -> cell type mapping)
    # Reads from adata.uns['cluster_label_map'] which was populated by
    # assign_labels_from_markers() from the actual data of this run.
    try:
        _cluster_col = CFG.cluster_key if CFG.cluster_key in adata.obs.columns else \
                       ("leiden" if "leiden" in adata.obs.columns else None)
        if _cluster_col is None:
            raise KeyError("No cluster column found in adata.obs")

        # Prefer the data-driven map; fall back to cell_type column if uns key absent
        label_map = adata.uns.get(
            "cluster_label_map",
            adata.obs.groupby(_cluster_col, observed=True)["cell_type"]
            .first().to_dict()
            if "cell_type" in adata.obs.columns else {},
        )

        supp_t1_rows = []
        for cid in sorted(label_map.keys(),
                          key=lambda x: (0, int(x)) if str(x).isdigit() else (1, x)):
            label = label_map[cid]
            mask  = adata.obs[_cluster_col].astype(str) == str(cid)
            n_adult = int((mask & (adata.obs["condition"] == "ADULT")).sum())
            n_aged  = int((mask & (adata.obs["condition"] == "AGED")).sum())
            supp_t1_rows.append({
                "cluster_id"   : cid,
                "cell_type"    : label,
                "n_cells_total": int(mask.sum()),
                "n_adult"      : n_adult,
                "n_aged"       : n_aged,
                "pct_total"    : round(100 * mask.sum() / adata.n_obs, 2),
            })
        import pandas as pd
        pd.DataFrame(supp_t1_rows).to_csv(
            OUTPUT_DIR / "supplementary_table1_cluster_annotations.csv", index=False
        )
        logger.info("Supplementary Table 1 CSV saved to %s/", OUTPUT_DIR)
    except Exception as e:
        logger.warning("Supp Table 1 generation failed: %s", e)

    # ------------------------------------------------------------------
    # Step 5b: Sub-cluster Cluster 15 (Oligodendrocyte maturing — mixed signal)
    # ------------------------------------------------------------------
    # Cluster 15 shows mixed Sox10+ (oligodendrocyte) / Gfap+ (astrocyte) /
    # Cobll1+ (endothelial) markers at the top-level Leiden resolution=0.5.
    # Sub-clustering at a higher resolution resolves whether this represents
    # a genuine transitional OL state, a glia boundary artefact, or mixed capture.
    try:
        from src.preprocessing import run_leiden
        import scanpy as sc

        mask_15 = adata.obs[CFG.cluster_key] == "15"
        if mask_15.sum() >= 50:
            logger.info(
                "Step 5b: Sub-clustering Cluster 15 (%d cells) at resolution 1.5 …",
                mask_15.sum(),
            )
            sub15 = adata[mask_15].copy()

            # Build a new KNN graph on the Harmony-corrected embedding for this subset
            rep = "X_pca_harmony" if "X_pca_harmony" in sub15.obsm else "X_pca"
            sc.pp.neighbors(
                sub15,
                n_neighbors=min(10, sub15.n_obs - 1),
                use_rep=rep,
                random_state=42,
            )
            sub15 = run_leiden(sub15, resolution=1.5, key_added="leiden_sub15")

            n_sub = sub15.obs["leiden_sub15"].nunique()
            logger.info("Cluster 15 sub-clusters: %d sub-clusters found", n_sub)

            # Write sub-cluster labels back into main adata
            sub_col = "cluster15_subtype"
            adata.obs[sub_col] = "—"  # default for all other clusters
            adata.obs.loc[mask_15, sub_col] = (
                "C15_sub" + sub15.obs["leiden_sub15"].astype(str)
            )
            adata.obs[sub_col] = adata.obs[sub_col].astype("category")

            # Score each sub-cluster with marker genes to suggest identity
            from src.cell_type_annotation import _get_lognorm
            sub_ids = sorted(sub15.obs["leiden_sub15"].unique(),
                             key=lambda x: int(x) if str(x).isdigit() else 0)

            marker_genes = {
                "OL":   ["Opalin", "Sox10", "Gjc3", "Zfp536"],
                "OPC":  ["Pdgfra", "Gpr17", "Sema3d"],
                "Ast":  ["Aqp4", "Gfap", "Ntsr2"],
                "Endo": ["Cldn5", "Cobll1", "Ly6a"],
            }
            logger.info("Cluster 15 sub-cluster marker summary:")
            X_sub = _get_lognorm(sub15)
            for sid in sub_ids:
                smask = sub15.obs["leiden_sub15"] == sid
                n_cells = smask.sum()
                scores = {}
                for ct, genes in marker_genes.items():
                    avail = [g for g in genes if g in sub15.var_names]
                    if avail:
                        gi = [list(sub15.var_names).index(g) for g in avail]
                        scores[ct] = float(X_sub[smask][:, gi].mean())
                    else:
                        scores[ct] = 0.0
                best = max(scores, key=scores.get)
                logger.info(
                    "  C15_sub%s  (%d cells)  best=%s  scores=%s",
                    sid, n_cells, best,
                    {k: f"{v:.3f}" for k, v in scores.items()},
                )

            # Save sub-cluster summary CSV
            sub_summary = (
                adata.obs[adata.obs[sub_col] != "—"][[sub_col, "condition"]]
                .groupby([sub_col, "condition"], observed=True)
                .size()
                .reset_index(name="n_cells")
            )
            sub_summary.to_csv(
                OUTPUT_DIR / "cluster15_subtype_counts.csv", index=False
            )
            logger.info(
                "Cluster 15 sub-cluster labels written to adata.obs['%s']. "
                "Summary: %s/cluster15_subtype_counts.csv",
                sub_col, OUTPUT_DIR,
            )
        else:
            logger.info(
                "Step 5b: Cluster 15 has only %d cells — skipping sub-clustering "
                "(need >= 50).", mask_15.sum(),
            )
    except Exception as e:
        logger.warning("Step 5b: Cluster 15 sub-clustering failed: %s", e)

    # ------------------------------------------------------------------
    # Step 6: Global DGE  (AGED vs ADULT)
    # ------------------------------------------------------------------
    logger.info("Step 6: Global DGE — AGED vs ADULT (method: %s) …", CFG.dge_method)

    dge = run_dge(
        adata,
        method                  = CFG.dge_method,
        condition_key           = "condition",
        condition_a             = "ADULT",
        condition_b             = "AGED",
        replicate_key           = "slide_id",          # 4 biological replicates per condition
        lfc_threshold           = CFG.dge_log2fc_threshold,   # forwarded to stringent_wilcoxon
        pval_threshold          = CFG.dge_pval_threshold,
    )
    lfc_col = "log2fc"
    p_col   = "pval_adj"
    g_col   = "gene" if "gene" in dge.columns else dge.columns[0]

    n_sig = ((dge[p_col] < CFG.dge_pval_threshold) & (dge[lfc_col].abs() > CFG.dge_log2fc_threshold)).sum()
    logger.info("Global DGE: %d significant genes", n_sig)
    dge.to_csv(OUTPUT_DIR / "global_dge_aged_vs_adult.csv", index=False)

    # ------------------------------------------------------------------
    # Step 7: Per-cluster DGE
    # ------------------------------------------------------------------
    logger.info("Step 7: Per-cluster DGE …")
    group_key = "cell_type" if "cell_type" in adata.obs else CFG.cluster_key
    try:
        cluster_dge = run_cluster_dge(
            adata,
            group_key             = group_key,
            condition_key         = "condition",
            condition_a           = "ADULT",
            condition_b           = "AGED",
            method                = CFG.dge_method,
            replicate_key         = "slide_id",          # 4 biological replicates per condition
            log2fc_thresh         = CFG.dge_log2fc_threshold,
            pval_thresh           = CFG.dge_pval_threshold,
            output_dir            = OUTPUT_DIR,
        )
    except Exception as e:
        logger.warning("Per-cluster DGE failed: %s", e)
        cluster_dge = None

    # ------------------------------------------------------------------
    # Step 8: Spatial statistics
    # ------------------------------------------------------------------
    logger.info("Step 8: Spatial statistics …")
    morans_df     = None
    neighborhood  = None

    if "spatial" in adata.obsm:
        # Use all panel genes for Moran's I — HVG selection is skipped for
        # targeted Xenium panels, so all genes are informative.
        morans_genes = adata.var_names.tolist()
        try:
            morans_df = morans_i_scan(adata, genes=morans_genes, n_neighbors=6)
            morans_df.to_csv(OUTPUT_DIR / "morans_i_mbh.csv", index=False)
        except Exception as e:
            logger.warning("Moran's I failed: %s", e)

        ct_key = "cell_type" if "cell_type" in adata.obs else CFG.cluster_key
        try:
            neighborhood = neighborhood_enrichment(
                adata, cell_type_key=ct_key, n_permutations=500
            )
        except Exception as e:
            logger.warning("Neighbourhood enrichment failed: %s", e)

    # ------------------------------------------------------------------
    # Step 9: Generate all figures
    # ------------------------------------------------------------------
    logger.info("Step 9: Generating figures …")

    # Inject into XeniumDGEPipeline for Figs 1-8
    pipe = XeniumDGEPipeline(CFG)
    pipe.adata       = adata
    pipe.dge_results = dge
    pipe.make_figures()

    # Fig 9: cell types
    if "cell_type" in adata.obs:
        fe.plot_cell_type_panel(
            adata,
            cell_type_key        = "cell_type",
            condition_key        = "condition",
            output_dir           = OUTPUT_DIR,
            spot_size            = CFG.spot_size,
            fmt                  = CFG.figure_format,
            dpi                  = CFG.dpi,
            representative_slides= CFG.representative_slides,
        )

    # Fig 10: spatial stats
    if morans_df is not None:
        fe.plot_spatial_stats(
            morans_df           = morans_df,
            neighborhood_result = neighborhood,
            n_top_genes         = min(30, len(morans_df)),
            output_dir          = OUTPUT_DIR,
            fmt                 = CFG.figure_format,
            dpi                 = CFG.dpi,
        )

    # Fig 11: cluster DGE
    if cluster_dge is not None and not cluster_dge.empty:
        fe.plot_cluster_dge(
            cluster_dge       = cluster_dge,
            adata             = adata,
            condition_key     = "condition",
            n_top_per_cluster = 5,
            output_dir        = OUTPUT_DIR,
            fmt               = CFG.figure_format,
            dpi               = CFG.dpi,
        )

    # Fig 12: Slide QC overview (per-slide cell counts + MBH yield)
    _plot_slide_qc(adata, OUTPUT_DIR, CFG.figure_format, CFG.dpi)

    # Fig 13: Panel composition + custom gene overlap QC
    # loader is None when the raw AnnData was loaded from cache; in that
    # case we skip Fig 13 rather than crash (delete the cache to regenerate).
    if loader is not None:
        try:
            fp.plot_panel_overview(
                adatas_raw          = loader.get_per_slide(),
                slide_ids           = manifest.slide_ids,
                conditions          = manifest.conditions,
                registry            = registry,
                harmonised          = loader.get_harmonised(),
                min_slides_threshold= min_slides,
                output_dir          = OUTPUT_DIR,
                fmt                 = CFG.figure_format,
                dpi                 = CFG.dpi,
            )
        except Exception as e:
            logger.warning("Panel overview figure failed: %s", e)
    else:
        logger.info(
            "Fig 13 skipped (raw AnnData loaded from cache). "
            "Delete %s to regenerate with panel QC.", h5_cache
        )

    # ------------------------------------------------------------------
    # Fig 14: Insulin & metabolic signalling panel
    # ------------------------------------------------------------------
    if cluster_dge is not None:
        try:
            fe.plot_insulin_panel(
                adata         = adata,
                dge_results   = dge,
                cluster_dge   = cluster_dge,
                condition_key = "condition",
                cluster_key   = group_key,
                output_dir    = OUTPUT_DIR,
                fmt           = CFG.figure_format,
                dpi           = CFG.dpi,
            )
        except Exception as e:
            logger.warning("Fig 14 (insulin panel) failed: %s", e)
    else:
        logger.info("Fig 14 skipped — cluster DGE not available.")

    # ------------------------------------------------------------------
    # Fig 15: Galanin (Gal) expression in the ageing MBH
    # ------------------------------------------------------------------
    # representative_slides is set on CFG in Step 4; use it directly here
    representative_slides = CFG.representative_slides
    try:
        fe.plot_galanin_panel(
            adata                = adata,
            dge_results          = dge,
            condition_key        = "condition",
            cluster_key          = group_key,
            cell_type_key        = "cell_type",
            representative_slides= representative_slides,
            output_dir           = OUTPUT_DIR,
            fmt                  = CFG.figure_format,
            dpi                  = CFG.dpi,
        )
    except Exception as e:
        logger.warning("Fig 15 (galanin panel) failed: %s", e)

    # ------------------------------------------------------------------
    # Fig 16: Cell type composition testing (scCODA)
    # ------------------------------------------------------------------
    if getattr(CFG, "run_sccoda", True) and "cell_type" in adata.obs.columns:
        logger.info("Step 9b: Cell type composition testing (scCODA) …")
        try:
            from src.composition_analysis import run_sccoda
            composition_results = run_sccoda(
                adata,
                cell_type_key         = "cell_type",
                condition_key         = "condition",
                replicate_key         = "slide_id",
                reference_cell_type   = getattr(CFG, "sccoda_reference_cell_type", "auto"),
                n_mcmc_samples        = getattr(CFG, "sccoda_n_mcmc_samples", 20_000),
                output_dir            = OUTPUT_DIR,
            )
            logger.info(
                "Composition testing complete: %d cell types, %d significant",
                len(composition_results),
                composition_results["significant"].sum() if not composition_results.empty else 0,
            )

            fe.plot_composition_panel(
                composition_results = composition_results,
                adata               = adata,
                cell_type_key       = "cell_type",
                condition_key       = "condition",
                replicate_key       = "slide_id",
                output_dir          = OUTPUT_DIR,
                fmt                 = CFG.figure_format,
                dpi                 = CFG.dpi,
            )
        except Exception as e:
            logger.warning("Fig 16 (composition panel) failed: %s", e)
            composition_results = None
    else:
        logger.info("Fig 16 skipped (run_sccoda=False or no cell_type annotation).")
        composition_results = None

    # ------------------------------------------------------------------
    # Save final AnnData
    # ------------------------------------------------------------------
    adata.write_h5ad(OUTPUT_DIR / "adata_mbh_final.h5ad")

    elapsed = time.time() - t0
    logger.info("=" * 65)
    logger.info("Pipeline complete in %.0f s", elapsed)
    logger.info("Outputs in: %s/", OUTPUT_DIR)
    logger.info("=" * 65)
    _print_figure_index(OUTPUT_DIR)


# ===========================================================================
# Figure 12: Slide-level QC overview (new, specific to multi-slide study)
# ===========================================================================

def _plot_slide_qc(adata, output_dir, fmt, dpi):
    """Per-slide cell counts, median counts, and MBH yield overview."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from src.figures import apply_nature_style, DOUBLE, WONG, _savefig

    apply_nature_style()

    slides = sorted(adata.obs["slide_id"].astype("category").cat.categories.tolist())
    cond_of_slide = adata.obs.groupby("slide_id", observed=True)["condition"].first()
    cond_pal = {"AGED": "#D55E00", "ADULT": "#0072B2"}

    fig = plt.figure(figsize=(DOUBLE, 2.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    x_pos = np.arange(len(slides))
    colours = [cond_pal.get(cond_of_slide[s], "#999999") for s in slides]

    # Cell counts per slide
    n_cells = [( adata.obs["slide_id"] == s).sum() for s in slides]
    ax1.bar(x_pos, n_cells, color=colours, width=0.6, linewidth=0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(slides, rotation=45, ha="right", fontsize=6)
    ax1.set_ylabel("Cells in MBH ROI")
    ax1.set_title("Cells per slide")

    # Median counts
    count_col = "total_counts" if "total_counts" in adata.obs.columns else "n_counts"
    med_counts = [
        adata.obs.loc[adata.obs["slide_id"] == s, count_col].median()
        if count_col in adata.obs.columns else 0
        for s in slides
    ]
    ax2.bar(x_pos, med_counts, color=colours, width=0.6, linewidth=0)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(slides, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Median total counts")
    ax2.set_title("Transcript yield")

    # Condition balance
    from collections import Counter
    cond_counts = Counter(adata.obs["condition"])
    conds = sorted(cond_counts.keys())
    ax3.bar(
        range(len(conds)),
        [cond_counts[c] for c in conds],
        color=[cond_pal.get(c, "#999") for c in conds],
        width=0.5, linewidth=0,
    )
    ax3.set_xticks(range(len(conds)))
    ax3.set_xticklabels(conds)
    ax3.set_ylabel("Total cells")
    ax3.set_title("Condition balance")

    # Shared legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=v, label=k) for k, v in cond_pal.items()]
    fig.legend(handles=handles, frameon=False, fontsize=6, loc="upper right",
               bbox_to_anchor=(1.0, 1.0))

    fig.suptitle("MBH ROI — per-slide quality overview", fontsize=8, y=1.02)
    fig.tight_layout(pad=0.4)
    _savefig(fig, output_dir / "fig12_slide_qc", fmt=fmt, dpi=dpi)
    plt.close(fig)


# ===========================================================================
# Helpers
# ===========================================================================

def _log_cell_counts(adata):
    if "slide_id" in adata.obs:
        by_slide = adata.obs.groupby(["condition", "slide_id"], observed=True).size().reset_index(name="n_cells")
        logger.info("Cells per slide:\n%s", by_slide.to_string(index=False))


def _print_figure_index(out: Path):
    print("\n Figure index")
    print(" " + "=" * 55)
    table = [
        ("fig1_qc",              "QC violins + spatial count density"),
        ("fig2_umap",             "UMAP by condition and cluster"),
        ("fig3_spatial_clusters", "Spatial cluster maps (MBH)"),
        ("fig4_dotplot",          "Marker gene dot plot"),
        ("fig5_volcano",          "AGED vs ADULT global volcano"),
        ("fig6_heatmap",          "Top DEG heatmap (z-scored)"),
        ("fig7_spatial_expr",     "Spatial expression of top DEGs (MBH)"),
        ("fig8_summary",          "6-panel composite summary"),
        ("fig9_cell_types",       "Cell type annotation in MBH"),
        ("fig10_spatial_stats",   "Moran's I + neighbourhood enrichment"),
        ("fig11_cluster_dge",     "Per-cluster DEG counts + bubble chart"),
        ("fig12_slide_qc",        "Per-slide QC + MBH yield overview"),
        ("fig13_panel_qc",        "Panel composition + custom gene overlap QC"),
        ("fig14_insulin_signalling", "Insulin & metabolic signalling panel"),
        ("fig15_galanin",           "Galanin (Gal) expression across cell types"),
        ("fig16_composition",       "Cell type composition testing (scCODA / CLR+t-test)"),
    ]
    for name, desc in table:
        f = out / f"{name}.pdf"
        status = "OK" if f.exists() else "missing"
        print(f"  {name:<32}  {desc}  [{status}]")
    print()


# ===========================================================================
# CLI
# ===========================================================================

def _apply_launcher_config(config_path: str):
    """
    Read a JSON config written by launcher.py and override the module-level
    study variables (ALL_SLIDES, ROOT_DATA, OUTPUT_DIR, etc.) and CFG.
    Called before main() when --launcher-config is provided.
    """
    import json as _json
    global ALL_SLIDES, AGED_SLIDES, ADULT_SLIDES
    global ROOT_DATA, OUTPUT_DIR, CACHE_DIR, ROI_CACHE, BASE_PANEL, CFG

    lcfg = _json.loads(Path(config_path).read_text())

    # Rebuild slide lists from GUI selections (only include slides with paths)
    raw_slides = [s for s in lcfg.get("slides", []) if s.get("run_dir")]
    AGED_SLIDES  = [s for s in raw_slides if s["condition"] == "AGED"]
    ADULT_SLIDES = [s for s in raw_slides if s["condition"] == "ADULT"]
    ALL_SLIDES   = AGED_SLIDES + ADULT_SLIDES

    # Paths
    OUTPUT_DIR = Path(lcfg["output_dir"])
    CACHE_DIR  = Path(str(OUTPUT_DIR) + "_cache")
    ROI_CACHE  = Path(lcfg.get("roi_cache_dir", "roi_cache"))
    BASE_PANEL = Path(lcfg.get("base_panel_csv",
                               "data/Xenium_mBrain_v1_1_metadata.csv"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Pipeline parameters
    CFG.output_dir           = OUTPUT_DIR
    CFG.cache_dir            = CACHE_DIR
    CFG.dge_method           = lcfg.get("dge_method",            CFG.dge_method)
    CFG.leiden_resolution    = float(lcfg.get("leiden_resolution",   CFG.leiden_resolution))
    CFG.n_neighbors          = int(lcfg.get("n_neighbors",           CFG.n_neighbors))
    CFG.min_counts           = int(lcfg.get("min_counts",            CFG.min_counts))
    CFG.max_counts           = int(lcfg.get("max_counts",            CFG.max_counts))
    CFG.min_genes            = int(lcfg.get("min_genes",             CFG.min_genes))
    CFG.max_genes            = int(lcfg.get("max_genes",             CFG.max_genes))
    CFG.n_top_genes          = int(lcfg.get("n_top_genes",           CFG.n_top_genes))
    CFG.harmony_max_iter     = int(lcfg.get("harmony_max_iter",      CFG.harmony_max_iter))
    CFG.dge_log2fc_threshold    = float(lcfg.get("log2fc_threshold",       CFG.dge_log2fc_threshold))
    CFG.filter_control_probes   = bool(lcfg.get("filter_control_probes",   CFG.filter_control_probes))
    CFG.filter_control_codewords= bool(lcfg.get("filter_control_codewords",CFG.filter_control_codewords))
    CFG.normalize_by_cell_area  = bool(lcfg.get("normalize_by_cell_area",  CFG.normalize_by_cell_area))
    CFG.dge_pval_threshold   = float(lcfg.get("pval_threshold",      CFG.dge_pval_threshold))
    CFG.figure_format        = lcfg.get("figure_format",             CFG.figure_format)
    CFG.dpi                  = int(lcfg.get("dpi",                   CFG.dpi))

    # Panel options from launcher
    panel_mode = lcfg.get("panel_mode", "partial_union")
    min_slides = int(lcfg.get("min_slides", 2))

    logger.info("Launcher config applied:")
    logger.info("  Slides    : %d (%d AGED, %d ADULT)",
                len(ALL_SLIDES), len(AGED_SLIDES), len(ADULT_SLIDES))
    logger.info("  Output    : %s", OUTPUT_DIR)
    logger.info("  DGE method: %s", CFG.dge_method)
    logger.info("  Panel mode: %s", panel_mode)

    no_roi_gui = bool(lcfg.get("no_roi_gui", True))  # always skip GUI from web launcher

    return panel_mode, min_slides, no_roi_gui


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AGED vs ADULT MBH Xenium DGE pipeline"
    )
    parser.add_argument(
        "--redraw-roi", action="store_true",
        help="Force re-drawing of ROIs even if they already exist",
    )
    parser.add_argument(
        "--no-roi-gui", action="store_true",
        help="Skip the interactive ROI GUI; use preset/saved ROIs only",
    )
    parser.add_argument(
        "--launcher-config", default=None, metavar="PATH",
        help="JSON config written by launcher.py (overrides hard-coded paths)",
    )
    args = parser.parse_args()

    if args.launcher_config:
        _launcher_panel_mode, _launcher_min_slides, _launcher_no_roi_gui = _apply_launcher_config(args.launcher_config)
    else:
        _launcher_panel_mode, _launcher_min_slides, _launcher_no_roi_gui = "partial_union", 2, False

    main(
        redraw_roi   = args.redraw_roi,
        no_roi_gui   = args.no_roi_gui or _launcher_no_roi_gui,
        panel_mode   = _launcher_panel_mode,
        min_slides   = _launcher_min_slides,
    )
