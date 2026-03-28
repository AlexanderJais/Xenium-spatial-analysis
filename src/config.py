"""
config.py
---------
Central configuration for the Xenium DGE pipeline.
Adjust paths, QC thresholds, clustering resolution, and
figure aesthetics here before running the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Nature figure standards
# ---------------------------------------------------------------------------
# Column widths in inches (1 pt = 1/72 inch)
NATURE_SINGLE_COL_IN  = 3.50   # 89 mm
NATURE_1P5_COL_IN     = 4.72   # 120 mm
NATURE_DOUBLE_COL_IN  = 7.20   # 183 mm

NATURE_FONTS = {
    "family"     : "Arial",
    "size"       : 7,           # body / tick labels
    "title_size" : 8,
    "label_size" : 7,
    "legend_size": 6,
    "panel_label": 9,           # A, B, C …
}

# Colour-blind-safe palette (Wong 2011)
WONG_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
]

# Diverging palette for DGE volcano / heatmap
DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = "Reds"
SPATIAL_CMAP = "magma"


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------
    # Input / output
    # ------------------------------------------------------------------
    condition_a_dir: Path = Path("data/condition_A")   # Xenium output dir
    condition_b_dir: Path = Path("data/condition_B")
    condition_a_label: str = "Control"
    condition_b_label: str = "Treatment"
    output_dir: Path = Path("figures_output")
    cache_dir: Path = Path(".cache")

    # ------------------------------------------------------------------
    # QC thresholds
    # ------------------------------------------------------------------
    min_counts: int = 10          # minimum transcript counts per cell
    max_counts: int = 5_000       # cap for doublet/artefact removal
    min_genes: int = 10           # minimum unique genes per cell (raised for Xenium 307-gene panel)
    max_genes: int = 500
    min_cells_per_gene: int = 10  # genes expressed in < N cells are dropped

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    target_sum: float = 100.0     # Xenium-specific: use 100, not 10 000 (Salas 2025 Nature Methods)
    n_top_genes: int = 0          # 0 = use all genes (recommended for targeted Xenium panels)

    # ── Xenium-specific QC (negative control probes) ─────────────────────
    # Cells where ANY negative control probe OR codeword was detected are
    # removed.  10x Genomics alert threshold: warning at 0.02, error at 0.05
    # (ratio of control counts to total).  Removing cells with any
    # positive control signal is the most conservative, widely-used approach.
    filter_control_probes: bool = True   # remove cells with control_probe_counts > 0
    filter_control_codewords: bool = True  # remove cells with control_codeword_counts > 0

    # ── Cell area QC ──────────────────────────────────────────────────────
    # Cells with area < min_cell_area µm² are very likely partial segmentation
    # captures at tissue edges. Default 20 µm² per Janesick et al. 2023.
    # Set to 0 to disable (applied only if 'cell_area' is in obs).
    min_cell_area: float = 20.0

    # ── Cell area normalization ───────────────────────────────────────────
    # Normalise expression by cell area after library-size normalisation.
    # Reduces spatial technical confound from variable cell sizes in brain tissue
    # (neurons >> glia in cross-sectional area).  Default off; set True if
    # DGE results show strong correlation with cell_area in PCA.
    normalize_by_cell_area: bool = False

    # ── Q-score note (informational only) ────────────────────────────────
    # Transcript-level Q20 filtering is applied internally by Xenium Onboard
    # Analysis (XOA) before building the cell-by-gene matrix.  If you are
    # working from raw transcript files (.parquet with qv column), apply
    # qv_threshold = 20 at load time.  This pipeline uses the XOA matrix
    # so Q20 is already enforced upstream.
    n_pcs: int = 50               # PCs for dimensionality reduction

    # ------------------------------------------------------------------
    # Integration (Harmony)
    # ------------------------------------------------------------------
    harmony_key: str = "slide_id"      # MUST be slide_id, not condition -- correcting on condition removes the biological signal you are testing
    harmony_max_iter: int = 20

    # ------------------------------------------------------------------
    # Neighbourhood graph & UMAP
    # ------------------------------------------------------------------
    n_neighbors: int = 12
    umap_min_dist: float = 0.3
    umap_spread: float = 1.0
    random_state: int = 42

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    leiden_resolution: float = 0.6
    cluster_key: str = "leiden"

    # ------------------------------------------------------------------
    # Differential gene expression
    # ------------------------------------------------------------------
    dge_method: str = "stringent_wilcoxon"  # "stringent_wilcoxon" | "wilcoxon" | "pydeseq2" | "cside" | "t-test"
    dge_group_key: str = "condition"
    dge_min_cells: int = 5
    dge_log2fc_threshold: float = 1.0    # aligned with stringent_wilcoxon
    dge_pval_threshold: float = 0.01     # aligned with stringent_wilcoxon
    n_top_dge_genes: int = 20      # genes to label on volcano

    # ------------------------------------------------------------------
    # Composition analysis (scCODA)
    # ------------------------------------------------------------------
    # scCODA (Büttner et al. 2021, Nat Commun 12:6876) tests whether cell type
    # proportions differ between conditions using a Bayesian Dirichlet-multinomial
    # model.  This is statistically valid at n=4 (no normal-distribution assumption)
    # and is the recommended primary analysis for testing compositional changes.
    # Requires pertpy; falls back to CLR + Welch t-test if not installed.
    run_sccoda: bool = True
    sccoda_reference_cell_type: str = "auto"  # "auto" = largest cell type; or e.g. "Astrocyte"
    sccoda_n_mcmc_samples: int = 20_000

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------
    spot_size: float = 8.0         # dot size for spatial scatter
    # Representative slide per condition for spatial figures (fig3, fig7).
    # e.g. {"ADULT": "ADULT_1", "AGED": "AGED_3"}. None = first slide.
    representative_slides: Optional[dict] = None
    alpha_spatial: float = 0.6

    # ------------------------------------------------------------------
    # Figure export
    # ------------------------------------------------------------------
    dpi: int = 300
    figure_format: str = "pdf"     # "pdf" | "svg" | "png"
    figure_transparent: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
