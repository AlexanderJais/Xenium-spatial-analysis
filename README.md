# Xenium DGE Pipeline
### Spatial transcriptomics · AGED vs ADULT mouse brain · Mediobasal hypothalamus

A Python pipeline for differential gene expression analysis of 10x Genomics Xenium
spatial transcriptomics data. Designed for a 4 AGED + 4 ADULT brain section study
using the Xenium_mBrain_v1_1 base panel (~247 genes) plus per-slide custom panels
(~50 genes each, partially overlapping), with interactive ROI selection for the
mediobasal hypothalamus (MBH).

Runs entirely on your Mac. No data leaves your machine.

---

## Two ways to run

### Option A — Local web interface (recommended)
A multi-page web app that opens in your browser. Point-and-click for everything:
folder selection, ROI drawing on the spatial scatter, live pipeline log, inline
figure viewer, and downloadable results.

### Option B — Desktop GUI launcher
A Tkinter window (`launcher.py`) for folder selection and settings, with the same
live log. Useful if you prefer a native app feel.

### Option C — Command line
Run `python run_xenium_mbh.py` directly after editing the paths inside
the file. Full control, no GUI.

---

## Installation (one-time, ~8 minutes)

Open Terminal, navigate to the project folder, then run:

    chmod +x install_mac.sh
    ./install_mac.sh

This installs Miniforge3 (ARM64 conda) if needed, creates a `xenium_dge` conda
environment with Python 3.11, and installs all dependencies as native Apple Silicon
binaries. Streamlit and Plotly (for the web interface) are included automatically.

---

## Starting the web interface

After installation, double-click `start_app.command` in Finder.
Or from Terminal:

    conda activate xenium_dge
    cd app
    streamlit run app.py

Your browser opens automatically at http://localhost:8501.

---

## Web interface pages

### 📁 Study Setup
Enter the path to each of the 8 Xenium output directories (4 AGED, 4 ADULT).
A green tick confirms the directory is valid (checks for cell_feature_matrix/,
cells.parquet, and all required MTX files). Shows cell count and gene count
per slide once validated. Save and reload the full configuration as a JSON file
so you never need to re-enter paths.

### ⚙️ Pipeline Settings
All analysis parameters in one place:

  Panel mode          partial_union (recommended), intersection, or union
  min_slides          Custom gene retention threshold (default 2 of 8 slides)
  DGE method          Wilcoxon (fast) or PyDESeq2 pseudobulk (publication)
  QC thresholds       min/max counts and genes per cell
  Leiden resolution   Cluster granularity
  Figure format       PDF (editable), PNG, or SVG
  DPI                 150 / 300 / 600

### 🗺️ ROI Manager
Interactive Plotly scatter of each tissue section. Select a slide, then use the
polygon tool in the chart toolbar to draw the MBH boundary:

  Left-click          place vertex
  Double-click        close polygon
  Toolbar eraser      remove last shape

A dashed orange ellipse provides an anatomical MBH atlas hint. ROIs are saved
as JSON files in roi_cache/ and reused automatically on every subsequent run.
You can also copy one ROI to multiple slides, or type coordinates manually if
the drawing tool does not suit your workflow.

### 🚀 Run Pipeline
Pre-flight checks confirm all slides are valid before launch. The Run Pipeline
button starts the analysis as a background subprocess. Output streams live into
a colour-coded log panel (green = ok, amber = warning, red = error). A Stop button
is always available. The page auto-refreshes every 1.5 seconds while running.

### 📊 Results
All 13 figures displayed inline with a dropdown selector and thumbnail gallery.
Individual download buttons for each figure. Separate tabs for:

  Global DGE table        searchable by gene name, colour-coded by log2FC
  Cluster DGE table       filterable by group and significance
  Moran's I results       spatially variable genes
  Panel validation        per-slide gene composition summary
  AnnData download        final adata_mbh_final.h5ad

### ℹ️ Help
Full documentation inline — panel structure explained, ROI drawing instructions,
parameter reference table, figure descriptions, troubleshooting guide.

---

## Panel structure

Every Xenium run produces one count matrix containing all genes for that slide:

  ~247 base genes    Xenium_mBrain_v1_1 — identical across ALL 8 slides
  +~50 custom genes  differs between slides, partial overlap between runs
  =~297 total genes  stored together in matrix.mtx.gz

The loader reads the full matrix without filtering. PanelRegistry classifies
each gene by comparing names against the base panel CSV.

Harmonisation modes:

  intersection      247 base genes only, no custom genes, zero zero-inflation
  partial_union *   base + custom genes present in >= min_slides slides (default 2)
  union             base + all custom genes, maximum zero-inflation

In partial_union mode, slides missing a retained custom gene receive a zero-filled
column flagged in adata.var['zero_filled'].

---

## Outputs

All figures and data files are written to the output directory you set in Study Setup.

  fig1_qc.pdf               QC violins: counts, genes per cell, spatial density
  fig2_umap.pdf             UMAP by condition and Leiden cluster
  fig3_spatial_clusters.pdf Spatial cluster maps in the MBH ROI
  fig4_dotplot.pdf          Marker gene dot plot
  fig5_volcano.pdf          AGED vs ADULT global volcano
  fig6_heatmap.pdf          Top DEG heatmap, z-scored
  fig7_spatial_expr.pdf     Spatial expression of top up-regulated DEGs
  fig8_summary.pdf          6-panel composite summary
  fig9_cell_types.pdf       Cell type annotation: UMAP + spatial + proportions
  fig10_spatial_stats.pdf   Moran's I + neighbourhood enrichment
  fig11_cluster_dge.pdf     Per-cluster DEG bubble chart
  fig12_slide_qc.pdf        Per-slide cell counts, transcript yield, condition balance
  fig13_panel_qc.pdf        Panel composition + custom gene overlap heatmap
  fig14_insulin.pdf         Insulin / metabolic signalling gene panel across cell types
  fig15_galanin.pdf         Galanin spatial maps, split violin per cell type, log2FC lollipop
  fig16_composition.pdf     Cell type composition testing (scCODA + CLR t-test fallback)

  global_dge_aged_vs_adult.csv    Full DGE results table
  cluster_dge_results.csv         Per-cluster DGE
  cluster_dge_summary.csv         DEG counts per cluster
  morans_i_mbh.csv                Spatially variable genes (Moran's I)
  panel_validation.csv            Per-slide gene panel composition and validation
  adata_mbh_final.h5ad            Final annotated AnnData

All figures follow Nature Publishing Group standards: column widths 89/183 mm,
Arial 6 pt minimum body text (7 pt labels, 8 pt titles), 300 DPI, editable PDF
with Type 42 embedded fonts. Colour-blind-safe Wong (2011) palette throughout.

---

## Project structure

    xenium_dge/
    │
    ├── start_app.command            Double-click to launch the web interface
    ├── install_mac.sh               One-command macOS installer (Apple Silicon)
    ├── launcher.py                  Alternative: desktop GUI launcher (Tkinter)
    │
    ├── app/                         Web interface (Streamlit)
    │   ├── app.py                   Main entry point
    │   ├── .streamlit/config.toml   Theme and server settings
    │   └── pages/
    │       ├── 1_study_setup.py     Slide folder configuration
    │       ├── 2_settings.py        All pipeline parameters
    │       ├── 3_roi_manager.py     Interactive ROI drawing (Plotly)
    │       ├── 4_run.py             Launch pipeline + live log
    │       ├── 5_results.py         Figure viewer + downloads
    │       └── 6_help.py            Inline documentation
    │
    ├── run_xenium_mbh.py            End-to-end pipeline for the 4+4 AGED/ADULT MBH study
    │                                (single production entry point — used by app, launcher, CLI)
    ├── launcher.py                  Tkinter GUI launcher
    ├── plot_gene.py                 CLI spatial expression map for any gene
    │
    │
    ├── xenium_analysis.ipynb        Interactive Jupyter notebook
    ├── requirements.txt             All Python dependencies
    │
    ├── data/
    │   └── Xenium_mBrain_v1_1_metadata.csv   Base panel gene list + annotations
    │
    └── src/                         Core analysis library
        ├── xenium_loader.py         Load Xenium run directories into AnnData
        ├── panel_registry.py        Gene classification and panel harmonisation
        ├── multislide_loader.py     Multi-slide loader: manifest, validation, concat
        ├── roi_selector.py          Matplotlib ROI tool (CLI mode)
        ├── config.py                PipelineConfig dataclass
        ├── preprocessing.py         QC, HVG, PCA, Harmony, UMAP, Leiden
        ├── dge_analysis.py          PyDESeq2 pseudobulk + Wilcoxon + C-SIDE DGE
        ├── cell_type_annotation.py  Marker scoring, correlation, threshold annotation
        ├── cluster_dge.py           Per-cluster / per-cell-type DGE
        ├── composition_analysis.py  Cell type composition testing (scCODA + CLR fallback)
        ├── spatial_stats.py         Moran's I, co-expression, neighbourhood enrichment
        ├── pipeline.py              Two-condition pipeline orchestrator
        ├── figures.py               Nature-grade figures 1-8
        ├── figures_extended.py      Nature-grade figures 9-11, 14-16
        └── figures_panel.py         Nature-grade figures 12-13 (slide/panel QC)

---

## DGE methods

  stringent_wilcoxon  Cell-level Wilcoxon + post-hoc filters: |log2FC| >= 1,
                      adj-p < 0.01, >= 10% cells expressing, consistent direction
                      in >= 3/4 replicates. Recommended for n=4 per condition.

  wilcoxon            Plain cell-level Wilcoxon rank-sum. Fast; inflated p-values
                      with large cell counts. Use for exploration only.

  pydeseq2            Pseudobulk DESeq2: cells aggregated per slide replicate.
                      Statistically correct but low power at n=4 — typically
                      returns no significant genes. Use when n >= 8 per condition.

  cside               C-SIDE pseudobulk (Cable et al. 2022, Nat Methods): per-cell-
                      type aggregation by slide replicate, then DESeq2. The only
                      published method designed for multi-replicate spatial DGE.
                      Recommended for publication when cell type labels are available.

Switch in Settings page or set CFG.dge_method in run_xenium_mbh.py.

---

## Memory usage on M4 Pro (48 GB)

  Load 8 slides (~3000 cells x ~297 genes)    ~2 GB
  After MBH ROI filter (~15% of tissue)       ~400 MB
  PCA + Harmony                               ~1 GB peak
  UMAP + figures                              ~1 GB
  Total peak                                  ~3-4 GB

---

## Requirements

See requirements.txt. Key packages:

  streamlit >= 1.35          Web interface
  plotly >= 5.20             Interactive ROI scatter
  scanpy >= 1.10             Single-cell analysis
  anndata >= 0.10
  harmonypy >= 0.0.9         Batch correction
  pydeseq2 >= 0.4            Pseudobulk DGE (optional, wilcoxon is fallback)
  leidenalg >= 0.10          Clustering
  pyarrow >= 14.0            Parquet support

---

## Troubleshooting

"No module named scanpy"
  Always activate the environment first:
  conda activate xenium_dge && cd app && streamlit run app.py

Plotly polygon tool not working in browser
  Use the manual coordinate entry panel in the ROI Manager instead.
  Paste x,y pairs (one per line, in micrometres) and click Save.

cell_feature_matrix/ not found
  The selected path must be the Xenium run output directory itself, not a
  parent folder. It must directly contain cell_feature_matrix/ and cells.parquet.

Custom genes not appearing after harmonisation
  Check panel_validation.csv in the output directory.
  Lower min_slides in Settings, or switch panel_mode to union.

Matplotlib window does not appear (CLI mode)
  echo "backend: MacOSX" >> ~/.matplotlib/matplotlibrc

leidenalg ImportError
  conda install -n xenium_dge -c conda-forge leidenalg -y

---

## Citation

  scanpy:    Wolf et al., Genome Biology 2018
  Harmony:   Korsunsky et al., Nature Methods 2019
  PyDESeq2:  Muzellec et al., Bioinformatics 2023
  UMAP:      McInnes et al., JOSS 2018
  Leiden:    Traag et al., Scientific Reports 2019
  scCODA:    Büttner et al., Nature Communications 2021
  C-SIDE:    Cable et al., Nature Methods 2022
  palette:   Wong, Nature Methods 2011
  Xenium:    10x Genomics Xenium In Situ platform
