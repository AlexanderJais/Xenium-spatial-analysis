# Xenium DGE Pipeline

**Spatial transcriptomics differential gene expression for AGED vs ADULT mouse brain (mediobasal hypothalamus)**

A Python pipeline for end-to-end analysis of [10x Genomics Xenium](https://www.10xgenomics.com/platforms/xenium) spatial transcriptomics data. Designed for a multi-replicate, two-condition study (4 AGED + 4 ADULT brain sections) using the `Xenium_mBrain_v1_1` base panel (~247 genes) plus per-slide custom panels (~50 genes each, partially overlapping), with interactive ROI selection for the mediobasal hypothalamus (MBH).

Runs entirely on your Mac. No data leaves your machine.

---

## Table of contents

- [Three ways to run](#three-ways-to-run)
- [Installation](#installation)
- [Web interface](#web-interface)
- [Panel structure](#panel-structure)
- [DGE methods](#dge-methods)
- [Outputs](#outputs)
- [Configuration file format](#configuration-file-format)
- [Project structure](#project-structure)
- [Memory usage](#memory-usage)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Three ways to run

| Method | Command | Best for |
|--------|---------|----------|
| **A. Web interface** (recommended) | `streamlit run app/app.py` | Interactive ROI drawing, live log, inline results |
| **B. Desktop GUI** | `python launcher.py` | Native Tkinter window, folder browser dialogs |
| **C. Command line** | `python run_xenium_mbh.py` | Scripted/headless runs, HPC |

All three methods run the same underlying pipeline (`run_xenium_mbh.py`). See the [Quick Start guide](QUICKSTART_MAC.md) for step-by-step instructions.

---

## Installation

> **macOS Apple Silicon (M1/M2/M3/M4) recommended.** The installer creates a native ARM64 conda environment. See [QUICKSTART_MAC.md](QUICKSTART_MAC.md) for detailed instructions.

```bash
cd /path/to/xenium_dge
chmod +x install_mac.sh
./install_mac.sh
```

This installs Miniforge3 (ARM64 conda) if needed, creates a `xenium_dge` environment with Python 3.11, and installs all dependencies as native Apple Silicon binaries. Takes ~8 minutes.

**Manual installation** (any platform):

```bash
conda create -n xenium_dge python=3.11 -y
conda activate xenium_dge
pip install -r requirements.txt
conda install -c conda-forge leidenalg igraph -y
```

---

## Web interface

After installation, launch with:

```bash
conda activate xenium_dge
streamlit run app/app.py
```

Or double-click `start_app.command` in Finder. Your browser opens at http://localhost:8501.

### Pages

| Page | Purpose |
|------|---------|
| **📁 Study Setup** | Enter paths to 8 Xenium output directories. Green tick = valid. Save/load full config as JSON. |
| **⚙️ Pipeline Settings** | All analysis parameters: panel mode, DGE method, QC thresholds, Leiden resolution, figure format, DPI. |
| **🗺️ ROI Manager** | Interactive Plotly scatter for each slide. Use sliders to frame the MBH bounding rectangle; dashed orange ellipse provides an atlas hint. Manual coordinate entry as fallback. ROIs saved to `roi_cache/` and reused automatically. |
| **🚀 Run Pipeline** | Pre-flight validation, then one-click launch. Live colour-coded log with progress stages. Stop button always available. |
| **📊 Results** | All 17 figures inline with dropdown + thumbnail gallery. Download buttons for each figure. Tabs for DGE tables, Moran's I, panel validation, and the final AnnData `.h5ad`. |
| **🔬 Gene Explorer** | On-demand spatial expression map for any gene across all MBH slides. No pipeline rerun needed — reads from the preprocessed AnnData cache. |
| **ℹ️ Help** | Full inline documentation: setup guide, panel structure, ROI drawing instructions, parameter reference, figure descriptions, troubleshooting. |

---

## Panel structure

Every Xenium run produces one count matrix containing all genes for that slide:

| Group | Count | Description |
|-------|-------|-------------|
| **Base panel** | ~247 | `Xenium_mBrain_v1_1` — identical across all 8 slides |
| **Custom panel** | ~50 | Additional genes — differs between slides, partial overlap |
| **Total** | ~297 | Stored together in `matrix.mtx.gz` |

The loader reads the full matrix without filtering. `PanelRegistry` classifies each gene by comparing names against the base panel CSV.

### Harmonisation modes

| Mode | Custom genes kept | Zero-filling | Recommended when |
|------|-------------------|--------------|------------------|
| `intersection` | None (base only) | None | You only need the 247 base panel genes |
| **`partial_union`** | Present in >= `min_slides` slides | Minimal | **Default — best for this study** |
| `union` | All custom genes | Extensive | Exploratory analysis only |

In `partial_union` mode, slides missing a retained custom gene receive a zero-filled column flagged in `adata.var['zero_filled']`. After harmonisation, `adata.var['panel_type']` is one of: `base`, `custom_shared`, or `custom_unique`.

---

## DGE methods

| Method | Type | Description | Recommended when |
|--------|------|-------------|------------------|
| **`stringent_wilcoxon`** | Cell-level | Wilcoxon rank-sum + post-hoc filters: \|log2FC\| >= 1, adj-p < 0.01, >= 10% expressing, consistent direction in >= 3/4 replicates | **Default — best for n=4 per condition** |
| `wilcoxon` | Cell-level | Plain Wilcoxon rank-sum. Fast but inflated p-values with large cell counts | Quick exploration |
| `pydeseq2` | Pseudobulk | Cells aggregated per slide replicate, then DESeq2. Statistically correct but low power at small n | n >= 8 per condition |
| `cside` | Pseudobulk | C-SIDE (Cable et al. 2022, *Nat Methods*): per-cell-type aggregation by slide, then DESeq2. The only published method designed for multi-replicate spatial DGE | **Publication — when cell type labels are available** |

Switch via the **⚙️ Settings** page (web app), the desktop launcher, or `CFG.dge_method` in `run_xenium_mbh.py`.

---

## Outputs

All files are written to the output directory configured in Study Setup.

### Figures

All figures follow **Nature Publishing Group** standards: column widths 89/183 mm, Arial 6 pt minimum, 300 DPI, editable PDF with Type 42 embedded fonts. Colour-blind-safe [Wong (2011)](https://doi.org/10.1038/nmeth.1618) palette throughout.

| Figure | Content |
|--------|---------|
| fig1_qc | QC violins (counts, genes per cell) + lollipop of cells per slide |
| fig2_umap | UMAP coloured by condition and Leiden cluster |
| fig3_spatial_clusters | Spatial cluster maps in MBH ROI, one panel per condition |
| fig4_dotplot | Marker gene dot plot (size = % expressing, colour = mean expression) |
| fig5_volcano | AGED vs ADULT global volcano plot, top genes labelled |
| fig6_heatmap | Top DEG heatmap, z-scored, cells sorted by condition |
| fig7_spatial_expr | Spatial expression of top up-regulated DEGs |
| fig8_summary | 6-panel composite (UMAP + spatial + volcano + barplots) |
| fig9_cell_types | Cell type annotation: UMAP + spatial + proportion bar + confidence |
| fig10_spatial_stats | Moran's I lollipop + spatial co-expression + neighbourhood enrichment |
| fig11_cluster_dge | Per-cluster DEG counts + bubble chart of top DEGs per cluster |
| fig12_slide_qc | Per-slide cell counts, transcript yield, condition balance |
| fig13_panel_qc | Panel composition bars + custom gene overlap heatmap + UpSet histogram |
| fig14_insulin | Insulin/metabolic signalling gene panel across cell types |
| fig15_galanin | Galanin spatial maps, split violin per cell type, log2FC lollipop |
| fig16_composition | Cell type composition testing (scCODA Bayesian model + CLR t-test fallback) |
| fig17_neuropeptide_modules | Neuropeptide co-expression modules: UMAP, z-scored heatmap, AGED vs ADULT bar comparison, spatial maps. Modules: AgRP/NPY, POMC/CART, KNDy, Somatostatin, TRH/Dopamine, Galanin |

### Data files

| File | Description |
|------|-------------|
| `global_dge_aged_vs_adult.csv` | Full DGE results table (all genes) |
| `cluster_dge_results.csv` | Per-cluster DGE results |
| `cluster_dge_summary.csv` | DEG counts per cluster |
| `morans_i_mbh.csv` | Spatially variable genes (Moran's I) |
| `panel_validation.csv` | Per-slide gene panel composition and validation |
| `adata_mbh_final.h5ad` | Final annotated AnnData object |

---

## Configuration file format

The web app and launcher can save/load a JSON configuration file so you never need to re-enter paths. The format is:

```json
{
  "slides": [
    {
      "run_dir": "/path/to/AGED_1_output",
      "slide_id": "AGED_1",
      "condition": "AGED"
    }
  ],
  "output_dir": "/path/to/results",
  "base_panel_csv": "data/Xenium_mBrain_v1_1_metadata.csv",
  "dge_method": "stringent_wilcoxon",
  "panel_mode": "partial_union",
  "min_slides": 2,
  "leiden_resolution": 0.6,
  "n_neighbors": 12,
  "min_counts": 10,
  "max_counts": 2000,
  "min_genes": 10,
  "max_genes": 300,
  "n_top_genes": 0,
  "harmony_max_iter": 30,
  "log2fc_threshold": 1.0,
  "pval_threshold": 0.01,
  "filter_control_probes": true,
  "filter_control_codewords": true,
  "normalize_by_cell_area": false,
  "figure_format": "pdf",
  "dpi": 300
}
```

Only `slides` and `output_dir` are required. All other fields fall back to Xenium-tuned defaults if omitted. Notably: `max_counts=2000` and `max_genes=300` are lower than typical scRNA-seq settings because the ~307-gene Xenium panel yields far fewer transcripts per cell.

---

## Project structure

```
xenium_dge/
├── start_app.command            Double-click to launch the web interface
├── install_mac.sh               One-command macOS installer (Apple Silicon)
├── launcher.py                  Desktop GUI launcher (Tkinter)
│
├── app/                         Web interface (Streamlit)
│   ├── app.py                   Main entry point
│   ├── ui_utils.py              Shared CSS injection and page header
│   ├── styles.css               Custom Streamlit styles
│   ├── .streamlit/config.toml   Theme and server settings
│   └── pages/
│       ├── 1_study_setup.py     Slide folder configuration
│       ├── 2_settings.py        All pipeline parameters
│       ├── 3_roi_manager.py     Interactive ROI drawing (Plotly)
│       ├── 4_run.py             Launch pipeline + live log
│       ├── 5_results.py         Figure viewer + data downloads
│       ├── 6_gene_explorer.py   On-demand spatial expression maps
│       └── 7_help.py            Inline documentation
│
├── run_xenium_mbh.py            End-to-end pipeline script
│                                (single production entry point used by app, launcher, CLI)
├── plot_gene.py                 CLI spatial expression map for any gene
├── xenium_analysis.ipynb        Interactive Jupyter notebook
├── requirements.txt             Python dependencies
│
├── data/
│   └── Xenium_mBrain_v1_1_metadata.csv   Base panel gene list + annotations
│
└── src/                         Core analysis library
    ├── config.py                PipelineConfig dataclass
    ├── xenium_loader.py         Load Xenium run directories into AnnData
    ├── multislide_loader.py     Multi-slide loader: manifest, validation, concat
    ├── panel_registry.py        Gene classification and panel harmonisation
    ├── roi_selector.py          Matplotlib ROI tool (CLI mode)
    ├── preprocessing.py         QC filtering, HVG, PCA, Harmony, UMAP, Leiden
    ├── cell_type_annotation.py  Marker scoring, correlation, threshold annotation
    ├── dge_analysis.py          PyDESeq2 pseudobulk + Wilcoxon + stringent Wilcoxon + C-SIDE
    ├── cluster_dge.py           Per-cluster / per-cell-type DGE
    ├── composition_analysis.py  Cell type composition testing (scCODA + CLR fallback)
    ├── spatial_stats.py         Moran's I, co-expression, neighbourhood enrichment
    ├── pipeline.py              Two-condition pipeline orchestrator
    ├── figures.py               Nature-grade figures 1--8
    ├── figures_extended.py      Nature-grade figures 9--11, 14--17
    └── figures_panel.py         Nature-grade figures 12--13 (slide/panel QC)
```

---

## Memory usage

Measured on M4 Pro with 48 GB RAM:

| Step | Approx. RAM |
|------|-------------|
| Loading 8 slides (~3 000 cells x ~297 genes each) | ~2 GB |
| After MBH ROI filter (~15% of tissue) | ~400 MB |
| PCA + Harmony batch correction | ~1 GB peak |
| UMAP + Leiden clustering | ~500 MB |
| DGE (pseudobulk or cell-level) | ~200 MB |
| Figure generation (17 figures) | ~500 MB |
| **Total peak** | **~3--4 GB** |

For datasets with 50 000+ cells per slide, set `n_top_genes = 100` to reduce the HVG feature space and memory footprint.

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

| Package | Min version | Purpose |
|---------|-------------|---------|
| streamlit | 1.35 | Web interface |
| plotly | 5.20 | Interactive ROI scatter |
| scanpy | 1.10 | Single-cell analysis |
| anndata | 0.10 | Annotated data matrices |
| harmonypy | 0.0.9 | Batch correction |
| pydeseq2 | 0.4 | Pseudobulk DGE (optional; Wilcoxon is the fallback) |
| leidenalg | 0.10 | Graph-based clustering |
| pyarrow | 14.0 | Parquet support (`cells.parquet`) |
| statsmodels | 0.14 | Multiple-testing correction (BH FDR) |
| scikit-misc | 0.3 | Seurat v3 HVG selection |
| adjustText | 1.1 | Volcano plot label placement (optional) |

---

## Troubleshooting

**"No module named scanpy"**
Always activate the environment first:
```bash
conda activate xenium_dge
```

**ROI sliders not responding**
Refresh the page (Ctrl+R / Cmd+R). If the issue persists, use the manual coordinate entry panel to type x,y pairs directly.

**`cell_feature_matrix/` not found**
The selected path must be the Xenium run output directory itself, not a parent folder. It must directly contain `cell_feature_matrix/` and `cells.parquet`.

**Custom genes not appearing after harmonisation**
Check `panel_validation.csv` in the output directory. Lower `min_slides` in Settings, or switch `panel_mode` to `union`.

**PyDESeq2 finds no significant genes with n=4**
PyDESeq2 pseudobulk needs >= 8 biological replicates per condition to have power. Use `stringent_wilcoxon` (default, recommended for n=4) or `cside` for per-cell-type pseudobulk.

**Matplotlib window does not appear (CLI mode)**
```bash
echo "backend: MacOSX" >> ~/.matplotlib/matplotlibrc
```

**leidenalg ImportError**
```bash
conda install -n xenium_dge -c conda-forge leidenalg -y
```

**Pipeline log shows Harmony error**
```bash
pip install harmonypy
```

---

## Citation

If you use this pipeline, please cite the underlying methods:

| Tool | Reference |
|------|-----------|
| scanpy | Wolf et al., *Genome Biology* 2018 |
| Harmony | Korsunsky et al., *Nature Methods* 2019 |
| PyDESeq2 | Muzellec et al., *Bioinformatics* 2023 |
| UMAP | McInnes et al., *JOSS* 2018 |
| Leiden | Traag et al., *Scientific Reports* 2019 |
| scCODA | Buttner et al., *Nature Communications* 2021 |
| C-SIDE | Cable et al., *Nature Methods* 2022 |
| Colour palette | Wong, *Nature Methods* 2011 |
| Xenium | 10x Genomics Xenium In Situ platform |
