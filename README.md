# Xenium Sample PCA

**Sample-level (pseudobulk) PCA for AGED vs ADULT mouse brain (mediobasal hypothalamus)**

A streamlined tool for the first exploratory step of a [10x Genomics Xenium](https://www.10xgenomics.com/platforms/xenium) spatial study: load the slides, frame the mediobasal hypothalamus (MBH) region on each, collapse every slide into a pseudobulk profile, and run PCA across the samples to see **how the samples cluster and how the AGED and ADULT groups separate** — before committing to any cell-level clustering or differential expression.

Designed for a multi-replicate, two-condition study (4 AGED + 4 ADULT brain sections) using the `Xenium_mBrain_v1_1` base panel (~247 genes) plus per-slide custom panels (~50 genes each, partially overlapping).

Runs entirely on your machine. No data leaves your computer.

---

## Table of contents

- [Two ways to run](#two-ways-to-run)
- [Installation](#installation)
- [Web interface](#web-interface)
- [Command line](#command-line)
- [How the PCA works](#how-the-pca-works)
- [Panel structure](#panel-structure)
- [Outputs](#outputs)
- [Configuration file format](#configuration-file-format)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Two ways to run

| Method | Command | Best for |
|--------|---------|----------|
| **A. Web interface** (recommended) | `streamlit run app/app.py` | Interactive ROI framing, inline Nature-style figures |
| **B. Command line** | `python run_sample_pca.py` | Scripted/headless runs once ROIs are saved |

Both paths share the same loader, ROI cache (`roi_cache/`), and PCA module (`src/sample_pca.py`). See the [Quick Start guide](QUICKSTART_MAC.md) for step-by-step instructions.

---

## Installation

```bash
cd /path/to/xenium-spatial-analysis
pip install -r requirements.txt
```

The dependency set is intentionally small (NumPy/pandas/SciPy/scikit-learn/Matplotlib + AnnData for data handling, Streamlit/Plotly for the UI, PyArrow for `cells.parquet`). No scanpy, Harmony, or DESeq2 is required.

> **macOS Apple Silicon:** `./install_mac.sh` creates a native ARM64 conda environment and installs everything for you.

---

## Web interface

```bash
streamlit run app/app.py
```

Or double-click `start_app.command` in Finder. Your browser opens at http://localhost:8501.

The app has exactly three steps:

| Step | Page | Purpose |
|------|------|---------|
| 1 | **📁 Study Setup** | Enter paths to the 8 Xenium output directories. A green tick confirms each is valid and shows its gene/cell counts. Save/load the full config as JSON. |
| 2 | **🗺️ ROI Manager** | Interactive Plotly scatter per slide. Use the four edge sliders to frame the MBH bounding rectangle; the cell count updates live. A dashed orange ellipse marks the atlas hint. Manual coordinate entry is available as a fallback. ROIs are saved to `roi_cache/` and reused automatically. |
| 3 | **📊 Sample PCA** | Loads the slides, applies the saved ROIs, pseudobulks each sample, and runs PCA across them. Shows the Nature-style PCA scatter (coloured by group), a hierarchically-clustered sample correlation heatmap, and a scree plot — inline, with PDF/CSV downloads. |

---

## Command line

Once paths and ROIs are set (the runner reads the same `roi_cache/`):

```bash
python run_sample_pca.py                 # load all configured slides, base panel only, run PCA
python run_sample_pca.py --samples AGED_1 ADULT_1   # run on a subset (>=2 samples)
python run_sample_pca.py --all-genes     # include per-slide add-on genes, not just the base panel
python run_sample_pca.py --no-roi        # use whole sections (skip ROI filtering)
python run_sample_pca.py --n-top-genes 200 --scale-genes   # restrict to top-variable genes, z-scored
python run_sample_pca.py --fmt png       # PNG instead of PDF figures
```

Slide paths are configured at the top of `run_sample_pca.py` (the `SLIDES` list), mirroring the web app's Study Setup. By default the PCA is restricted to the shared base panel and uses every configured slide; `--samples` selects a subset (minimum 2) and `--all-genes` opts back into the add-on genes. The web app's Sample PCA page exposes the same controls (a sample multiselect and a "Base panel only" toggle).

---

## How the PCA works

The analysis lives in `src/sample_pca.py` and runs in four steps:

0. **Restrict to the base panel** (default) — drop per-slide add-on genes so every sample is compared on the shared `Xenium_mBrain_v1_1` panel (~247 genes). This matters because samples can carry different add-on panels; pass `--all-genes` (or untick "Base panel only") to keep them.
1. **Pseudobulk** (`pseudobulk_samples`) — sum raw counts across all cells of each slide, giving one expression profile per biological replicate (one point per sample).
2. **Normalise** (`normalize_pseudobulk`) — library-size normalise each sample to counts-per-million, then `log1p`. Without this, PCA would just rank samples by cell number / sequencing depth.
3. **PCA** (`run_sample_pca`) — PCA across samples via scikit-learn. Uses all (base panel) genes by default (recommended for targeted Xenium panels); optionally restricts to the top-variable genes and/or z-scores genes.
4. **Plot** — a PC1/PC2 scatter coloured by group with sample labels, a sample-by-sample correlation heatmap ordered by hierarchical clustering, and a scree plot. (With only two samples PCA yields a single component, so the scatter spreads the samples along PC1.)

Pseudobulk PCA is the standard QC / sanity-check for replicated studies (cf. DESeq2's `plotPCA`): each point is one biological replicate, so it is robust at n=4 per group, and it makes outlier slides immediately visible.

---

## Panel structure

Every Xenium run produces one count matrix containing all genes for that slide:

| Group | Count | Description |
|-------|-------|-------------|
| **Base panel** | ~247 | `Xenium_mBrain_v1_1` — identical across all slides |
| **Custom panel** | ~50 | Additional genes — differs between slides, partial overlap |
| **Total** | ~297 | Stored together in `matrix.mtx.gz` |

`PanelRegistry` classifies each gene by comparing names against the base panel CSV and harmonises the slides to a common gene set before concatenation.

### Harmonisation modes

| Mode | Custom genes kept | Recommended when |
|------|-------------------|------------------|
| `intersection` | None (base only) | You only need the 247 base panel genes |
| **`partial_union`** | Present in ≥ `min_slides` slides | **Default — best for this study** |
| `union` | All custom genes | Exploratory analysis only |

In `partial_union` mode, slides missing a retained custom gene receive a zero-filled column. Because a gene can be zero-filled in one slide yet measured in another, the concatenated AnnData records this per slide in `adata.varm['zero_filled_by_slide']` (genes × slides), with study-level summaries in `adata.var['zero_filled_any']` and `adata.var['n_slides_zero_filled']`.

---

## Outputs

All files are written to `<output_dir>/sample_pca/` (web app) or `figures_output_sample_pca/` (CLI):

| File | Description |
|------|-------------|
| `sample_pca_scatter.pdf` | PC1 vs PC2, coloured by group, samples labelled (Nature-style) |
| `sample_correlation_heatmap.pdf` | Sample-by-sample correlation, hierarchically ordered |
| `sample_pca_scree.pdf` | Variance explained per PC + cumulative line |
| `sample_pca_coordinates.csv` | PC coordinates + condition / n_cells / total_counts per sample |
| `sample_pca_variance.csv` | Variance ratio and cumulative variance per PC |
| `pseudobulk_samples.h5ad` | Pseudobulk AnnData (counts, lognorm, `obsm['X_pca']`) |

Figures follow **Nature Publishing Group** conventions: Arial fonts, thin spines, editable PDF (Type 42 fonts), colour-blind-safe [Wong (2011)](https://doi.org/10.1038/nmeth.1618) group colours.

---

## Configuration file format

Study Setup can save/load a JSON configuration so you never re-enter paths:

```json
{
  "slides": [
    { "run_dir": "/path/to/AGED_1_output", "slide_id": "AGED_1", "condition": "AGED" }
  ],
  "output_dir": "/path/to/results",
  "base_panel_csv": "data/Xenium_mBrain_v1_1_metadata.csv",
  "roi_cache_dir": "roi_cache"
}
```

Only `slides` is required; the rest fall back to sensible defaults.

---

## Project structure

```
xenium-spatial-analysis/
├── start_app.command            Double-click to launch the web interface
├── install_mac.sh               macOS installer (Apple Silicon)
├── run_sample_pca.py            CLI entry point for the sample PCA
├── requirements.txt             Python dependencies
│
├── app/                         Web interface (Streamlit)
│   ├── app.py                   3-step landing page
│   ├── ui_utils.py              Shared CSS injection and page header
│   ├── styles.css               Custom Streamlit styles
│   ├── .streamlit/config.toml   Theme and server settings
│   └── pages/
│       ├── 1_study_setup.py     Slide folder configuration + JSON save/load
│       ├── 2_roi_manager.py     Interactive ROI framing (Plotly + atlas hint)
│       └── 3_sample_pca.py      Pseudobulk PCA + Nature-style figures
│
├── data/
│   └── Xenium_mBrain_v1_1_metadata.csv   Base panel gene list + annotations
│
└── src/                         Core analysis library
    ├── xenium_loader.py         Load a Xenium run directory into AnnData
    ├── multislide_loader.py     Multi-slide manifest, validation, concat
    ├── panel_registry.py        Gene classification and panel harmonisation
    ├── roi_selector.py          ROI persistence + apply (reads roi_cache/)
    └── sample_pca.py            Pseudobulk, normalise, PCA, and figures
```

---

## Requirements

See [`requirements.txt`](requirements.txt). Key packages:

| Package | Min version | Purpose |
|---------|-------------|---------|
| streamlit | 1.35 | Web interface |
| plotly | 5.20 | Interactive ROI scatter |
| numpy / pandas / scipy | — | Core numerics |
| scikit-learn | 1.3 | PCA |
| matplotlib | 3.8 | Nature-style figures |
| anndata | 0.10 | Annotated data matrices |
| pyarrow | 14.0 | Parquet support (`cells.parquet`) |

---

## Troubleshooting

**ROI sliders not responding**
Refresh the page (Cmd+R / Ctrl+R). If it persists, use the manual coordinate entry panel to type x,y pairs directly.

**`cell_feature_matrix/` not found**
The selected path must be the Xenium run output directory itself, not a parent folder. It must directly contain `cell_feature_matrix/` (with `matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`) and `cells.parquet`.

**PCA separates samples by cell number, not biology**
This usually means library-size normalisation was bypassed. The built-in workflow always CPM-normalises before PCA; if you are calling the functions directly, run `normalize_pseudobulk` before `run_sample_pca`.

**Custom genes not appearing after harmonisation**
Lower `min_slides`, or switch `panel_mode` to `union`.

**ROI selects 0 cells**
The MBH sits in the ventral 50–80% of a coronal section (larger y, since y increases toward ventral). Re-frame using the dashed orange atlas-hint ellipse as a guide.

---

## Citation

If you use this tool, please cite the underlying methods:

| Tool | Reference |
|------|-----------|
| scikit-learn (PCA) | Pedregosa et al., *JMLR* 2011 |
| AnnData | Virshup et al., *JOSS* 2024 |
| Colour palette | Wong, *Nature Methods* 2011 |
| Xenium | 10x Genomics Xenium In Situ platform |
