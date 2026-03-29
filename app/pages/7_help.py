"""
pages/7_help.py
Help & documentation page.
"""

import streamlit as st
from pathlib import Path

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(page_title="Help · Xenium DGE", page_icon="ℹ️", layout="wide",
    initial_sidebar_state="expanded")


inject_css()
page_header("ℹ️ Help & Documentation", "Setup guide, parameter reference, and troubleshooting")

tab_start, tab_panel, tab_roi, tab_params, tab_figs, tab_trouble = st.tabs([
    "Getting started", "Panel structure", "ROI drawing",
    "Parameters", "Figures", "Troubleshooting"
])

with tab_start:
    st.markdown("""
## Getting started

### 1 — Study Setup
Go to **📁 Study Setup** and enter the path to each Xenium output directory.
Each path must point directly to the folder that contains:
```
cell_feature_matrix/
    barcodes.tsv.gz
    features.tsv.gz
    matrix.mtx.gz
cells.parquet
experiment.xenium
```

**macOS tip:** Drag a folder from Finder into Terminal and copy the path that appears.
Or right-click → Get Info → copy the *Where* path.

### 2 — Settings
Check **⚙️ Pipeline Settings**. The defaults are sensible for an MBH study.
Key decisions:
- **Panel mode:** `partial_union` (recommended) keeps base genes + shared custom genes
- **DGE method:** `stringent_wilcoxon` (default, recommended); `wilcoxon` for speed; `pydeseq2` for pseudobulk; `cside` for per-cell-type pseudobulk DESeq2 (Cable 2022, recommended for publication)
- **min_slides:** How many slides a custom gene must appear in to be retained (default 2)

### 3 — ROI Manager
Go to **🗺️ ROI Manager** and use the sliders to frame the mediobasal hypothalamus boundary on each section.
The dashed orange ellipse is an anatomical hint. Saved ROIs are reused automatically.

### 4 — Run
Click **▶ Run Pipeline** on the **🚀 Run Pipeline** page.
The log streams live. Runtime on an M4 Pro: approximately 15–25 minutes for 8 slides.

### 5 — Results
Browse **📊 Results** to view all figures inline and download CSVs and the AnnData.
""")

with tab_panel:
    st.markdown("""
## Panel structure

Every Xenium run produces **one** count matrix containing all genes measured on that slide.
The genes come in two groups:

| Group | Count | Description |
|-------|-------|-------------|
| **Base panel** | ~247 | `Xenium_mBrain_v1_1` — **identical across ALL 8 slides** |
| **Custom panel** | ~50 | Additional genes — **differs between slides**, partial overlap |
| **Total** | ~297 | All stored together in `matrix.mtx.gz` |

The loader reads the full matrix without filtering. `PanelRegistry` then classifies
each gene by comparing its name against the base panel CSV.

### Harmonisation modes

| Mode | Custom genes kept | Zero-filling | Use when |
|------|---|---|---|
| `intersection` | None | None | You only care about base panel genes |
| `partial_union` ✓ | Present in ≥ min_slides slides | Minimal | **This study — default** |
| `union` | All custom genes | Many | Exploratory only |

### What `partial_union` does

1. Base genes (247): kept in every slide — no zero-filling needed
2. Custom gene shared by ≥ 2 slides: kept; slides missing it get a zero column (flagged in `adata.var['zero_filled']`)
3. Custom gene unique to 1 slide: dropped — pure zero columns only add noise to DGE

After harmonisation, `adata.var['panel_type']` contains:
- `base` — one of the 247 Xenium_mBrain_v1_1 genes
- `custom_shared` — a custom gene present in multiple slides (kept)
- `custom_unique` — a custom gene present in only one slide (dropped in partial_union)
""")

with tab_roi:
    st.markdown("""
## ROI drawing

### Slider-based selection
1. Open **🗺️ ROI Manager** and select a slide from the dropdown
2. Use the four sliders to frame the MBH region:
   - **Left edge / Right edge** — x-axis bounds (µm)
   - **Top edge / Bottom edge** — y-axis bounds (dorsal → ventral, µm)
3. The scatter plot and cell count update live as you move the sliders
4. Click **💾 Save ROI** when the rectangle covers the MBH boundary
5. All slider values are computed from the actual tissue bounds of each slide

### Atlas hint
The dashed orange ellipse marks the approximate location of the mediobasal hypothalamus
(ventromedial nucleus + arcuate nucleus region). It is based on relative tissue coordinates
and should be used as a starting guide only — adjust your sliders to the actual histology
of each section.

### Manual coordinate entry
If you need precise coordinates, use the **Enter coordinates manually** panel.
Paste x,y vertex pairs (in µm) one per line:
```
3200, 4100
3800, 4100
3800, 4700
3200, 4700
```

### Copying ROIs
If your sections are at similar coordinates you can define the ROI on one slide
and copy it to others using the **Copy this ROI** expander.

### ROI files
ROIs are saved as JSON files in `roi_cache/`:
```
roi_cache/
    AGED_1_roi.json
    AGED_2_roi.json
    ...
```
These are plain text and can be edited manually if needed.
""")

with tab_params:
    st.markdown("""
## Parameter reference

### QC
| Parameter | Default | Description |
|-----------|---------|-------------|
| min_counts | 10 | Cells with fewer total transcripts are removed |
| max_counts | 2000 | Cells above this are likely doublets (lower than scRNA-seq because the ~307-gene panel yields fewer transcripts) |
| min_genes | 10 | Cells with fewer unique genes are removed |
| max_genes | 300 | Upper gene count filter (panel has ~307 genes total) |

### Preprocessing
| Parameter | Default | Description |
|-----------|---------|-------------|
| n_top_genes | 0 | Highly variable genes for PCA (0 = disabled, use all genes — recommended for Xenium) |
| leiden_resolution | 0.6 | Higher = more, smaller clusters |
| n_neighbors | 12 | KNN graph neighbours for UMAP and clustering |
| harmony_max_iter | 30 | Harmony batch correction iterations (30 for robust convergence across 8 slides) |

**Important:** `harmony_key` is always `slide_id` (not condition).
This corrects technical variation between the 8 slides while preserving
the biological AGED vs ADULT signal.

### DGE
| Parameter | Default | Description |
|-----------|---------|-------------|
| dge_method | stringent_wilcoxon | `stringent_wilcoxon` (recommended), `wilcoxon`, `pydeseq2` (pseudobulk), or `cside` (per-cell-type pseudobulk DESeq2, Cable 2022) |
| log2fc_threshold | 1.0 | Minimum |log₂FC| for significance (stringent_wilcoxon uses this) |
| pval_threshold | 0.01 | Adjusted p-value threshold (stringent_wilcoxon uses this) |

### Panel
| Parameter | Default | Description |
|-----------|---------|-------------|
| panel_mode | partial_union | How to harmonise custom genes across slides |
| min_slides | 2 | Custom genes must appear in ≥ this many slides |

### Figures
| Parameter | Default | Description |
|-----------|---------|-------------|
| figure_format | pdf | Output format: `pdf` (editable, recommended), `png`, or `svg` |
| dpi | 300 | Figure resolution: 150, 300, or 600 DPI |
""")

with tab_figs:
    st.markdown("""
## Figures

All figures follow **Nature Publishing Group** standards:
- Column widths: 89 mm (single) / 183 mm (double column)
- Font: Arial, 6 pt minimum (body), 7 pt axis labels, 8 pt axis titles
- 300 DPI minimum
- Editable PDFs: Type 42 embedded fonts (open in Adobe Illustrator / Affinity Publisher)
- Wong (2011) colour-blind-safe palette for conditions; tab20 extended for >8 clusters

| Figure | Description |
|--------|-------------|
| fig1_qc | QC violins: counts, genes per cell; lollipop of cells per slide |
| fig2_umap | UMAP coloured by condition and Leiden cluster |
| fig3_spatial_clusters | Spatial cluster maps in the MBH ROI, one panel per condition |
| fig4_dotplot | Marker gene dot plot (size = % expressing, colour = mean expression) |
| fig5_volcano | AGED vs ADULT global volcano plot, top genes labelled |
| fig6_heatmap | Top DEG heatmap, z-scored, cells sorted by condition |
| fig7_spatial_expr | Spatial expression maps of top up-regulated DEGs |
| fig8_summary | 6-panel composite: UMAP + spatial + volcano + barplots |
| fig9_cell_types | Cell type annotation: UMAP + spatial + proportion bar + confidence |
| fig10_spatial_stats | Moran's I lollipop + spatial co-expression + neighbourhood enrichment |
| fig11_cluster_dge | Per-cluster DEG counts (panel a), bubble chart of top DEGs per cluster (panel b) |
| fig12_slide_qc | Per-slide cell counts, transcript yield, condition balance |
| fig13_panel_qc | Panel composition bars + custom gene presence heatmap + UpSet histogram |
| fig14_insulin | Insulin/metabolic signalling gene panel across cell types |
| fig15_galanin | Galanin (Gal): spatial maps ADULT/AGED (panels a,b), split violin per cell type (c), per-cell-type log₂FC lollipop with BH-corrected significance (d) |
| fig16_composition | Cell type composition testing: stacked proportion bars per replicate (a), forest plot of log₂FC per cell type with credible intervals and significance (b); scCODA Bayesian model (Büttner 2021) with CLR+Welch t-test fallback |
| fig17_neuropeptide_modules | Neuropeptide co-expression modules: UMAP coloured by dominant module (a), z-scored mean score per cell type × module heatmap (b), AGED vs ADULT grouped bar with Mann-Whitney significance (c), spatial dominant-module maps per condition (d). Modules: AgRP/NPY, POMC/CART, KNDy, Somatostatin, TRH/Dopamine, Galanin. |
""")

with tab_trouble:
    st.markdown("""
## Troubleshooting

**`No module named scanpy`**
Activate the conda environment before starting the app:
```bash
conda activate xenium_dge
cd app && streamlit run app.py
```

**`cell_feature_matrix/ not found`**
Check that the selected folder is the *Xenium run output* directory, not a parent folder.
It must directly contain `cell_feature_matrix/` and `cells.parquet`.

**Plotly polygon tool not working**
Use the manual coordinate entry panel in the ROI Manager instead.
Paste x,y pairs (µm) one per line and click Save.

**Custom genes not appearing after harmonisation**
Check `panel_validation.csv` in the output directory.
If `n_custom = 0` for a slide, lower `min_slides` in Settings or switch to `union` mode.

**PyDESeq2 finds no significant genes**
PyDESeq2 pseudobulk requires ≥ 8 biological replicates per condition to have
adequate power. With n=4 per group it typically returns no significant results.
Use **Stringent Wilcoxon** (recommended for n=4) or **C-SIDE pseudobulk** for
per-cell-type analysis. PyDESeq2 is included for studies with larger n.

**Pipeline log shows `harmony` error**
```bash
conda install -n xenium_dge -c conda-forge harmonypy -y
```

**Figures are not appearing in Results**
Check the output directory path in Study Setup. The pipeline writes figures to the
directory you set there. Make sure it matches what you see in Results.

**App is slow to load slide data**
The ROI Manager loads `cells.parquet` (all cell centroids) on demand.
For very large slides (>50,000 cells) this may take a few seconds.
The scatter is automatically subsampled to 15,000 cells for display speed.
""")

st.divider()
st.markdown(
    "**Key references:** "
    "Wolf et al. 2018 (scanpy) · "
    "Korsunsky et al. 2019 (Harmony) · "
    "Muzellec et al. 2023 (PyDESeq2) · "
    "McInnes et al. 2018 (UMAP) · "
    "Büttner et al. 2021 *Nat Commun* (scCODA) · "
    "Cable et al. 2022 *Nat Methods* (C-SIDE) · "
    "Wong 2011 *Nat Methods* (colour palette) · "
    "10x Genomics (Xenium)"
)
