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
- **DGE method:** `stringent_wilcoxon` (default, recommended); `wilcoxon` for speed; `pydeseq2` requires n≥8 replicates
- **min_slides:** How many slides a custom gene must appear in to be retained (default 2)

### 3 — ROI Manager
Go to **🗺️ ROI Manager** and draw the mediobasal hypothalamus boundary on each section.
The dashed orange ellipse is an anatomical hint. Saved polygons are reused automatically.

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

### Drawing a polygon
1. Open **🗺️ ROI Manager** and select a slide
2. Click the **polygon tool** (pentagon icon) in the Plotly chart toolbar (top-right of the chart)
3. Click on the scatter plot to place vertices around the MBH
4. **Double-click** to close the polygon
5. The polygon is saved automatically

### Atlas hint
The dashed orange ellipse marks the approximate location of the mediobasal hypothalamus
(ventromedial nucleus + arcuate nucleus region). It is based on relative tissue coordinates
and should be used as a starting guide only — adjust your polygon to the actual histology
of each section.

### Manual coordinate entry
If the Plotly drawing tool does not work in your browser, use the
**Enter coordinates manually** panel on the right side of the ROI Manager.
Paste x,y pairs (in µm) one per line:
```
3200, 4100
3800, 4100
3800, 4700
3200, 4700
```

### Copying ROIs
If your sections are at similar coordinates you can draw the ROI on one slide
and copy it to others using the **Copy this ROI** expander.

### ROI files
Polygons are saved as JSON files in `roi_cache/`:
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
| max_counts | 2000 | Cells above this are likely doublets |
| min_genes | 10 | Cells with fewer unique genes are removed |
| max_genes | 300 | Upper gene count filter |

### Preprocessing
| Parameter | Default | Description |
|-----------|---------|-------------|
| n_top_genes | 0 | Highly variable genes for PCA (0 = disabled, use all genes — recommended for Xenium) |
| leiden_resolution | 0.6 | Higher = more, smaller clusters |
| n_neighbors | 12 | KNN graph neighbours for UMAP and clustering |
| harmony_max_iter | 20 | Harmony batch correction iterations |

**Important:** `harmony_key` is always `slide_id` (not condition).
This corrects technical variation between the 8 slides while preserving
the biological AGED vs ADULT signal.

### DGE
| Parameter | Default | Description |
|-----------|---------|-------------|
| dge_method | stringent_wilcoxon | `stringent_wilcoxon` (recommended), `wilcoxon`, or `pydeseq2` (pseudobulk) |
| log2fc_threshold | 1.0 | Minimum |log₂FC| for significance (stringent_wilcoxon uses this) |
| pval_threshold | 0.01 | Adjusted p-value threshold (stringent_wilcoxon uses this) |

### Panel
| Parameter | Default | Description |
|-----------|---------|-------------|
| panel_mode | partial_union | How to harmonise custom genes across slides |
| min_slides | 2 | Custom genes must appear in ≥ this many slides |
""")

with tab_figs:
    st.markdown("""
## Figures

All figures follow **Nature Publishing Group** standards:
- Column widths: 89 mm (single) / 183 mm (double column)
- Font: Arial 7 pt body, 8 pt axis titles
- 300 DPI minimum
- Editable PDFs: Type 42 embedded fonts (open in Adobe Illustrator / Affinity Publisher)
- Wong (2011) colour-blind-safe palette

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
| fig15_galanin | Galanin regulation: log₂FC per cluster (bar) + spatial Gal expression ADULT and AGED |
""")

with tab_trouble:
    st.markdown("""
## Troubleshooting

**`No module named scanpy`**
Activate the conda environment before starting the app:
```bash
conda activate xenium_dge
streamlit run app/app.py
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

**PyDESeq2 is slow**
PyDESeq2 pseudobulk with 4 replicates is the correct method for publication but
takes longer than Wilcoxon. Use Wilcoxon for exploratory runs and switch to
PyDESeq2 for final figures.

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
    "**Citation:** Wolf et al. (scanpy) · Korsunsky et al. (Harmony) · "
    "Muzellec et al. (PyDESeq2) · McInnes et al. (UMAP) · 10x Genomics (Xenium)"
)
