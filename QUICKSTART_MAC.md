# Xenium DGE Pipeline — macOS Quick Start (Apple Silicon)

> Tested on MacBook Pro M1/M2/M3/M4, macOS Ventura/Sonoma/Sequoia.

---

## Prerequisites (5 minutes)

You need **nothing** pre-installed except macOS. The installer handles everything.

---

## 1. Install (one-time)

Open **Terminal**, navigate to the project folder, then run:

```bash
cd /path/to/xenium_dge
chmod +x install_mac.sh
./install_mac.sh
```

This installs:

| Package | Version | How |
|---------|---------|-----|
| Miniforge3 (ARM64 conda) | latest | curl direct |
| Python 3.11 | ARM64 native | conda-forge |
| numpy, pandas, scipy, matplotlib, seaborn | latest | conda-forge |
| statsmodels | latest | conda-forge |
| scanpy + anndata | latest | conda-forge |
| leidenalg + igraph + umap-learn | latest | conda-forge (native ARM64) |
| streamlit + plotly | latest | pip |
| harmonypy | latest | pip |
| PyDESeq2 | latest | pip |
| squidpy | latest | pip (optional) |
| Tkinter | bundled | conda-forge `tk` |

Total disk: ~1.8 GB. Install time: ~8 minutes on a fast connection.

---

## 2. Place your panel metadata

Your `Xenium_mBrain_v1_1_metadata.csv` is already included at:

```
xenium_dge/data/Xenium_mBrain_v1_1_metadata.csv
```

---

## 3. Launch the web interface (recommended)

**Option A — double-click (easiest):**
Find `start_app.command` in Finder and double-click it.
Your browser opens automatically at http://localhost:8501.

**Option B — terminal:**
```bash
conda activate xenium_dge
cd app && streamlit run app.py
```

**Option C — desktop GUI (alternative):**
```bash
conda activate xenium_dge
python launcher.py
```
A Tkinter window opens with fields for slide folders, output directory,
DGE method, panel mode, and a live log panel.

---

## 4. Enter your Xenium folder paths

Each Xenium run directory must contain:

```
<run_dir>/
    cell_feature_matrix/
        barcodes.tsv.gz
        features.tsv.gz
        matrix.mtx.gz
    cells.parquet
    experiment.xenium
```

**Web app:** Go to **📁 Study Setup** and paste the full path to each run directory.
A green tick confirms the directory is valid.

**Desktop launcher:** Click **Browse …** next to each slide row and navigate to the folder.

Rename Slide IDs in the text boxes if needed (e.g. `AGED_Bregma-1.8`).

---

## 5. Configure options

| Option | Recommendation |
|--------|---------------|
| **DGE method** | **stringent_wilcoxon** (default) — cell-level Wilcoxon with post-hoc replication filter; best for n=4 per condition. Use **cside** (Cable et al. 2022) for per-cell-type pseudobulk DESeq2 (publication). **pydeseq2** needs ≥8 replicates to have power. |
| **Panel mode** | **partial_union** (default) — keeps base genes + custom genes present in ≥2 slides. Recommended for this study. |
| **ROI mode** | **Polygon** — click vertices around the MBH on each section. A dashed atlas hint ellipse is shown. |

---

## 6. Save / load your config

Click **Save config** to write a JSON file with all paths and settings.
Click **Load config** to restore a previous session — no need to re-browse all 8 folders.

---

## 7. Draw ROIs, then run

**Web app:** Go to **🗺️ ROI Manager** and draw the MBH boundary on each slide
*before* clicking Run Pipeline. Use the polygon tool in the chart toolbar,
double-click to close. The dashed orange ellipse is an anatomical atlas hint.
Saved ROIs are reused automatically on every subsequent run.

Then go to **🚀 Run Pipeline** and click **▶ Run Pipeline**.
The log streams live; a Stop button is always available.

**CLI / launcher (first run only):**

1. A Matplotlib drawing window opens for each slide.
2. Draw a polygon around the mediobasal hypothalamus.
3. Right-click or press Enter to close the polygon.
4. The ROI is saved to `roi_cache/<slide_id>_roi.json`.
5. Subsequent runs reuse the saved ROIs automatically.

---

## 8. Outputs

All 17 figures are saved to your chosen output directory as **editable PDFs**
(Type 42 fonts, readable in Adobe Illustrator / Affinity Publisher):

| Figure | Content |
|--------|---------|
| fig1_qc.pdf | QC violins + spatial count density |
| fig2_umap.pdf | UMAP by condition and cluster |
| fig3_spatial_clusters.pdf | Spatial cluster maps (MBH) |
| fig4_dotplot.pdf | Marker gene dot plot |
| fig5_volcano.pdf | AGED vs ADULT global volcano |
| fig6_heatmap.pdf | Top DEG heatmap (z-scored) |
| fig7_spatial_expr.pdf | Spatial expression of top DEGs |
| fig8_summary.pdf | 6-panel composite summary |
| fig9_cell_types.pdf | Cell type annotation in MBH |
| fig10_spatial_stats.pdf | Moran's I + neighbourhood enrichment |
| fig11_cluster_dge.pdf | Per-cluster DEG counts + bubble chart |
| fig12_slide_qc.pdf | Per-slide QC + MBH yield overview |
| fig13_panel_qc.pdf | Panel composition + custom gene overlap |
| fig14_insulin.pdf | Insulin/metabolic signalling gene panel |
| fig15_galanin.pdf | Galanin spatial maps, violin & log₂FC lollipop |
| fig16_composition.pdf | Cell type composition testing (scCODA) |
| fig17_neuropeptide_modules.pdf | Neuropeptide co-expression modules (AgRP/NPY, POMC/CART, KNDy, SST, TRH/DA, Galanin) |

Plus:
- `global_dge_aged_vs_adult.csv` — full DGE results table
- `cluster_dge_results.csv` — per-cluster DGE
- `cluster_dge_summary.csv` — DEG counts per cluster
- `morans_i_mbh.csv` — spatially variable genes
- `panel_validation.csv` — per-slide gene panel composition
- `adata_mbh_final.h5ad` — final annotated AnnData

---

## Running without the GUI (terminal only)

```bash
conda activate xenium_dge

# Edit the paths directly in run_xenium_mbh.py, then:
python run_xenium_mbh.py

# Skip ROI drawing (uses saved or preset atlas ROIs):
python run_xenium_mbh.py --no-roi-gui

# Force ROI redraw for all slides:
python run_xenium_mbh.py --redraw-roi
```

---

## Memory usage (48 GB M4 Pro)

| Step | Approx. RAM |
|------|------------|
| Loading 8 slides (2 500–3 500 cells each) | ~2 GB |
| After MBH ROI filter (~15% of tissue) | ~400 MB |
| PCA + Harmony | ~1 GB peak |
| UMAP | ~500 MB |
| PyDESeq2 pseudobulk | ~200 MB |
| All figures | ~500 MB |
| **Total peak** | **~3–4 GB** |

Your 48 GB is more than sufficient. For datasets with 50 000+ cells per slide,
set `n_top_genes = 100` in the **⚙️ Settings** page (web app) or edit
`CFG.n_top_genes` in `run_xenium_mbh.py` to reduce memory.

---

## Troubleshooting

**"No module named scanpy"**
```bash
conda activate xenium_dge
cd app && streamlit run app.py   # web interface
# or:
python launcher.py               # desktop launcher
```

**matplotlib window does not appear**
```bash
# Ensure the MacOSX backend is set
echo "backend: MacOSX" >> ~/.matplotlib/matplotlibrc
```

**leidenalg ImportError**
```bash
conda install -n xenium_dge -c conda-forge leidenalg -y
```

**PyDESeq2 is slow or finds no significant genes with n=4**
PyDESeq2 pseudobulk requires ≥8 biological replicates per condition to have
adequate power. With n=4 per group it typically returns no significant results.
Use **stringent_wilcoxon** (default, recommended for n=4) or **cside** for
per-cell-type pseudobulk analysis (publication-grade). Switch in the
**⚙️ Settings** page or set `CFG.dge_method` in `run_xenium_mbh.py`.
