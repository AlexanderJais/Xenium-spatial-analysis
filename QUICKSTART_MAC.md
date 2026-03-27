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
| Miniforge3 (ARM64 conda) | latest | Homebrew cask |
| Python 3.11 | ARM64 native | conda-forge |
| numpy, pandas, scipy, matplotlib, seaborn | latest | conda-forge |
| scanpy + anndata | latest | conda-forge |
| leidenalg + igraph | latest | conda-forge (native ARM64) |
| harmonypy | latest | pip |
| PyDESeq2 | latest | pip |
| umap-learn | latest | conda-forge |
| Tkinter | bundled | conda-forge `tk` |

Total disk: ~1.8 GB. Install time: ~8 minutes on a fast connection.

---

## 2. Place your panel metadata

Your `Xenium_mBrain_v1_1_metadata.csv` is already included at:

```
xenium_dge/data/Xenium_mBrain_v1_1_metadata.csv
```

---

## 3. Launch the GUI

```bash
conda activate xenium_dge
python launcher.py
```

A window opens:

```
┌─────────────────────────────────────────────────────────────────┐
│  Xenium DGE Pipeline — AGED vs ADULT MBH                       │
├─────────────────────────────────────────────────────────────────┤
│  Slide folders                                                  │
│  AGED    AGED_1   [/path/to/aged_run_1]        [Browse…] [✕]   │
│  AGED    AGED_2   [/path/to/aged_run_2]        [Browse…] [✕]   │
│  AGED    AGED_3   [/path/to/aged_run_3]        [Browse…] [✕]   │
│  AGED    AGED_4   [/path/to/aged_run_4]        [Browse…] [✕]   │
│  ADULT   ADULT_1  [/path/to/adult_run_1]       [Browse…] [✕]   │
│  ADULT   ADULT_2  [/path/to/adult_run_2]       [Browse…] [✕]   │
│  ADULT   ADULT_3  [/path/to/adult_run_3]       [Browse…] [✕]   │
│  ADULT   ADULT_4  [/path/to/adult_run_4]       [Browse…] [✕]   │
├─────────────────────────────────────────────────────────────────┤
│  Files    Base panel CSV: [data/Xenium_mBrain_v1_1…] [Browse…] │
│           Output dir:     [~/xenium_dge_output]       [Browse…] │
│           ROI cache:      [roi_cache/]                [Browse…] │
├─────────────────────────────────────────────────────────────────┤
│  Options  DGE method:   ● Wilcoxon  ○ PyDESeq2                 │
│           Panel mode:   ● Intersection  ○ Union                │
│           ROI mode:     ● Polygon  ○ Lasso  ○ Rectangle        │
├─────────────────────────────────────────────────────────────────┤
│  [✓ Validate]  [💾 Save config]  [📂 Load config]  [▶ Run]     │
├─────────────────────────────────────────────────────────────────┤
│  Pipeline log (live)                                           │
│  > Loading AGED_1 …                                            │
│  > Harmony integration …                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Browse to your Xenium folders

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

Click **Browse …** next to each slide row and navigate to the folder.
Rename Slide IDs in the text boxes if needed (e.g. `AGED_Bregma-1.8`).

---

## 5. Configure options

| Option | Recommendation |
|--------|---------------|
| **DGE method** | Start with **Wilcoxon** (fast). Switch to **PyDESeq2** for publication (needs 3+ true biological replicates). |
| **Panel mode** | **Intersection** — keeps only the 247 base genes guaranteed in every slide. Safe default. |
| **ROI mode** | **Polygon** — click vertices around the MBH on each section. A dashed atlas hint ellipse is shown. |

---

## 6. Save / load your config

Click **Save config** to write a JSON file with all paths and settings.
Click **Load config** to restore a previous session — no need to re-browse all 8 folders.

---

## 7. Run

Click **▶ Run Pipeline**. The log panel streams live output. On the first run:

1. An MBH ROI drawing window opens for each slide.
2. Draw a polygon around the mediobasal hypothalamus.
3. Right-click or press Enter to close the polygon.
4. The ROI is saved to `roi_cache/<slide_id>_roi.json`.
5. Subsequent runs reuse the saved ROIs — no GUI needed.

---

## 8. Outputs

All 12 figures are saved to your chosen output directory as **editable PDFs**
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

Plus:
- `global_dge_aged_vs_adult.csv` — full DGE results table
- `cluster_dge_results.csv` — per-cluster DGE
- `morans_i_mbh.csv` — spatially variable genes
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
set `n_top_genes = 100` in the launcher to reduce memory.

---

## Troubleshooting

**"No module named scanpy"**
```bash
conda activate xenium_dge
python launcher.py   # always activate first
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

**PyDESeq2 is slow with 4 replicates**
Switch to `wilcoxon` in the launcher options for exploratory runs.
PyDESeq2 is recommended for final publication figures only.
