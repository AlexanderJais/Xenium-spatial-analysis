# Xenium DGE Pipeline -- macOS Quick Start (Apple Silicon)

> Tested on MacBook Pro M1/M2/M3/M4, macOS Ventura/Sonoma/Sequoia.
> For full documentation see [README.md](README.md).

---

## Prerequisites

You need **nothing** pre-installed except macOS. The installer handles everything.

---

## 1. Install (~8 minutes, one-time)

Open **Terminal**, navigate to the project folder, and run:

```bash
cd /path/to/xenium_dge
chmod +x install_mac.sh
./install_mac.sh
```

This installs:

| Component | Details |
|-----------|---------|
| Miniforge3 (ARM64 conda) | Downloaded automatically if not present |
| Python 3.11 | Native Apple Silicon via conda-forge |
| Scientific stack | numpy, pandas, scipy, matplotlib, seaborn, statsmodels |
| Single-cell | scanpy, anndata, leidenalg, igraph, umap-learn |
| Batch correction | harmonypy |
| DGE | PyDESeq2 (optional, Wilcoxon fallback available) |
| Web interface | streamlit, plotly |
| Spatial (optional) | squidpy |
| GUI support | Tkinter (bundled via conda-forge `tk`) |

Total disk: ~1.8 GB.

---

## 2. Launch

**Option A -- Web app (recommended):**

Double-click `start_app.command` in Finder. Your browser opens at http://localhost:8501.

Or from Terminal:
```bash
conda activate xenium_dge
streamlit run app/app.py
```

**Option B -- Desktop GUI:**
```bash
conda activate xenium_dge
python launcher.py
```

**Option C -- Command line (headless):**
```bash
conda activate xenium_dge
python run_xenium_mbh.py               # interactive ROI drawing
python run_xenium_mbh.py --no-roi-gui  # use saved/preset ROIs
python run_xenium_mbh.py --redraw-roi  # force ROI redraw for all slides
```

---

## 3. Enter your Xenium folder paths

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

**Web app:** Go to **📁 Study Setup** and paste the full path to each run directory. A green tick confirms validity. The page shows cell count and gene count per slide once validated.

**Desktop launcher:** Click **Browse...** next to each slide row.

**Tip:** On macOS, right-click a folder in Finder -> Get Info -> copy the path from *Where*.

---

## 4. Configure settings

Go to **⚙️ Pipeline Settings** (web app) or the Settings panel (launcher). Key decisions:

| Option | Default | Recommendation |
|--------|---------|----------------|
| **DGE method** | `stringent_wilcoxon` | Best for n=4 per condition. Use `cside` for per-cell-type pseudobulk (publication). `pydeseq2` needs n >= 8. |
| **Panel mode** | `partial_union` | Keeps base + custom genes in >= 2 slides. Best balance of coverage vs noise. |
| **Leiden resolution** | 0.6 | Increase for finer clusters, decrease for coarser. |
| **Figure format** | `pdf` | Editable in Illustrator/Affinity. `png` or `svg` also available. |

All parameters have sensible defaults -- you can skip this step for a first run.

---

## 5. Save / load config

Click **Save config** to write all paths and settings to a JSON file. Click **Load config** to restore a previous session -- no need to re-enter all 8 folder paths.

See the [README](README.md#configuration-file-format) for the JSON schema.

---

## 6. Draw ROIs

**Before running the pipeline**, define the mediobasal hypothalamus boundary on each slide.

**Web app (🗺️ ROI Manager):**
1. Select a slide from the dropdown
2. Click the **polygon tool** (pentagon icon) in the Plotly chart toolbar
3. Click to place vertices around the MBH
4. **Double-click** to close the polygon
5. The dashed orange ellipse is an anatomical atlas hint -- adjust to your histology
6. ROI is saved automatically to `roi_cache/`

**If the drawing tool doesn't work in your browser:** use the **manual coordinate entry** panel. Paste x,y pairs in micrometres, one per line:
```
3200, 4100
3800, 4100
3800, 4700
3200, 4700
```

**Copy ROIs:** If sections are at similar coordinates, draw once and copy to other slides using the **Copy this ROI** expander.

**CLI mode:** A Matplotlib window opens per slide. Draw a polygon, then right-click or press Enter to close.

Saved ROIs are reused automatically on every subsequent run.

---

## 7. Run the pipeline

**Web app:** Go to **🚀 Run Pipeline** and click **Run Pipeline**. Pre-flight checks validate all slides first. The log streams live with colour-coded output. A Stop button is always available.

**Runtime:** ~15--25 minutes for 8 slides on M4 Pro.

---

## 8. View results

**Web app (📊 Results):**
- All 17 figures displayed inline with dropdown selector and thumbnail gallery
- Download buttons for each figure
- Tabs for: Global DGE table, Cluster DGE table, Moran's I, Panel validation, AnnData `.h5ad`

**Gene Explorer (🔬):**
- Generate on-demand spatial expression maps for any gene
- Reads from the preprocessed AnnData cache -- no pipeline rerun needed

### Output files

All files are saved to your output directory:

| File | Description |
|------|-------------|
| `fig1_qc.pdf` ... `fig17_neuropeptide_modules.pdf` | 17 publication-ready figures (Nature PG standards, editable PDF) |
| `global_dge_aged_vs_adult.csv` | Full DGE results |
| `cluster_dge_results.csv` | Per-cluster DGE |
| `cluster_dge_summary.csv` | DEG counts per cluster |
| `morans_i_mbh.csv` | Spatially variable genes |
| `panel_validation.csv` | Per-slide panel composition |
| `adata_mbh_final.h5ad` | Final annotated AnnData |

---

## Memory usage

| Step | Approx. RAM |
|------|-------------|
| Loading 8 slides (~3 000 cells x ~297 genes each) | ~2 GB |
| After MBH ROI filter (~15% of tissue) | ~400 MB |
| PCA + Harmony | ~1 GB peak |
| UMAP + figures | ~1 GB |
| **Total peak** | **~3--4 GB** |

48 GB is more than sufficient. For very large slides (50 000+ cells), set `n_top_genes = 100` in Settings.

---

## Troubleshooting

**"No module named scanpy"**
```bash
conda activate xenium_dge   # always activate first
```

**Matplotlib window does not appear**
```bash
echo "backend: MacOSX" >> ~/.matplotlib/matplotlibrc
```

**leidenalg ImportError**
```bash
conda install -n xenium_dge -c conda-forge leidenalg -y
```

**PyDESeq2 finds no significant genes**
Expected with n=4 per condition -- pseudobulk needs >= 8 replicates for power. Use `stringent_wilcoxon` (default) or `cside` instead.

**Pipeline log shows Harmony error**
```bash
pip install harmonypy
```

**Figures not appearing in Results**
Verify the output directory in Study Setup matches what the pipeline used. Check the log for the exact output path.

**App is slow to load slide scatter**
The ROI Manager loads `cells.parquet` on demand. For very large slides (>50 000 cells) this may take a few seconds. The scatter is automatically subsampled to 15 000 cells for display speed.

For more troubleshooting, see the **ℹ️ Help** page in the web app or the [README](README.md#troubleshooting).
