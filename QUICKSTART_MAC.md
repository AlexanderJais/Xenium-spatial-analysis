# Xenium Sample PCA — macOS Quick Start (Apple Silicon)

> Tested on MacBook Pro M1/M2/M3/M4, macOS Ventura/Sonoma/Sequoia.
> For full documentation see [README.md](README.md).

---

## Prerequisites

You need **nothing** pre-installed except macOS. The installer handles everything.

---

## 1. Install (one-time)

Open **Terminal**, navigate to the project folder, and run:

```bash
cd /path/to/xenium-spatial-analysis
chmod +x install_mac.sh
./install_mac.sh
```

This installs Miniforge3 (ARM64 conda) if needed, creates a Python 3.11 environment, and installs the dependencies:

| Component | Details |
|-----------|---------|
| Python 3.11 | Native Apple Silicon via conda-forge |
| Core stack | numpy, pandas, scipy, scikit-learn, matplotlib |
| Data | anndata, pyarrow (`cells.parquet`) |
| Web interface | streamlit, plotly |

Alternatively, in any environment: `pip install -r requirements.txt`.

---

## 2. Launch

**Option A — Web app (recommended):**

Double-click `start_app.command` in Finder, or from Terminal:
```bash
conda activate xenium_sample_pca
streamlit run app/app.py
```
Your browser opens at http://localhost:8501.

**Option B — Command line (headless, after ROIs are saved):**
```bash
conda activate xenium_sample_pca
python run_sample_pca.py            # apply saved ROIs, run PCA
python run_sample_pca.py --no-roi   # use whole sections
```

---

## 3. Step 1 — Study Setup

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

Go to **📁 Study Setup** and paste the full path to each run directory. Use **➕ Add slide** / the 🗑 button to match your sample count (the page starts with the 4 + 4 AGED/ADULT template), and edit the condition labels if your groups differ. A green tick confirms validity; the page shows the cell and gene counts per slide once validated.

**Tip:** On macOS, right-click a folder in Finder → Get Info → copy the path from *Where*.

Click **Save configuration to JSON** to store all paths so you never re-enter them — **Load** restores them next session. See the [README](README.md#configuration-file-format) for the schema.

---

## 4. Step 2 — ROI Manager

Define the mediobasal hypothalamus (MBH) boundary on each slide.

1. Select a slide from the dropdown.
2. Use the four sliders to frame the MBH bounding rectangle (left/right x, top/bottom y).
3. The scatter and cell count update live as you adjust.
4. The dashed orange ellipse is an anatomical atlas hint — the MBH sits in the ventral 50–80% of a coronal section (larger y = ventral).
5. Click **Save ROI** when the rectangle covers the MBH.

**Precise coordinates:** use the *Paste coordinates* panel — one `x, y` pair per line in micrometres:
```
3200, 4100
3800, 4100
3800, 4700
3200, 4700
```

**Copy ROIs:** if sections are at similar coordinates, save once and copy to other slides via *Copy to other slides*. Saved ROIs live in `roi_cache/` and are reused automatically on every run.

---

## 5. Step 3 — Sample PCA

Go to **📊 Sample PCA** and click **Run sample PCA**. The app loads the slides, applies the saved ROIs, pseudobulks each sample, and runs PCA across them.

Options:
- **Samples to include** — pick which samples go into the PCA (minimum 2); the rest are ignored for that run.
- **Base panel only** — on by default; restricts the PCA to the shared base panel so samples with different add-on panels stay comparable. Untick to include add-on genes.
- **Apply MBH ROIs** — on by default once ROIs exist; turn off to use whole sections.
- **Top variable genes** — 0 uses all genes (recommended for the targeted panel).
- **Z-score genes** — off by default (`log1p` already stabilises variance).

You get three figures inline:
- **PCA scatter** — PC1 vs PC2, samples coloured by group (AGED/ADULT) and individually labelled.
- **Correlation heatmap** — sample-by-sample correlation, hierarchically ordered (spot outliers).
- **Scree plot** — variance explained per PC.

### Output files

Written to `<output_dir>/sample_pca/`:

| File | Description |
|------|-------------|
| `sample_pca_scatter.pdf` | PC1 vs PC2 coloured by group |
| `sample_correlation_heatmap.pdf` | Hierarchically-ordered sample correlation |
| `sample_pca_scree.pdf` | Variance explained per PC |
| `sample_pca_coordinates.csv` | PC coordinates + metadata per sample |
| `sample_pca_variance.csv` | Variance ratios |
| `pseudobulk_samples.h5ad` | Pseudobulk AnnData |

---

## Troubleshooting

**ROI sliders not responding**
Refresh the page (Cmd+R). If it persists, use the *Paste coordinates* panel.

**`cell_feature_matrix/` not found**
The path must be the Xenium run directory itself (containing `cell_feature_matrix/` and `cells.parquet`), not a parent folder.

**ROI selects 0 cells**
The MBH is ventral (larger y). Use the dashed orange atlas-hint ellipse as a guide and re-frame.

**PCA separates samples by cell number, not biology**
The built-in workflow always CPM-normalises before PCA. If you call the functions directly, run `normalize_pseudobulk` before `run_sample_pca`.

**App is slow to load a slide scatter**
The ROI Manager loads `cells.parquet` on demand and subsamples large slides for display. A few seconds for very large sections is normal.

For more, see the [README](README.md#troubleshooting).
