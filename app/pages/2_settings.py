"""
pages/2_settings.py
Pipeline Settings page.
"""

import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(page_title="Settings · Xenium DGE", page_icon="⚙️", layout="wide",
    initial_sidebar_state="expanded")


inject_css()
# ── Defaults ─────────────────────────────────────────────────────────────────
for k, v in {
    "panel_mode": "partial_union", "min_slides": 2,
    "dge_method": "stringent_wilcoxon", "leiden_resolution": 0.6,
    "n_neighbors": 12, "min_counts": 10, "max_counts": 2000,
    "min_genes": 10, "max_genes": 300, "log2fc_threshold": 1.0,
    "pval_threshold": 0.01, "n_top_genes": 200, "harmony_max_iter": 20,
    "roi_mode": "polygon", "figure_format": "pdf", "dpi": 300,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

page_header("⚙️ Pipeline Settings", "Adjust QC thresholds, integration, DGE, and figure parameters")
st.markdown("All parameters are saved in your session and passed to the pipeline on run.")
st.divider()

# ── Panel harmonisation ───────────────────────────────────────────────────────
st.subheader("Gene panel harmonisation")
st.markdown(
    "Each slide has **~247 base genes** (identical across all slides) "
    "plus **~50 custom genes** (differs between slides). "
    "Choose how custom genes are handled when combining all 8 slides."
)

col1, col2 = st.columns([2, 1])
with col1:
    mode = st.radio(
        "Panel mode",
        options=["partial_union", "intersection", "union"],
        format_func=lambda x: {
            "partial_union" : "✅ Partial union — base genes + shared custom genes (recommended)",
            "intersection"  : "🔒 Intersection — base genes only (no custom genes)",
            "union"         : "🔓 Union — all custom genes from any slide (max zero-inflation)",
        }[x],
        index=["partial_union","intersection","union"].index(
            st.session_state["panel_mode"]
        ),
        help="Partial union is best: keeps custom genes present in multiple slides.",
    )
    st.session_state["panel_mode"] = mode

with col2:
    if mode == "partial_union":
        min_slides = st.number_input(
            "Min slides (custom gene threshold)",
            min_value=1, max_value=8,
            value=st.session_state["min_slides"],
            help=(
                "A custom gene is kept only if it appears in at least this many slides. "
                "Slides missing a kept gene receive a zero-filled column. "
                "Default 2 = keep genes present in ≥2/8 slides."
            ),
        )
        st.session_state["min_slides"] = int(min_slides)
        st.caption(
            f"Custom genes in ≥ **{int(min_slides)}** of 8 slides are retained. "
            f"Unique-to-one-slide genes are dropped."
        )
    else:
        st.info("min_slides not used in this mode")

st.divider()

# ── QC ────────────────────────────────────────────────────────────────────────
st.subheader("Quality control")
qc1, qc2, qc3, qc4 = st.columns(4)

with qc1:
    v = st.number_input("Min counts per cell", 1, 500,
                         st.session_state["min_counts"],
                         help="Cells with fewer total transcripts are removed.")
    st.session_state["min_counts"] = int(v)

with qc2:
    v = st.number_input("Max counts per cell", 100, 10000,
                         st.session_state["max_counts"],
                         help="Cells exceeding this are likely doublets.")
    st.session_state["max_counts"] = int(v)

with qc3:
    v = st.number_input("Min genes per cell", 1, 100,
                         st.session_state["min_genes"],
                         help="Cells with fewer unique genes are removed. "
                              "Raised to 10 (from 5) for the Xenium 307-gene panel — "
                              "any cell expressing fewer than 10 genes is likely empty or damaged.")
    st.session_state["min_genes"] = int(v)

with qc4:
    v = st.number_input("Max genes per cell", 50, 500,
                         st.session_state["max_genes"])
    st.session_state["max_genes"] = int(v)

# ── Xenium-specific QC controls ───────────────────────────────────────────────
st.markdown("**Xenium negative control probe filter**")
xqc1, xqc2, xqc3 = st.columns([1, 1, 2])
with xqc1:
    v = st.checkbox(
        "Filter control probe counts > 0",
        value=st.session_state.get("filter_control_probes", True),
        help="Remove cells where any negative control probe signal was detected. "
             "The Xenium Mouse Brain panel has 27 negative control probe sets — "
             "the most of any Xenium panel. 10x Genomics sets a warning threshold "
             "at 0.02 and error at 0.05 for this metric. Enabled by default.",
    )
    st.session_state["filter_control_probes"] = v

with xqc2:
    v = st.checkbox(
        "Filter control codeword counts > 0",
        value=st.session_state.get("filter_control_codewords", True),
        help="Remove cells with spurious barcode decoding events "
             "(control_codeword_counts > 0). These indicate optical artefacts.",
    )
    st.session_state["filter_control_codewords"] = v

with xqc3:
    st.info(
        "ℹ️  **Q-score:** Transcript-level Q20 filtering is applied automatically "
        "by Xenium Onboard Analysis (XOA) before the cell-by-gene matrix is built. "
        "No additional setting is needed here.",
        icon=None,
    )

st.divider()

# ── Preprocessing ─────────────────────────────────────────────────────────────
st.subheader("Preprocessing")
p1, p2, p3 = st.columns(3)

with p1:
    st.info(
        "**Normalisation: target_sum = 100** (Xenium-specific). "
        "Standard scRNA-seq uses 10 000, but Xenium cells detect far fewer "
        "transcripts from a targeted panel. The Salas 2025 *Nature Methods* "
        "benchmark found target_sum = 100 is the top-performing setting for "
        "Xenium data (three orders of magnitude lower than the scRNA-seq default).",
        icon="ℹ️",
    )
    st.info(
        "**HVG selection: disabled.** "
        "~96% of Xenium panel genes are spatially variable — filtering adds no "
        "discriminatory value and introduces stochasticity. All genes are used for PCA.",
        icon="ℹ️",
    )
    st.session_state["n_top_genes"] = 0

with p2:
    v = st.slider(
        "Leiden resolution",
        0.1, 2.0, float(st.session_state["leiden_resolution"]), 0.05,
        help="Higher values produce more, smaller clusters. "
             "Raised to 0.6 (from 0.5) per Xenium benchmark recommendation — "
             "the 307-gene panel resolves more subtypes than generic scRNA-seq panels.",
    )
    st.session_state["leiden_resolution"] = float(v)

with p3:
    v = st.number_input(
        "KNN neighbours (UMAP / graph)",
        5, 50, st.session_state["n_neighbors"],
        help="Number of nearest neighbours for the KNN graph. "
             "Lowered to 12 (from 15) per Xenium benchmark recommendation — "
             "smaller neighbourhood better preserves fine spatial structure in targeted panels.",
    )
    st.session_state["n_neighbors"] = int(v)

# Cell area normalisation — separate row
st.markdown("**Cell area normalisation**")
ca1, ca2 = st.columns([1, 3])
with ca1:
    v = st.checkbox(
        "Normalise by cell area",
        value=st.session_state.get("normalize_by_cell_area", False),
        help="Divide log-normalised expression by cell cross-sectional area "
             "(from cells.parquet). Reduces the technical confound caused by "
             "variable cell sizes in brain tissue: neurons are dramatically larger "
             "than glia, and partially-captured surface cells appear artificially small. "
             "Requires cell_area column in cells.parquet (XOA 2.x+). Default off.",
    )
    st.session_state["normalize_by_cell_area"] = v
with ca2:
    if v:
        st.info(
            "Cell area normalisation is **enabled**. Expression values will be "
            "scaled by median_area / cell_area after log-normalisation, so cells "
            "larger than the median are scaled down and smaller cells are scaled up. "
            "Check PCA plots to confirm this reduces size-driven PC1 variance.",
            icon="⚠️",
        )
    else:
        st.caption(
            "Off by default. Enable if your PCA shows a strong size-correlated "
            "component that does not separate by cell type."
        )

st.divider()

# ── Integration ───────────────────────────────────────────────────────────────
st.subheader("Harmony batch correction")
st.markdown(
    "Harmony corrects for technical variation between slides. "
    "The batch key is always **slide_id** (not condition), "
    "so biological differences between AGED and ADULT are preserved."
)
h1, h2 = st.columns(2)
with h1:
    v = st.number_input(
        "Max Harmony iterations",
        5, 100, st.session_state["harmony_max_iter"],
    )
    st.session_state["harmony_max_iter"] = int(v)
with h2:
    st.info(
        "Batch key: **slide_id** (fixed)\n\n"
        "This ensures all 8 slides integrate correctly while "
        "AGED vs ADULT differences remain detectable in DGE."
    )

st.divider()

# ── DGE ───────────────────────────────────────────────────────────────────────
st.subheader("Differential gene expression")
d1, d2, d3 = st.columns(3)

with d1:
    _methods = ["stringent_wilcoxon", "wilcoxon", "pydeseq2"]
    _current = st.session_state["dge_method"]
    if _current not in _methods:
        _current = "stringent_wilcoxon"
    method = st.radio(
        "DGE method",
        _methods,
        index=_methods.index(_current),
        format_func=lambda x: {
            "stringent_wilcoxon": "★ Stringent Wilcoxon (recommended)",
            "wilcoxon"          : "Wilcoxon rank-sum (permissive)",
            "pydeseq2"          : "PyDESeq2 pseudobulk (n≥8 replicates needed)",
        }[x],
        help=(
            "Stringent Wilcoxon applies Wilcoxon then filters to: "
            "|log₂FC| ≥ 1.0, adj.p < 0.01, expressed in ≥10% of cells, "
            "and consistent direction in ≥3/4 biological replicates. "
            "This is the recommended approach for n=4 per condition."
        ),
    )
    st.session_state["dge_method"] = method

    if method == "stringent_wilcoxon":
        st.success(
            "★ **Stringent Wilcoxon** applies |log₂FC| ≥ 1.0, "
            "adj.p < 0.01, and requires the same direction of change in "
            "≥3/4 biological replicates. Statistically defensible for n=4."
        )
    elif method == "wilcoxon":
        st.warning(
            "⚠️ Plain Wilcoxon treats all cells as independent observations. "
            "With 92,000 cells, p-values are inflated. Consider Stringent Wilcoxon."
        )
    elif method == "pydeseq2":
        st.info(
            "ℹ️ PyDESeq2 is statistically correct but typically finds no "
            "significant genes at n=4 replicates per condition due to low power."
        )

with d2:
    v = st.number_input(
        "log₂FC threshold",
        0.1, 3.0, float(st.session_state["log2fc_threshold"]), 0.1,
        format="%.2f",
        help="Minimum absolute log2 fold-change for a gene to be called significant.",
    )
    st.session_state["log2fc_threshold"] = float(v)

with d3:
    v = st.number_input(
        "Adjusted p-value threshold",
        0.001, 0.1, float(st.session_state["pval_threshold"]), 0.005,
        format="%.3f",
    )
    st.session_state["pval_threshold"] = float(v)

st.divider()

# ── ROI & Figures ─────────────────────────────────────────────────────────────
st.subheader("ROI & figure export")
r1, r2, r3 = st.columns(3)

with r1:
    roi_mode = st.radio(
        "ROI drawing mode",
        ["polygon", "lasso", "rectangle"],
        index=["polygon","lasso","rectangle"].index(st.session_state["roi_mode"]),
        format_func=lambda x: {
            "polygon"   : "Polygon — click to place vertices",
            "lasso"     : "Lasso — freehand draw",
            "rectangle" : "Rectangle — click and drag",
        }[x],
    )
    st.session_state["roi_mode"] = roi_mode

with r2:
    fmt = st.selectbox(
        "Figure format",
        ["pdf", "png", "svg"],
        index=["pdf","png","svg"].index(st.session_state["figure_format"]),
        help="PDF = editable in Illustrator/Affinity. PNG = quick preview.",
    )
    st.session_state["figure_format"] = fmt

with r3:
    dpi = st.selectbox(
        "Figure DPI",
        [150, 300, 600],
        index=[150,300,600].index(st.session_state["dpi"]),
        help="300 is the Nature minimum for print.",
    )
    st.session_state["dpi"] = dpi

st.divider()
st.success(
    "✅ All settings auto-saved to your session. "
    "Go to **🚀 Run Pipeline** when ready."
)
