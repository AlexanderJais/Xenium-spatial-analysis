"""
app.py
------
Xenium DGE Pipeline — Local Web Interface
Run with:  streamlit run app/app.py
"""

import streamlit as st
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from ui_utils import inject_css, page_header

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title          = "Xenium DGE Pipeline",
    page_icon           = "🧠",
    layout              = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS (single source of truth — styles.css) ─────────────────────────
inject_css()

# ── Shared session-state defaults ───────────────────────────────────────────
DEFAULTS = {
    "slides": [
        {"slide_id": f"AGED_{i}",  "condition": "AGED",  "run_dir": ""}
        for i in range(1, 5)
    ] + [
        {"slide_id": f"ADULT_{i}", "condition": "ADULT", "run_dir": ""}
        for i in range(1, 5)
    ],
    "base_panel_csv"      : str(Path(__file__).parent / "data" / "Xenium_mBrain_v1_1_metadata.csv"),
    "output_dir"          : str(Path.home() / "xenium_dge_output"),
    "roi_cache_dir"       : str(Path(__file__).parent / "roi_cache"),
    "panel_mode"          : "partial_union",
    "min_slides"          : 2,
    "dge_method"          : "stringent_wilcoxon",
    "leiden_resolution"   : 0.6,
    "n_neighbors"         : 12,
    "min_counts"          : 10,
    "max_counts"          : 5000,
    "min_genes"           : 10,
    "max_genes"           : 500,
    "log2fc_threshold"    : 1.0,
    "pval_threshold"      : 0.01,
    "n_top_genes"         : 0,
    "filter_control_probes"   : True,
    "filter_control_codewords": True,
    "normalize_by_cell_area"  : False,
    "harmony_max_iter"    : 20,
    "roi_mode"            : "polygon",
    "figure_format"       : "pdf",
    "dpi"                 : 300,
    "pipeline_running"    : False,
    "pipeline_log"        : [],
    "pipeline_returncode" : None,
    "pipeline_proc"       : None,
    "pipeline_log_queue"  : None,
    "roi_polygons"        : {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Derived state helpers ────────────────────────────────────────────────────
def _slides_configured() -> int:
    return sum(1 for s in st.session_state["slides"]
               if s["run_dir"] and Path(s["run_dir"]).exists())

def _rois_saved() -> int:
    return len(st.session_state["roi_polygons"])

def _n_slides() -> int:
    return len(st.session_state["slides"])

out = Path(st.session_state["output_dir"])

def _n_figs() -> int:
    return len(list(out.glob("fig*.pdf"))) if out.exists() else 0

def _pipeline_done() -> bool:
    return st.session_state.get("pipeline_returncode") == 0

def _pipeline_running() -> bool:
    return bool(st.session_state.get("pipeline_running"))

# Compute which workflow step is "current"
# Steps: 1=setup, 2=settings, 3=roi, 4=run, 5=results
configured = _slides_configured()
n_slides   = _n_slides()
n_roi      = _rois_saved()
n_figs     = _n_figs()
panel_ok   = Path(st.session_state["base_panel_csv"]).exists()

def _current_step() -> int:
    if not configured:
        return 1
    if configured and not n_roi:
        return 3   # skip 2 (settings has sane defaults — not a blocker)
    if n_roi and not _pipeline_done():
        return 4
    return 5

current_step = _current_step()

def _step_state(step_n: int) -> str:
    """Return 'done', 'current', or 'pending'."""
    done_map = {1: configured > 0, 2: True, 3: n_roi > 0, 4: _pipeline_done(), 5: n_figs > 0}
    if done_map.get(step_n):
        return "done"
    if step_n == current_step:
        return "current"
    return "pending"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; text-align:center;">
        <div style="font-size:2rem; margin-bottom:0.25rem;">🧠</div>
        <div style="font-size:1rem; font-weight:600; color:#FFFFFF; letter-spacing:-0.01em;">
            Xenium DGE
        </div>
        <div style="font-size:11px; color:rgba(255,255,255,0.55); margin-top:2px;">
            Spatial transcriptomics pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Workflow progress in sidebar
    STEP_LABELS = [
        "Study Setup",
        "Settings",
        "ROI Manager",
        "Run Pipeline",
        "Results",
    ]
    sidebar_steps_html = []
    for i, label in enumerate(STEP_LABELS, 1):
        state = _step_state(i)
        if state == "done":
            circle = '<span style="min-width:20px;height:20px;background:#0A7E6E;color:#fff;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;">✓</span>'
            text_style = "color:rgba(255,255,255,0.65);"
            label_html = label
        elif state == "current":
            circle = f'<span style="min-width:20px;height:20px;background:#90C8F0;color:#0F2E52;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;">{i}</span>'
            text_style = "color:#FFFFFF;font-weight:600;"
            label_html = f"<strong>{label}</strong>"
        else:
            circle = f'<span style="min-width:20px;height:20px;background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.4);border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;flex-shrink:0;">{i}</span>'
            text_style = "color:rgba(255,255,255,0.40);"
            label_html = label
        sidebar_steps_html.append(
            f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;{text_style}">'
            f'{circle}'
            f'<span style="font-size:12px;">{label_html}</span>'
            f'</div>'
        )

    st.markdown(
        '<div style="padding:0.25rem 0;">'
        + "\n".join(sidebar_steps_html)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Quick status pills
    def _ok(cond: bool) -> str:
        return '<span class="pill pill-ok">✓</span>' if cond else \
               '<span class="pill pill-missing">✗</span>'

    slides_ok  = configured > 0
    rois_ok    = n_roi > 0
    results_ok = n_figs > 0

    st.markdown(f"""
    <div style="font-size:11.5px; line-height:2.3;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>Base panel CSV</span> {_ok(panel_ok)}
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>Slides configured</span> {_ok(slides_ok)}
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>ROIs defined</span> {_ok(rois_ok)}
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>Results ready</span> {_ok(results_ok)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Pipeline run state
    if _pipeline_running():
        st.markdown("""
        <div style="background:rgba(255,255,255,0.08); border-radius:6px;
                    padding:0.6rem 0.8rem; font-size:12px; color:rgba(255,255,255,0.9);">
            ⏳ &nbsp;<strong>Pipeline running…</strong>
        </div>
        """, unsafe_allow_html=True)
    elif _pipeline_done():
        st.markdown("""
        <div style="background:rgba(10,126,110,0.25); border-radius:6px;
                    padding:0.6rem 0.8rem; font-size:12px; color:#90E4D6;">
            ✅ &nbsp;<strong>Last run succeeded</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute; bottom:1.5rem; left:1rem; right:1rem;
                font-size:10px; color:rgba(255,255,255,0.30); text-align:center;">
        Runs entirely on your local machine.<br>No data leaves this computer.
    </div>
    """, unsafe_allow_html=True)


# ── Home page ─────────────────────────────────────────────────────────────────
page_header(
    "Xenium DGE Pipeline",
    "Spatial transcriptomics  ·  AGED vs ADULT mouse brain  ·  Mediobasal hypothalamus  ·  4 + 4 replicates",
)

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Slides configured", f"{configured} / {n_slides}")
with col2:
    st.metric("ROIs saved", f"{n_roi} / {n_slides}")
with col3:
    st.metric("Figures ready", n_figs)
with col4:
    log_path = out / "pipeline_run.log"
    log_sz   = f"{log_path.stat().st_size // 1024} KB" if log_path.exists() else "—"
    st.metric("Run log", log_sz)

st.divider()

# ── Main body: workflow steps + analysis summary ──────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("#### Workflow")

    STEPS = [
        (1, "Study Setup",   "Add the path to each Xenium run folder and confirm gene counts"),
        (2, "Settings",      "Review QC thresholds, Harmony parameters, and DGE method"),
        (3, "ROI Manager",   "Draw the MBH boundary on each section with the interactive scatter"),
        (4, "Run Pipeline",  "Launch and watch the live log; a timestamped log file is saved automatically"),
        (5, "Results",       "Browse all Nature-grade figures inline; download PDFs, CSVs, and the final AnnData"),
    ]

    step_items = []
    for step_n, title, desc in STEPS:
        state = _step_state(step_n)
        if state == "done":
            num_html = (
                '<span style="min-width:26px;height:26px;background:#0A7E6E;color:white;'
                'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
                'font-size:12px;font-weight:700;flex-shrink:0;">✓</span>'
            )
            row_style = "opacity:0.72;"
        elif state == "current":
            num_html = (
                f'<span style="min-width:26px;height:26px;background:#1B4F8A;color:white;'
                f'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
                f'font-size:12px;font-weight:700;flex-shrink:0;'
                f'box-shadow:0 0 0 3px rgba(27,79,138,0.22);">{step_n}</span>'
            )
            row_style = "background:rgba(27,79,138,0.04);border-radius:6px;padding-left:0.4rem;padding-right:0.4rem;"
        else:
            num_html = (
                f'<span style="min-width:26px;height:26px;background:#D8DCE4;color:#8A95A3;'
                f'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
                f'font-size:12px;font-weight:600;flex-shrink:0;">{step_n}</span>'
            )
            row_style = "opacity:0.50;"

        step_items.append(
            f'<li style="display:flex;align-items:flex-start;gap:0.85rem;'
            f'padding:0.65rem 0;border-bottom:1px solid #D8DCE4;{row_style}">'
            f'{num_html}'
            f'<span style="font-size:13.5px;">'
            f'<strong style="color:#0F2E52;">{title}</strong>'
            f' — {desc}'
            f'</span></li>'
        )
    # Remove last border
    if step_items:
        step_items[-1] = step_items[-1].replace("border-bottom:1px solid #D8DCE4;", "border-bottom:none;")

    st.markdown(
        '<ol style="list-style:none;padding:0;margin:0;">'
        + "\n".join(step_items)
        + "</ol>",
        unsafe_allow_html=True,
    )

    # CTA: link to the current step's page
    CTA_PAGES = {
        1: "pages/1_study_setup.py",
        2: "pages/2_settings.py",
        3: "pages/3_roi_manager.py",
        4: "pages/4_run_pipeline.py",
        5: "pages/5_results.py",
    }
    CTA_LABELS = {
        1: "→ Configure slides",
        2: "→ Review settings",
        3: "→ Draw ROIs",
        4: "→ Run pipeline",
        5: "→ View results",
    }
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        st.page_link(
            CTA_PAGES[current_step],
            label=CTA_LABELS[current_step],
            icon="▶",
        )
    except Exception:
        # st.page_link not available in older Streamlit versions
        st.info(f"Next step: **{STEP_LABELS[current_step - 1]}** — use the sidebar to navigate.")


with right:
    st.markdown("#### Analysis summary")

    # Dynamic values from session state
    dge_method_labels = {
        "stringent_wilcoxon": "Stringent Wilcoxon",
        "wilcoxon"          : "Wilcoxon rank-sum",
        "pydeseq2"          : "PyDESeq2 pseudobulk",
        "cside"             : "C-SIDE pseudobulk",
        "t-test"            : "t-test",
    }
    dge_label   = dge_method_labels.get(st.session_state["dge_method"], st.session_state["dge_method"])
    n_figs_disp = n_figs if n_figs > 0 else "16 (planned)"
    panel_mode_label = {
        "partial_union": "Partial union",
        "intersection" : "Intersection",
        "union"        : "Full union",
    }.get(st.session_state["panel_mode"], st.session_state["panel_mode"])

    # Unique conditions from configured slides
    conds = sorted({s["condition"] for s in st.session_state["slides"] if s["condition"]})
    study_design_label = " vs ".join(conds) if conds else "AGED vs ADULT"
    n_per_cond = n_slides // max(len(conds), 1)

    rows = [
        ("Study design",    f"{n_per_cond} {' + '.join(conds)} replicates"),
        ("Base panel",      "247 genes (mBrain v1.1)"),
        ("Panel mode",      panel_mode_label),
        ("ROI",             "Mediobasal hypothalamus"),
        ("Batch correction","Harmony (per slide)"),
        ("DGE method",      dge_label),
        ("Figures",         f"{n_figs_disp} Nature-grade PDFs"),
    ]

    rows_html = []
    for label, value in rows:
        rows_html.append(
            f'<div style="display:flex;justify-content:space-between;padding:0.28rem 0;'
            f'border-bottom:1px solid #F0F2F5;">'
            f'<span style="color:#5A6474;">{label}</span>'
            f'<strong style="color:#0F2E52;text-align:right;max-width:55%;">{value}</strong>'
            f'</div>'
        )
    if rows_html:
        rows_html[-1] = rows_html[-1].replace("border-bottom:1px solid #F0F2F5;", "border-bottom:none;")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #D8DCE4;border-radius:8px;'
        'padding:1rem 1.2rem;font-size:13px;">'
        + "\n".join(rows_html)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Show a brief note if pipeline succeeded
    if _pipeline_done():
        st.success(f"✅ Pipeline complete — {n_figs} figures saved to `{out}`")
    elif _pipeline_running():
        st.info("⏳ Pipeline is running…")

st.divider()
st.markdown(
    '<p style="font-size:11.5px; color:#8A95A3; text-align:center;">'
    '🔒 &nbsp;All processing runs locally. No data is transmitted externally.'
    '</p>',
    unsafe_allow_html=True,
)
