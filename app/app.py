"""
app.py
------
Xenium DGE Pipeline — Local Web Interface
Run with:  streamlit run app/app.py
"""

import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title          = "Xenium DGE Pipeline",
    page_icon           = "🧠",
    layout              = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
# IBM Plex Sans for body, IBM Plex Mono for code/data, refined scientific palette
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Root tokens ─────────────────────────────────────────────────────────── */
:root {
    --navy:       #1B4F8A;
    --navy-dark:  #0F2E52;
    --navy-light: #E8EFF8;
    --teal:       #0A7E6E;
    --amber:      #C97A0A;
    --red:        #B02A2A;
    --bg:         #F5F6F8;
    --surface:    #FFFFFF;
    --border:     #D8DCE4;
    --text:       #0F1923;
    --muted:      #5A6474;
    --mono:       'IBM Plex Mono', monospace;
}

/* ── Typography ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px;
    color: var(--text);
}
h1 { font-size: 1.75rem !important; font-weight: 600 !important;
     color: var(--navy-dark) !important; letter-spacing: -0.02em; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.25rem !important; font-weight: 600 !important;
     color: var(--navy-dark) !important; letter-spacing: -0.01em; }
h3 { font-size: 1.05rem !important; font-weight: 500 !important;
     color: var(--navy) !important; }
h4 { font-size: 0.95rem !important; font-weight: 500 !important; color: var(--text) !important; }

/* ── App background ──────────────────────────────────────────────────────── */
.stApp { background: var(--bg) !important; }
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1300px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--navy-dark) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.90) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 0.75rem 0 !important;
}
[data-testid="stSidebar"] a { color: #90C8F0 !important; }
/* Active page nav highlight */
[data-testid="stSidebarNav"] li:has(a[aria-current="page"] ) {
    background: rgba(255,255,255,0.10) !important;
    border-left: 3px solid #90C8F0 !important;
    border-radius: 4px;
}
[data-testid="stSidebarNav"] a {
    padding: 0.45rem 0.75rem !important;
    border-radius: 4px !important;
    transition: background 0.15s;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.08) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
}
.stButton > button:hover {
    border-color: var(--navy) !important;
    color: var(--navy) !important;
    box-shadow: 0 2px 6px rgba(27,79,138,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: var(--navy) !important;
    color: #FFFFFF !important;
    border-color: var(--navy) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--navy-dark) !important;
    border-color: var(--navy-dark) !important;
    box-shadow: 0 3px 10px rgba(27,79,138,0.30) !important;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 600 !important;
    color: var(--navy-dark) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Alerts / callouts ───────────────────────────────────────────────────── */
.stAlert {
    border-radius: 6px !important;
    border-left-width: 3px !important;
    font-size: 13px !important;
}
[data-baseweb="notification"][kind="info"]    { border-left-color: var(--navy) !important; }
[data-baseweb="notification"][kind="success"] { border-left-color: var(--teal) !important; }
[data-baseweb="notification"][kind="warning"] { border-left-color: var(--amber) !important; }
[data-baseweb="notification"][kind="error"]   { border-left-color: var(--red) !important; }

/* ── Input widgets ───────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12.5px !important;
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    transition: border-color 0.15s;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 2px rgba(27,79,138,0.12) !important;
}
.stSelectbox > div > div,
.stRadio > div {
    font-size: 13px !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    padding: 0.6rem 1.1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--navy) !important; }
.stTabs [aria-selected="true"] {
    color: var(--navy) !important;
    border-bottom-color: var(--navy) !important;
    font-weight: 600 !important;
}

/* ── DataFrames / tables ─────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    font-size: 12px !important;
}
.stDataFrame thead th {
    background: var(--navy-light) !important;
    color: var(--navy-dark) !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
details summary {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--navy) !important;
}
details {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.8rem !important;
    background: var(--surface) !important;
    margin-bottom: 0.5rem !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

/* ── Code blocks ─────────────────────────────────────────────────────────── */
code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
.stCodeBlock {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--navy) !important;
    border-color: var(--navy) !important;
}

/* ── Spinner / progress ──────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--navy) !important; }

/* ── Success/warning/error chips in caption text ─────────────────────────── */
.stCaption { font-size: 11.5px !important; color: var(--muted) !important; }

/* ── Hide Streamlit footer and app hamburger menu ────────────────────────── */
/* Only hide the deploy toolbar and footer. Never touch sidebar controls.    */
footer { visibility: hidden !important; height: 0 !important; }
#MainMenu { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar collapsed-state toggle — force always visible ───────────────── */
/* Streamlit uses different test-ids across versions; cover all of them.     */
/* These rules ensure the chevron button to reopen the sidebar is ALWAYS     */
/* rendered and clickable, regardless of which version is installed.         */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[aria-label="Open sidebar"],
button[title="Open sidebar"],
section[data-testid="stSidebarCollapsedControl"] {
    display:    flex        !important;
    visibility: visible     !important;
    opacity:    1           !important;
    pointer-events: auto   !important;
    z-index: 999999         !important;
    background: var(--navy-dark) !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 3px 2px 10px rgba(0,0,0,0.22) !important;
    top: 1rem !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
button[aria-label="Open sidebar"] {
    color: rgba(255,255,255,0.90) !important;
    background: transparent !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover,
button[aria-label="Open sidebar"]:hover {
    color: #FFFFFF !important;
    background: rgba(255,255,255,0.12) !important;
}

/* ── Page header accent bar ──────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy) 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.page-header h1 { color: white !important; margin: 0 !important; font-size: 1.5rem !important; }
.page-header p  { color: rgba(255,255,255,0.75) !important; margin: 0.3rem 0 0 !important;
                  font-size: 13px !important; }

/* ── Stat card grid ──────────────────────────────────────────────────────── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-label {
    font-size: 10.5px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.25rem;
}
.stat-value {
    font-size: 1.6rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--navy-dark);
}
.stat-sub { font-size: 11px; color: var(--muted); margin-top: 0.15rem; }

/* ── Status pills ────────────────────────────────────────────────────────── */
.pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 500;
}
.pill-ok      { background: #E5F4F0; color: #0A7E6E; }
.pill-warn    { background: #FEF3E2; color: #C97A0A; }
.pill-missing { background: #F5E8E8; color: #B02A2A; }

/* ── Step list (home page) ───────────────────────────────────────────────── */
.step-list { counter-reset: steps; list-style: none; padding: 0; margin: 0; }
.step-list li {
    counter-increment: steps;
    display: flex; align-items: flex-start; gap: 0.85rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
}
.step-list li:last-child { border-bottom: none; }
.step-num {
    min-width: 26px; height: 26px;
    background: var(--navy);
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600; flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
}
.step-text strong { color: var(--navy-dark); }
</style>
""", unsafe_allow_html=True)

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
    "max_counts"          : 2000,
    "min_genes"           : 10,
    "max_genes"           : 300,
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; text-align:center;">
        <div style="font-size:2rem; margin-bottom:0.25rem;">🧠</div>
        <div style="font-size:1rem; font-weight:600; color:#FFFFFF; letter-spacing:-0.01em;">
            Xenium DGE
        </div>
        <div style="font-size:11px; color:rgba(255,255,255,0.55); margin-top:2px;">
            AGED vs ADULT · MBH
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Status block
    def _ok(cond: bool) -> str:
        return '<span class="pill pill-ok">✓ Ready</span>' if cond else \
               '<span class="pill pill-missing">✗ Missing</span>'

    slides_ok    = any(s["run_dir"] and Path(s["run_dir"]).exists()
                       for s in st.session_state["slides"])
    panel_ok     = Path(st.session_state["base_panel_csv"]).exists()
    rois_ok      = bool(st.session_state["roi_polygons"])
    out          = Path(st.session_state["output_dir"])
    results_ok   = out.exists() and len(list(out.glob("fig*.pdf"))) > 0

    st.markdown(f"""
    <div style="padding:0.5rem 0; font-size:12px; line-height:2.2;">
        <div>Base panel CSV &nbsp; {_ok(panel_ok)}</div>
        <div>Slides configured &nbsp; {_ok(slides_ok)}</div>
        <div>ROIs defined &nbsp; {_ok(rois_ok)}</div>
        <div>Results ready &nbsp; {_ok(results_ok)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Pipeline indicator
    if st.session_state.get("pipeline_running"):
        st.markdown("""
        <div style="background:rgba(255,255,255,0.08); border-radius:6px;
                    padding:0.6rem 0.8rem; font-size:12px; color:rgba(255,255,255,0.9);">
            ⏳ &nbsp;<strong>Pipeline running…</strong>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.get("pipeline_returncode") == 0:
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
st.markdown("""
<div class="page-header">
    <h1>Xenium DGE Pipeline</h1>
    <p>Spatial transcriptomics &nbsp;·&nbsp; AGED vs ADULT mouse brain
       &nbsp;·&nbsp; Mediobasal hypothalamus &nbsp;·&nbsp; 4 + 4 replicates</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
configured = sum(1 for s in st.session_state["slides"]
                 if s["run_dir"] and Path(s["run_dir"]).exists())
n_roi      = len(st.session_state["roi_polygons"])
n_figs     = len(list(out.glob("fig*.pdf"))) if out.exists() else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Slides configured", f"{configured} / 8")
with col2:
    st.metric("ROIs saved", f"{n_roi} / 8")
with col3:
    st.metric("Figures ready", n_figs)
with col4:
    log_path = out / "pipeline_run.log"
    log_sz   = f"{log_path.stat().st_size // 1024} KB" if log_path.exists() else "—"
    st.metric("Run log", log_sz)

st.divider()

# Two-column layout: steps + feature cards
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("#### Workflow")
    st.markdown("""
<ol class="step-list">
  <li><span class="step-num">1</span>
      <span class="step-text"><strong>Study Setup</strong> — add the path to each of the 8 Xenium run folders and confirm gene counts</span></li>
  <li><span class="step-num">2</span>
      <span class="step-text"><strong>Settings</strong> — review QC thresholds, Harmony parameters, and DGE method</span></li>
  <li><span class="step-num">3</span>
      <span class="step-text"><strong>ROI Manager</strong> — draw the MBH boundary on each section using the interactive scatter</span></li>
  <li><span class="step-num">4</span>
      <span class="step-text"><strong>Run Pipeline</strong> — launch and watch the live log; a timestamped log file is saved automatically</span></li>
  <li><span class="step-num">5</span>
      <span class="step-text"><strong>Results</strong> — browse all 13 Nature-grade figures inline and download PDFs, CSVs, and the final AnnData</span></li>
</ol>
""", unsafe_allow_html=True)

with right:
    st.markdown("#### Analysis summary")
    st.markdown("""
<div style="background:#FFFFFF; border:1px solid #D8DCE4; border-radius:8px;
            padding:1rem 1.2rem; font-size:13px; line-height:1.9;">
    <div style="display:flex; justify-content:space-between; border-bottom:1px solid #EEF0F3; padding-bottom:0.5rem; margin-bottom:0.5rem;">
        <span style="color:#5A6474; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Parameter</span>
        <span style="color:#5A6474; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Value</span>
    </div>
    <div style="display:flex; justify-content:space-between;"><span>Study design</span><strong>4 AGED + 4 ADULT</strong></div>
    <div style="display:flex; justify-content:space-between;"><span>Base panel</span><strong>247 genes (mBrain v1.1)</strong></div>
    <div style="display:flex; justify-content:space-between;"><span>Custom genes</span><strong>~50 / slide (partial union)</strong></div>
    <div style="display:flex; justify-content:space-between;"><span>ROI</span><strong>Mediobasal hypothalamus</strong></div>
    <div style="display:flex; justify-content:space-between;"><span>Batch correction</span><strong>Harmony (per slide)</strong></div>
    <div style="display:flex; justify-content:space-between;"><span>DGE method</span><strong>Stringent Wilcoxon</strong></div>
    <div style="display:flex; justify-content:space-between;"><span>Figures produced</span><strong>13 Nature-grade PDFs</strong></div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown(
    '<p style="font-size:11.5px; color:#8A95A3; text-align:center;">'
    '🔒 &nbsp;All processing runs locally. No data is transmitted externally.'
    '</p>',
    unsafe_allow_html=True,
)
