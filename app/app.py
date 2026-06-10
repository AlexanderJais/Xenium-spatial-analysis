"""
app.py
------
Xenium Sample PCA — streamlined local web interface.

Three steps only:
    1. Study Setup   — point to the 8 Xenium output directories
    2. ROI Manager   — frame the MBH region per slide
    3. Sample PCA    — pseudobulk PCA of the 8 samples (Nature-style)

Run with:  streamlit run app/app.py
"""

import streamlit as st
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from ui_utils import inject_css, page_header

st.set_page_config(
    page_title            = "Xenium Sample PCA",
    page_icon             = "🧠",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

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
    "base_panel_csv": str(Path(__file__).parent.parent / "data" / "Xenium_mBrain_v1_1_metadata.csv"),
    "output_dir"    : str(Path.home() / "xenium_sample_pca_output"),
    "roi_cache_dir" : str(Path(__file__).parent.parent / "roi_cache"),
    "panel_mode"    : "partial_union",
    "min_slides"    : 2,
    "roi_polygons"  : {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Derived state ────────────────────────────────────────────────────────────
def _slides_configured() -> int:
    return sum(1 for s in st.session_state["slides"]
               if s["run_dir"] and Path(s["run_dir"]).exists())

def _rois_saved() -> int:
    return len(st.session_state["roi_polygons"])

configured = _slides_configured()
n_slides   = len(st.session_state["slides"])
n_roi      = _rois_saved()
panel_ok   = Path(st.session_state["base_panel_csv"]).exists()
pca_done   = (Path(st.session_state["output_dir"]) / "sample_pca"
              / "sample_pca_scatter.pdf").exists()


def _current_step() -> int:
    if not configured:
        return 1
    if not n_roi:
        return 2
    return 3

current_step = _current_step()


def _step_state(step_n: int) -> str:
    done_map = {1: configured > 0, 2: n_roi > 0, 3: pca_done}
    if done_map.get(step_n):
        return "done"
    return "current" if step_n == current_step else "pending"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; text-align:center;">
        <div style="font-size:2rem; margin-bottom:0.25rem;">🧠</div>
        <div style="font-size:1rem; font-weight:600; color:#FFFFFF; letter-spacing:-0.01em;">
            Xenium Sample PCA
        </div>
        <div style="font-size:11px; color:rgba(255,255,255,0.55); margin-top:2px;">
            Pseudobulk PCA · AGED vs ADULT
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    STEP_LABELS = ["Study Setup", "ROI Manager", "Sample PCA"]
    steps_html = []
    for i, label in enumerate(STEP_LABELS, 1):
        state = _step_state(i)
        if state == "done":
            circle = ('<span style="min-width:20px;height:20px;background:#0A7E6E;color:#fff;'
                      'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
                      'font-size:10px;font-weight:700;flex-shrink:0;">✓</span>')
            text_style, label_html = "color:rgba(255,255,255,0.65);", label
        elif state == "current":
            circle = (f'<span style="min-width:20px;height:20px;background:#90C8F0;color:#0F2E52;'
                      f'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
                      f'font-size:10px;font-weight:700;flex-shrink:0;">{i}</span>')
            text_style, label_html = "color:#FFFFFF;font-weight:600;", f"<strong>{label}</strong>"
        else:
            circle = (f'<span style="min-width:20px;height:20px;background:rgba(255,255,255,0.12);'
                      f'color:rgba(255,255,255,0.4);border-radius:50%;display:inline-flex;'
                      f'align-items:center;justify-content:center;font-size:10px;font-weight:600;'
                      f'flex-shrink:0;">{i}</span>')
            text_style, label_html = "color:rgba(255,255,255,0.40);", label
        steps_html.append(
            f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;{text_style}">'
            f'{circle}<span style="font-size:12px;">{label_html}</span></div>'
        )
    st.markdown('<div style="padding:0.25rem 0;">' + "\n".join(steps_html) + "</div>",
                unsafe_allow_html=True)

    st.divider()

    def _ok(c: bool) -> str:
        return ('<span class="pill pill-ok">✓</span>' if c
                else '<span class="pill pill-missing">✗</span>')

    st.markdown(f"""
    <div style="font-size:11.5px; line-height:2.3;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>Base panel CSV</span> {_ok(panel_ok)}</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>Slides configured</span> {_ok(configured > 0)}</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>ROIs defined</span> {_ok(n_roi > 0)}</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>PCA complete</span> {_ok(pca_done)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute; bottom:1.5rem; left:1rem; right:1rem;
                font-size:10px; color:rgba(255,255,255,0.30); text-align:center;">
        Runs entirely on your local machine.<br>No data leaves this computer.
    </div>
    """, unsafe_allow_html=True)


# ── Home ────────────────────────────────────────────────────────────────────
page_header(
    "Xenium Sample PCA",
    "Pseudobulk PCA  ·  AGED vs ADULT mouse brain  ·  Mediobasal hypothalamus  ·  4 + 4 replicates",
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Slides configured", f"{configured} / {n_slides}")
with col2:
    st.metric("ROIs saved", f"{n_roi} / {n_slides}")
with col3:
    st.metric("Sample PCA", "ready" if pca_done else "—")

st.divider()

st.markdown("#### Workflow")
STEPS = [
    (1, "Study Setup", "Enter the path to each Xenium run folder; a green tick confirms it is valid. Save/load the full config as JSON."),
    (2, "ROI Manager", "Frame the mediobasal hypothalamus on each section with the interactive scatter; a dashed orange ellipse marks the atlas hint."),
    (3, "Sample PCA",  "Pseudobulk each slide and run PCA across the 8 samples — see how samples and the AGED/ADULT groups separate."),
]
items = []
for step_n, title, desc in STEPS:
    state = _step_state(step_n)
    if state == "done":
        num = ('<span style="min-width:26px;height:26px;background:#0A7E6E;color:white;'
               'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
               'font-size:12px;font-weight:700;flex-shrink:0;">✓</span>')
        row_style = "opacity:0.72;"
    elif state == "current":
        num = (f'<span style="min-width:26px;height:26px;background:#1B4F8A;color:white;'
               f'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
               f'font-size:12px;font-weight:700;flex-shrink:0;'
               f'box-shadow:0 0 0 3px rgba(27,79,138,0.22);">{step_n}</span>')
        row_style = "background:rgba(27,79,138,0.04);border-radius:6px;padding-left:0.4rem;padding-right:0.4rem;"
    else:
        num = (f'<span style="min-width:26px;height:26px;background:#D8DCE4;color:#8A95A3;'
               f'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
               f'font-size:12px;font-weight:600;flex-shrink:0;">{step_n}</span>')
        row_style = "opacity:0.50;"
    items.append(
        f'<li style="display:flex;align-items:flex-start;gap:0.85rem;padding:0.65rem 0;'
        f'border-bottom:1px solid #D8DCE4;{row_style}">{num}'
        f'<span style="font-size:13.5px;"><strong style="color:#0F2E52;">{title}</strong>'
        f' — {desc}</span></li>'
    )
if items:
    items[-1] = items[-1].replace("border-bottom:1px solid #D8DCE4;", "border-bottom:none;")
st.markdown('<ol style="list-style:none;padding:0;margin:0;">' + "\n".join(items) + "</ol>",
            unsafe_allow_html=True)

CTA_PAGES = {1: "pages/1_study_setup.py", 2: "pages/2_roi_manager.py", 3: "pages/3_sample_pca.py"}
CTA_LABELS = {1: "→ Configure slides", 2: "→ Draw ROIs", 3: "→ Run sample PCA"}
st.markdown("<br>", unsafe_allow_html=True)
try:
    st.page_link(CTA_PAGES[current_step], label=CTA_LABELS[current_step], icon="▶")
except (AttributeError, KeyError):
    st.info(f"Next step: **{STEP_LABELS[current_step - 1]}** — use the sidebar to navigate.")

st.divider()
st.markdown(
    '<p style="font-size:11.5px; color:#8A95A3; text-align:center;">'
    '🔒 &nbsp;All processing runs locally. No data is transmitted externally.</p>',
    unsafe_allow_html=True,
)
