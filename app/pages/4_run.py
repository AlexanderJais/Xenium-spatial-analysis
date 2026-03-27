"""
pages/4_run.py
Run Pipeline page — pre-flight checks, launch, live log.
"""

import json
import queue
from datetime import datetime
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(page_title="Run Pipeline · Xenium DGE", page_icon="🚀", layout="wide",
    initial_sidebar_state="expanded")


inject_css()
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Defaults ─────────────────────────────────────────────────────────────────
for k, v in {
    "slides": [], "base_panel_csv": "", "output_dir": "",
    "roi_cache_dir": "", "panel_mode": "partial_union",
    "min_slides": 2, "dge_method": "stringent_wilcoxon",
    "leiden_resolution": 0.5, "n_neighbors": 15,
    "min_counts": 10, "max_counts": 2000,
    "min_genes": 5, "max_genes": 300,
    "log2fc_threshold": 1.0, "pval_threshold": 0.01,
    "n_top_genes": 0, "filter_control_probes": True,
    "filter_control_codewords": True, "normalize_by_cell_area": False, "harmony_max_iter": 20,
    "figure_format": "pdf", "dpi": 300,
    "roi_polygons": {},
    "pipeline_running": False,
    "pipeline_log": [],
    "pipeline_returncode": None,
    "pipeline_proc": None,
    "pipeline_log_queue": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────

def _xenium_dir_ok(path_str: str) -> bool:
    """Accept both new-format (cells.parquet) and old-format (cells.csv.gz) runs."""
    p = Path(path_str)
    if not (p.exists() and p.is_dir()):
        return False
    mtx = p / "cell_feature_matrix"
    if not mtx.exists():
        return False
    if not all((mtx / f).exists() for f in ["matrix.mtx.gz","barcodes.tsv.gz","features.tsv.gz"]):
        return False
    # Accept either cells.parquet (new) or cells.csv[.gz] (old)
    has_cells = (
        (p / "cells.parquet").exists()
        or (p / "cells.csv.gz").exists()
        or (p / "cells.csv").exists()
    )
    return has_cells


def _preflight() -> list[str]:
    """Return list of blocking errors."""
    errors = []

    slides = st.session_state["slides"]
    configured = [s for s in slides if s.get("run_dir") and _xenium_dir_ok(s["run_dir"])]
    if not configured:
        errors.append("No valid slide directories. Go to 📁 Study Setup.")
    else:
        aged  = [s for s in configured if s["condition"] == "AGED"]
        adult = [s for s in configured if s["condition"] == "ADULT"]
        if not aged:
            errors.append("No AGED slides configured.")
        if not adult:
            errors.append("No ADULT slides configured.")

    if not Path(st.session_state["base_panel_csv"]).exists():
        errors.append("Base panel CSV not found.")

    return errors


def _build_launcher_config() -> dict:
    """Assemble the JSON config consumed by run_xenium_mbh.py."""
    out_dir   = Path(st.session_state["output_dir"])
    cache_dir = out_dir.parent / (out_dir.name + "_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Persist ROI polygons to roi_cache/ before launching.
    # Include n_cells_selected by reading back the count stored by the ROI Manager
    # (saved under the key f"n_cells_{sid}" when the user clicks Save ROI).
    roi_cache = Path(st.session_state["roi_cache_dir"])
    roi_cache.mkdir(parents=True, exist_ok=True)
    for sid, verts in st.session_state["roi_polygons"].items():
        roi = {
            "slide_id"         : sid,
            "roi_name"         : "MBH",
            "vertices"         : verts,
            "n_cells_selected" : st.session_state.get(f"n_cells_{sid}"),
            "created_at"       : datetime.now().isoformat(),
            "method"           : "web_ui",
        }
        safe = sid.replace("/","_").replace(" ","_")
        (roi_cache / f"{safe}_roi.json").write_text(json.dumps(roi, indent=2))

    return {
        "slides"           : [s for s in st.session_state["slides"]
                               if s.get("run_dir") and _xenium_dir_ok(s["run_dir"])],
        "base_panel_csv"   : st.session_state["base_panel_csv"],
        "output_dir"       : str(out_dir),
        "roi_cache_dir"    : str(roi_cache),
        "panel_mode"       : st.session_state["panel_mode"],
        "min_slides"       : st.session_state["min_slides"],
        "dge_method"       : st.session_state["dge_method"],
        "leiden_resolution": st.session_state["leiden_resolution"],
        "n_neighbors"      : st.session_state["n_neighbors"],
        "min_counts"              : st.session_state["min_counts"],
        "max_counts"              : st.session_state["max_counts"],
        "min_genes"               : st.session_state["min_genes"],
        "max_genes"               : st.session_state["max_genes"],
        "log2fc_threshold"        : st.session_state["log2fc_threshold"],
        "pval_threshold"          : st.session_state["pval_threshold"],
        "n_top_genes"             : st.session_state["n_top_genes"],
        "filter_control_probes"   : st.session_state.get("filter_control_probes",    True),
        "filter_control_codewords": st.session_state.get("filter_control_codewords", True),
        "normalize_by_cell_area"  : st.session_state.get("normalize_by_cell_area",   False),
        "harmony_max_iter"        : st.session_state["harmony_max_iter"],
        "figure_format"           : st.session_state["figure_format"],
        "dpi"                     : st.session_state["dpi"],
        "no_roi_gui"              : True,   # always skip Qt GUI when running via web
    }


def _launch_pipeline(cfg: dict):
    cfg_path = Path(cfg["output_dir"]) / ".web_launcher_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    cmd = [
        sys.executable,
        str(_ROOT / "run_xenium_mbh.py"),
        "--launcher-config", str(cfg_path),
        "--no-roi-gui",
    ]

    st.session_state["pipeline_log"]        = []
    st.session_state["pipeline_returncode"] = None
    st.session_state["pipeline_running"]    = True

    log_q = queue.Queue()
    st.session_state["pipeline_log_queue"] = log_q

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        cwd=str(_ROOT),
    )
    st.session_state["pipeline_proc"] = proc

    def _stream():
        for line in proc.stdout:
            log_q.put(line.rstrip())
        ret = proc.wait()
        log_q.put(f"__DONE__{ret}")

    threading.Thread(target=_stream, daemon=True).start()


# ── Page ──────────────────────────────────────────────────────────────────────
page_header("🚀 Run Pipeline", "Pre-flight checks, pipeline launch, and live log")

# ── Pre-flight checks ─────────────────────────────────────────────────────────
st.subheader("Pre-flight checks")
errors = _preflight()
slides_ok = [s for s in st.session_state["slides"]
             if s.get("run_dir") and _xenium_dir_ok(s["run_dir"])]

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Valid slides", len(slides_ok))
with col_b:
    n_roi = len(st.session_state["roi_polygons"])
    st.metric("ROIs saved", n_roi)
with col_c:
    csv_ok = Path(st.session_state["base_panel_csv"]).exists()
    st.metric("Base panel", "✓ Found" if csv_ok else "✗ Missing")
with col_d:
    out = Path(st.session_state["output_dir"])
    n_figs = len(list(out.glob("fig*.pdf"))) if out.exists() else 0
    st.metric("Existing figures", n_figs)

if errors:
    for e in errors:
        st.error(f"❌ {e}")
    st.stop()

# ── Slide summary ─────────────────────────────────────────────────────────────
with st.expander("Slide summary", expanded=False):
    rows = []
    for s in slides_ok:
        sid   = s["slide_id"]
        has_r = sid in st.session_state["roi_polygons"]
        rows.append({
            "Slide ID" : sid,
            "Condition": s["condition"],
            "Path"     : s["run_dir"],
            "ROI"      : "✅" if has_r else "⬜ (will use full slide)",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Run / stop controls ───────────────────────────────────────────────────────
st.divider()

running = st.session_state["pipeline_running"]

col_run, col_stop, col_spacer = st.columns([1, 1, 4])
with col_run:
    run_clicked = st.button(
        "▶  Run Pipeline",
        disabled=running or bool(errors),
        type="primary",
        use_container_width=True,
    )
with col_stop:
    stop_clicked = st.button(
        "■  Stop",
        disabled=not running,
        use_container_width=True,
    )

if run_clicked and not running:
    cfg = _build_launcher_config()
    _launch_pipeline(cfg)
    st.rerun()

if stop_clicked and running:
    proc = st.session_state.get("pipeline_proc")
    if proc and proc.poll() is None:
        proc.terminate()
    st.session_state["pipeline_running"]    = False
    st.session_state["pipeline_returncode"] = -1
    st.warning("Pipeline stopped by user.")

# ── Live log ──────────────────────────────────────────────────────────────────
st.divider()

# ── Progress animation (shown while pipeline is running) ──────────────────────
PIPELINE_STAGES = [
    ("🏊", "Loading slides",      "Reads count matrices, cells.parquet, and experiment metadata"),
    ("🚴", "Preprocessing",       "QC filtering, normalisation, PCA, Harmony integration, UMAP"),
    ("🏃", "DGE analysis",        "Differential expression (stringent Wilcoxon or PyDESeq2)"),
    ("🥇", "Figures & outputs",   "Generating 13 Nature-grade figures and saving results"),
]

def _current_stage(log_lines: list[str]) -> int:
    """Infer the current pipeline stage from recent log output."""
    recent = " ".join(log_lines[-40:]).lower()
    if any(k in recent for k in ("step 4", "generating figures", "fig1", "fig2", "make_figures")):
        return 3
    if any(k in recent for k in ("step 3", "running dge", "wilcoxon", "deseq", "run_dge")):
        return 2
    if any(k in recent for k in ("step 2", "preprocessing", "normalise", "harmony", "leiden", "umap")):
        return 1
    if any(k in recent for k in ("step 1", "loading", "slide", "xenium", "cells", "harmonise")):
        return 0
    return 0

if running:
    stage_idx = _current_stage(st.session_state.get("pipeline_log", []))

    stage_html_parts = []
    for i, (icon, label, desc) in enumerate(PIPELINE_STAGES):
        if i < stage_idx:
            # Completed stage
            box_style = (
                "background:#E5F4F0;border:1px solid #A8DDD4;border-radius:10px;"
                "padding:0.9rem 1rem;text-align:center;opacity:0.75;"
            )
            icon_html = f'<div style="font-size:1.6rem;margin-bottom:4px;filter:grayscale(40%)">{icon}</div>'
            label_html = f'<div style="font-size:11px;font-weight:600;color:#0A7E6E;">✓ {label}</div>'
        elif i == stage_idx:
            # Active stage — pulsing border + bounce animation
            box_style = (
                "background:#EBF2FB;border:2px solid #1B4F8A;border-radius:10px;"
                "padding:0.9rem 1rem;text-align:center;"
                "animation:pulse-border 1.4s ease-in-out infinite;"
            )
            icon_html = (
                f'<div style="font-size:2rem;margin-bottom:4px;'
                f'display:inline-block;animation:bounce 0.7s ease-in-out infinite alternate;">'
                f'{icon}</div>'
            )
            label_html = (
                f'<div style="font-size:11.5px;font-weight:700;color:#1B4F8A;">{label}</div>'
                f'<div style="font-size:10px;color:#5A6474;margin-top:3px;">{desc}</div>'
            )
        else:
            # Upcoming stage
            box_style = (
                "background:#F5F6F8;border:1px solid #D8DCE4;border-radius:10px;"
                "padding:0.9rem 1rem;text-align:center;opacity:0.45;"
            )
            icon_html = f'<div style="font-size:1.6rem;margin-bottom:4px;">{icon}</div>'
            label_html = f'<div style="font-size:11px;font-weight:500;color:#5A6474;">{label}</div>'

        arrow = '<div style="font-size:1.1rem;color:#C0C8D4;align-self:center;">›</div>' \
                if i < len(PIPELINE_STAGES) - 1 else ""
        stage_html_parts.append(
            f'<div style="{box_style}">{icon_html}{label_html}</div>{arrow}'
        )

    elapsed_lines = [l for l in st.session_state.get("pipeline_log", []) if "elapsed" in l.lower()]
    elapsed_note  = ""
    n_log         = len(st.session_state.get("pipeline_log", []))

    st.markdown(f"""
<style>
@keyframes bounce {{
  from {{ transform: translateY(0px); }}
  to   {{ transform: translateY(-6px); }}
}}
@keyframes pulse-border {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(27,79,138,0.0); }}
  50%       {{ box-shadow: 0 0 0 5px rgba(27,79,138,0.18); }}
}}
</style>
<div style="background:#FFFFFF;border:1px solid #D8DCE4;border-radius:12px;
            padding:1.2rem 1.5rem 1rem;margin-bottom:1rem;">
  <div style="display:flex;align-items:stretch;gap:0.6rem;justify-content:center;
              flex-wrap:nowrap;">
    {"".join(stage_html_parts)}
  </div>
  <div style="margin-top:0.9rem;text-align:center;">
    <span style="font-size:11px;color:#8A95A3;">
      ⏳ &nbsp;Pipeline running &nbsp;·&nbsp; {n_log:,} log lines captured
      &nbsp;·&nbsp; Refreshing every 1.5 s
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

st.subheader("Pipeline log")

log_placeholder = st.empty()

# Drain the queue each rerun
log_q = st.session_state.get("pipeline_log_queue")
if log_q:
    finished = False
    try:
        while True:
            line = log_q.get_nowait()
            if line.startswith("__DONE__"):
                ret = int(line.replace("__DONE__",""))
                st.session_state["pipeline_returncode"] = ret
                st.session_state["pipeline_running"]    = False
                st.session_state["pipeline_log"].append(
                    f"{'✅ Pipeline completed successfully.' if ret==0 else f'❌ Pipeline exited with code {ret}.'}"
                )
                finished = True
                break
            else:
                st.session_state["pipeline_log"].append(line)
    except queue.Empty:
        pass

    if running and not finished:
        # Auto-refresh every 1.5 s while running
        time.sleep(1.5)
        st.rerun()

# Render log
log_lines = st.session_state["pipeline_log"]

if log_lines:
    # Colour-code lines
    html_lines = []
    for line in log_lines[-300:]:   # keep last 300 lines
        lc = line.lower()
        if "error" in lc or "fail" in lc or "❌" in lc:
            colour = "#F44747"
        elif "warn" in lc:
            colour = "#CE9178"
        elif "✅" in line or "complete" in lc or "saved" in lc:
            colour = "#4EC9B0"
        elif line.startswith("=") or line.startswith("-"):
            colour = "#6A9955"
        else:
            colour = "#D4D4D4"
        escaped = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        html_lines.append(f'<span style="color:{colour}">{escaped}</span>')

    log_html = "\n".join(html_lines)
    log_placeholder.markdown(
        f'<div style="background:#1E1E1E;padding:14px;border-radius:6px;'
        f'font-family:\'SF Mono\',monospace;font-size:12px;line-height:1.55;'
        f'height:420px;overflow-y:auto;white-space:pre">{log_html}</div>',
        unsafe_allow_html=True,
    )
else:
    log_placeholder.markdown(
        '<div style="background:#1E1E1E;padding:14px;border-radius:6px;'
        'color:#555;font-family:monospace;font-size:12px;height:100px;'
        'display:flex;align-items:center;justify-content:center">'
        'Log will appear here when the pipeline starts.</div>',
        unsafe_allow_html=True,
    )

# ── Completion message ────────────────────────────────────────────────────────
ret = st.session_state.get("pipeline_returncode")
if ret is not None:
    if ret == 0:
        st.success(
            "✅ Pipeline finished! "
            "Go to **📊 Results** to view your figures."
        )
        st.balloons()
        # Offer log file download
        _log_path = Path(st.session_state.get("output_dir", "")) / "pipeline_run.log"
        if _log_path.exists():
            with open(_log_path, "rb") as _lf:
                st.download_button(
                    "⬇️ Download run log (pipeline_run.log)",
                    data=_lf.read(),
                    file_name="pipeline_run.log",
                    mime="text/plain",
                    use_container_width=False,
                )
    elif ret == -1:
        st.warning("Pipeline was stopped.")
    else:
        st.error(f"Pipeline exited with code {ret}. Check the log for errors.")
        _log_path = Path(st.session_state.get("output_dir", "")) / "pipeline_run.log"
        if _log_path.exists():
            with open(_log_path, "rb") as _lf:
                st.download_button(
                    "⬇️ Download run log (for diagnostics)",
                    data=_lf.read(),
                    file_name="pipeline_run.log",
                    mime="text/plain",
                )
