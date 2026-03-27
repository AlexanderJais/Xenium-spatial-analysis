"""
pages/1_study_setup.py
Study Setup page — configure the 8 slide folders.
"""

import json

import pandas as pd
import streamlit as st
from pathlib import Path

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(page_title="Study Setup · Xenium DGE", page_icon="📁", layout="wide",
    initial_sidebar_state="expanded")


inject_css()
# ── Shared defaults (duplicated here so pages work standalone) ───────────────
if "slides" not in st.session_state:
    st.session_state["slides"] = [
        {"slide_id": f"AGED_{i}",  "condition": "AGED",  "run_dir": ""} for i in range(1,5)
    ] + [
        {"slide_id": f"ADULT_{i}", "condition": "ADULT", "run_dir": ""} for i in range(1,5)
    ]
for k, v in {
    "base_panel_csv": str(Path(__file__).parent.parent / "data" / "Xenium_mBrain_v1_1_metadata.csv"),
    "output_dir"    : str(Path.home() / "xenium_dge_output"),
    "roi_cache_dir" : str(Path(__file__).parent.parent / "roi_cache"),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ──────────────────────────────────────────────────────────────────
CONDITION_COLOURS = {"AGED": "#D55E00", "ADULT": "#0072B2"}

def _xenium_dir_status(path_str: str) -> tuple[bool, str]:
    """Return (valid, message) for a Xenium run directory.

    Supports both old-format runs (no stain, imported segmentation, cells.csv.gz
    instead of cells.parquet) and new-format runs (xenium_cell_segmentation_stains_v1
    with cells.parquet).  The matrix files are the only strict requirement.
    """
    if not path_str.strip():
        return False, "No path entered"
    p = Path(path_str)
    if not p.exists():
        return False, f"Directory not found: {p}"
    if not p.is_dir():
        return False, "Path is not a directory"
    mtx = p / "cell_feature_matrix"
    if not mtx.exists():
        return False, "Missing cell_feature_matrix/"
    for f in ["matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"]:
        if not (mtx / f).exists():
            return False, f"Missing {f}"
    # cells.parquet is present in newer runs (xenium_cell_segmentation_stains_v1).
    # Older runs (imported segmentation) may have cells.csv.gz or similar instead.
    # We warn but do not block -- xenium_loader handles the fallback.
    has_cells_parquet = (p / "cells.parquet").exists()
    has_cells_csv     = (p / "cells.csv.gz").exists() or (p / "cells.csv").exists()
    if not has_cells_parquet and not has_cells_csv:
        return True, "Valid (no cells.parquet — spatial coords may be unavailable)"
    return True, "Valid Xenium run directory"

# ── Page ─────────────────────────────────────────────────────────────────────
page_header("📁 Study Setup", "Configure Xenium run directories for all 8 slides")
st.markdown(
    "Enter the path to each Xenium output directory. "
    "Each folder must contain `cell_feature_matrix/`, `cells.parquet`, "
    "and `experiment.xenium`."
)
st.info(
    "💡 **Tip:** On macOS, right-click a folder in Finder → "
    "**Get Info** → copy the path from *Where*, "
    "or drag the folder into this browser window's address bar to get its path."
)
st.divider()

# ── Slide table ──────────────────────────────────────────────────────────────
st.subheader("Slide folders")

slides = st.session_state["slides"]

for i, slide in enumerate(slides):
    cond   = slide["condition"]
    colour = CONDITION_COLOURS.get(cond, "#888")
    badge  = f'<span style="background:{colour};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold">{cond}</span>'

    col_badge, col_id, col_path, col_status = st.columns([1, 1.5, 5, 1.5])

    with col_badge:
        st.markdown(badge + "<br>", unsafe_allow_html=True)

    with col_id:
        new_id = st.text_input(
            "Slide ID",
            value=slide["slide_id"],
            key=f"slide_id_{i}",
            label_visibility="collapsed",
        )
        slides[i]["slide_id"] = new_id

    with col_path:
        new_path = st.text_input(
            "Run directory",
            value=slide["run_dir"],
            placeholder=f"/path/to/xenium_run_{cond.lower()}_{i%4+1}",
            key=f"run_dir_{i}",
            label_visibility="collapsed",
        )
        slides[i]["run_dir"] = new_path

    with col_status:
        if new_path.strip():
            ok, msg = _xenium_dir_status(new_path)
            if ok:
                st.success("✓", icon=None)
            else:
                st.error(msg[:40])
        else:
            st.caption("—")

    # Show gene count if valid
    if slide["run_dir"].strip():
        ok, _ = _xenium_dir_status(slide["run_dir"])
        if ok:
            try:
                feat_path = (
                    Path(slide["run_dir"]) / "cell_feature_matrix" / "features.tsv.gz"
                )
                # features.tsv.gz columns: gene_id, gene_name, feature_type
                # feature_type values: "Gene Expression", "Blank Codeword",
                #   "Negative Control Codeword", "Negative Control Probe"
                feats = pd.read_csv(
                    feat_path, sep="\t", header=None,
                    names=["gene_id", "gene_name", "feature_type"],
                    compression="gzip",
                )
                type_counts = feats["feature_type"].value_counts()
                n_rna       = int(type_counts.get("Gene Expression", 0))
                n_blank     = int(type_counts.get("Blank Codeword", 0))
                n_neg_cw    = int(type_counts.get("Negative Control Codeword", 0))
                n_neg_probe = int(type_counts.get("Negative Control Probe", 0))
                # Derive base panel count from the CSV rather than hardcoding 247.
                # Fall back to 247 only if the CSV cannot be read.
                try:
                    _base_csv = Path(st.session_state.get("base_panel_csv", ""))
                    n_predesigned = len(pd.read_csv(_base_csv)) if _base_csv.exists() else 247
                except Exception:
                    n_predesigned = 247
                n_custom      = max(0, n_rna - n_predesigned)

                bc_path = (
                    Path(slide["run_dir"]) / "cell_feature_matrix" / "barcodes.tsv.gz"
                )
                n_cells = len(pd.read_csv(bc_path, header=None, compression="gzip"))

                control_parts = []
                if n_blank > 0:
                    control_parts.append(f"{n_blank} blank codewords")
                if n_neg_cw > 0:
                    control_parts.append(f"{n_neg_cw} neg. control codewords")
                if n_neg_probe > 0:
                    control_parts.append(f"{n_neg_probe} neg. control probes")
                control_str = (
                    f" + {', '.join(control_parts)}" if control_parts else ""
                )

                st.caption(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {n_cells:,} cells · "
                    f"{n_rna} RNA targets "
                    f"({n_predesigned} predesigned + {n_custom} custom)"
                    f"{control_str}",
                    unsafe_allow_html=True,
                )
            except Exception as _e:
                st.caption(f"Could not read gene counts: {_e}")

    if i == 3:  # separator between AGED and ADULT
        st.divider()

st.session_state["slides"] = slides

# ── Summary banner ────────────────────────────────────────────────────────────
st.divider()
n_ok = sum(1 for s in slides if _xenium_dir_status(s["run_dir"])[0])
n_aged  = sum(1 for s in slides if s["condition"] == "AGED"  and _xenium_dir_status(s["run_dir"])[0])
n_adult = sum(1 for s in slides if s["condition"] == "ADULT" and _xenium_dir_status(s["run_dir"])[0])

if n_ok == 8:
    st.success(f"✅ All 8 slides configured ({n_aged} AGED, {n_adult} ADULT)")
elif n_ok > 0:
    st.warning(f"⚠️ {n_ok}/8 slides configured — {8-n_ok} still need paths")
else:
    st.error("No valid slide directories entered yet")

# ── File paths ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("File paths")

col_a, col_b = st.columns(2)

with col_a:
    csv = st.text_input(
        "Base panel CSV (Xenium_mBrain_v1_1_metadata.csv)",
        value=st.session_state["base_panel_csv"],
        help="The 10x Genomics metadata CSV that defines the 247 base panel genes.",
    )
    st.session_state["base_panel_csv"] = csv
    if Path(csv).exists():
        try:
            n = len(pd.read_csv(csv))
            st.caption(f"✓ Found — {n} genes")
        except Exception as e:
            st.error(str(e))
    else:
        st.error("File not found")

    roi_dir = st.text_input(
        "ROI cache directory",
        value=st.session_state["roi_cache_dir"],
        help="Polygon ROIs for each slide are stored here as JSON files.",
    )
    st.session_state["roi_cache_dir"] = roi_dir
    roi_path = Path(roi_dir)
    if roi_path.exists():
        n_saved = len(list(roi_path.glob("*_roi.json")))
        st.caption(f"{n_saved} ROI file(s) saved")
    else:
        st.caption("Directory will be created when the pipeline runs.")

with col_b:
    out = st.text_input(
        "Output directory",
        value=st.session_state["output_dir"],
        help="All figures and results files are written here.",
    )
    st.session_state["output_dir"] = out
    st.caption(f"Will be created if it does not exist: {out}")

# ── Save / load config ────────────────────────────────────────────────────────
st.divider()
st.subheader("Save / load configuration")

col_save, col_load = st.columns(2)

with col_save:
    if st.button("💾 Save configuration to JSON", use_container_width=True):
        cfg = {
            "slides"        : st.session_state["slides"],
            "base_panel_csv": st.session_state["base_panel_csv"],
            "output_dir"    : st.session_state["output_dir"],
            "roi_cache_dir" : st.session_state["roi_cache_dir"],
        }
        cfg_str = json.dumps(cfg, indent=2)
        st.download_button(
            "⬇️ Download config.json",
            data=cfg_str,
            file_name="xenium_pipeline_config.json",
            mime="application/json",
            use_container_width=True,
        )

with col_load:
    uploaded = st.file_uploader(
        "📂 Load configuration from JSON", type="json"
    )
    if uploaded:
        try:
            cfg = json.load(uploaded)
            if "slides" in cfg:
                st.session_state["slides"] = cfg["slides"]
            for k in ["base_panel_csv", "output_dir", "roi_cache_dir"]:
                if k in cfg:
                    st.session_state[k] = cfg[k]
            st.success("Configuration loaded — refresh the page to see updated paths.")
        except Exception as e:
            st.error(f"Could not load config: {e}")
