"""
pages/5_results.py
Results page — view all figures inline and download outputs.
"""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(page_title="Results · Xenium DGE", page_icon="📊", layout="wide",
    initial_sidebar_state="expanded")


inject_css()
if "output_dir" not in st.session_state:
    st.session_state["output_dir"] = str(Path.home() / "xenium_dge_output")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _pdf_viewer(path: Path, height: int = 600):
    """Embed a PDF inline using an <object> tag."""
    b64 = _b64(path)
    st.markdown(
        f'<object data="data:application/pdf;base64,{b64}" '
        f'type="application/pdf" width="100%" height="{height}px">'
        f'<p>PDF cannot be displayed inline. '
        f'<a href="data:application/pdf;base64,{b64}" download="{path.name}">Download</a></p>'
        f'</object>',
        unsafe_allow_html=True,
    )


def _png_viewer(path: Path):
    st.image(str(path), use_container_width=True)


def _download_button(path: Path, label: str):
    with open(path, "rb") as f:
        data = f.read()
    mime = {
        ".pdf" : "application/pdf",
        ".png" : "image/png",
        ".svg" : "image/svg+xml",
        ".csv" : "text/csv",
        ".h5ad": "application/octet-stream",
    }.get(path.suffix, "application/octet-stream")
    st.download_button(label, data=data, file_name=path.name, mime=mime,
                       use_container_width=True)


FIGURE_CATALOG = [
    ("fig1_qc",              "QC — counts, genes, spatial density"),
    ("fig2_umap",            "UMAP — condition & cluster"),
    ("fig3_spatial_clusters","Spatial cluster maps (MBH)"),
    ("fig4_dotplot",         "Marker gene dot plot"),
    ("fig5_volcano",         "AGED vs ADULT volcano"),
    ("fig6_heatmap",         "Top DEG heatmap (z-scored)"),
    ("fig7_spatial_expr",    "Spatial expression of top DEGs"),
    ("fig8_summary",         "6-panel composite summary"),
    ("fig9_cell_types",      "Cell type annotation"),
    ("fig10_spatial_stats",  "Moran's I + neighbourhood enrichment"),
    ("fig11_cluster_dge",    "Per-cluster DEG bubble chart"),
    ("fig12_slide_qc",       "Per-slide QC + MBH yield"),
    ("fig13_panel_qc",       "Panel composition + custom gene overlap"),
    ("fig14_insulin",        "Insulin pathway spatial expression"),
    ("fig15_galanin",        "Galanin co-expression network"),
    ("fig16_composition",    "Cell type composition (scCODA)"),
]

# ── Page ──────────────────────────────────────────────────────────────────────
page_header("📊 Results", "Browse figures, download DGE tables, and access the final AnnData")

out = Path(st.session_state["output_dir"])
fmt = st.session_state.get("figure_format", "pdf")

if not out.exists():
    st.info(f"Output directory does not exist yet: `{out}`\n\nRun the pipeline first.")
    st.stop()

# ── Figures ───────────────────────────────────────────────────────────────────
available = []
for stem, desc in FIGURE_CATALOG:
    for ext in [fmt, "pdf", "png"]:
        p = out / f"{stem}.{ext}"
        if p.exists():
            available.append((stem, desc, p))
            break

st.subheader(f"Figures ({len(available)} / {len(FIGURE_CATALOG)} available)")

if not available:
    st.warning("No figures found yet. Run the pipeline first.")
else:
    # Figure selector
    fig_names = [desc for _, desc, _ in available]
    selected_fig = st.selectbox(
        "Select figure to view",
        options=range(len(available)),
        format_func=lambda i: f"{'Fig ' + str(i+1):8s} — {fig_names[i]}",
    )

    stem, desc, fig_path = available[selected_fig]

    col_view, col_dl = st.columns([5, 1])
    with col_view:
        st.markdown(f"**{desc}**")
        if fig_path.suffix == ".pdf":
            _pdf_viewer(fig_path, height=620)
        else:
            _png_viewer(fig_path)
    with col_dl:
        st.markdown("&nbsp;")
        _download_button(fig_path, f"⬇️ Download {fig_path.name}")

    # Thumbnail gallery
    st.divider()
    st.markdown("#### All figures")
    cols = st.columns(4)
    for idx, (stem, desc, path) in enumerate(available):
        with cols[idx % 4]:
            if path.suffix in [".png", ".jpg"]:
                st.image(str(path), caption=f"Fig {idx+1}: {desc[:30]}", use_container_width=True)
            else:
                if st.button(f"Fig {idx+1}", key=f"thumb_{idx}", use_container_width=True):
                    st.session_state["selected_fig_idx"] = idx
                    st.rerun()
                st.caption(desc[:35])

# ── Data tables ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Data files")

tab_dge, tab_cluster, tab_morans, tab_panel, tab_adata, tab_log = st.tabs([
    "Global DGE", "Cluster DGE", "Moran's I", "Panel validation", "AnnData", "Run log"
])

with tab_dge:
    csv_path = out / "global_dge_aged_vs_adult.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # run_dge() always writes canonical column names: 'log2fc' and 'pval_adj'
        lfc_col = "log2fc"
        p_col   = "pval_adj"
        g_col   = "gene" if "gene" in df.columns else df.columns[0]
        if lfc_col not in df.columns or p_col not in df.columns:
            st.error(f"DGE CSV is missing expected columns (`{lfc_col}`, `{p_col}`). Found: {list(df.columns)}")
            st.stop()

        _p_thr  = float(st.session_state.get("pval_threshold",  0.01))
        _lfc_thr = float(st.session_state.get("log2fc_threshold", 1.0))
        n_sig = int(((df[p_col] < _p_thr) & (df[lfc_col].abs() > _lfc_thr)).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total genes tested", len(df))
        c2.metric(f"Significant (p<{_p_thr}, |log₂FC|>{_lfc_thr})", n_sig)
        c3.metric("Up in AGED", int(((df[p_col] < _p_thr) & (df[lfc_col] > _lfc_thr)).sum()))

        # Searchable table
        search = st.text_input("Search gene", key="dge_search")
        if search:
            df_show = df[df[g_col].str.contains(search, case=False, na=False)]
        else:
            df_show = df[df[p_col] < _p_thr].sort_values(p_col).head(200)

        st.dataframe(
            df_show.style.background_gradient(subset=[lfc_col], cmap="RdBu_r"),
            use_container_width=True,
            height=400,
        )
        _download_button(csv_path, "⬇️ Download full DGE table (CSV)")
    else:
        st.info("No DGE results yet.")

with tab_cluster:
    csv_path = out / "cluster_dge_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Summary pivot
        if "group" in df.columns and "direction" in df.columns:
            summary = (
                df.groupby("group", observed=True)["direction"]
                .value_counts().unstack(fill_value=0)
                .reset_index()
            )
            st.dataframe(summary, use_container_width=True)
            st.divider()

        group_filter = st.selectbox(
            "Filter by group",
            ["All"] + sorted(df["group"].unique().tolist()) if "group" in df.columns else ["All"],
        )
        df_show = df if group_filter == "All" else df[df["group"] == group_filter]
        sig_only = st.checkbox("Significant only", value=True)
        if sig_only and "significant" in df_show.columns:
            df_show = df_show[df_show["significant"]]
        st.dataframe(df_show.head(500), use_container_width=True, height=380)
        _download_button(csv_path, "⬇️ Download cluster DGE (CSV)")
    else:
        st.info("No cluster DGE results yet.")

with tab_morans:
    csv_path = out / "morans_i_mbh.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.markdown(
            f"**{len(df)} genes tested** for spatial autocorrelation (Moran's I). "
            f"Genes with I > 0 and adj-p < 0.05 are spatially patterned."
        )
        p_col_m = "p_adj" if "p_adj" in df.columns else "p_value"
        df_show = df[df[p_col_m] < 0.05].sort_values("morans_i", ascending=False)
        st.dataframe(df_show.head(100), use_container_width=True, height=380)
        _download_button(csv_path, "⬇️ Download Moran's I (CSV)")
    else:
        st.info("No Moran's I results yet.")

with tab_panel:
    csv_path = out / "panel_validation.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
        _download_button(csv_path, "⬇️ Download panel validation (CSV)")
    else:
        # Try to build it from uns info
        st.info(
            "Panel validation CSV not found. "
            "It is generated automatically during loading. "
            "Run the pipeline to produce it."
        )

with tab_adata:
    h5ad_path = out / "adata_mbh_final.h5ad"
    if h5ad_path.exists():
        size_mb = h5ad_path.stat().st_size / 1e6
        st.success(
            f"✅ Final AnnData found — **{size_mb:.1f} MB**\n\n"
            f"`{h5ad_path}`"
        )
        st.markdown(
            "Load in Python:\n"
            "```python\n"
            "import anndata as ad\n"
            f"adata = ad.read_h5ad('{h5ad_path}')\n"
            "```"
        )
        _download_button(h5ad_path, "⬇️ Download adata_mbh_final.h5ad")
    else:
        st.info("AnnData not found yet. Run the pipeline first.")

with tab_log:
    log_path = out / "pipeline_run.log"
    if log_path.exists():
        log_bytes = log_path.read_bytes()
        log_text  = log_bytes.decode("utf-8", errors="replace")
        log_lines = log_text.splitlines()

        size_kb = log_path.stat().st_size // 1024
        n_errors   = sum(1 for l in log_lines if " ERROR " in l or " CRITICAL " in l)
        n_warnings = sum(1 for l in log_lines if " WARNING " in l)

        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Log size",    f"{size_kb} KB")
        mc2.metric("Total lines", f"{len(log_lines):,}")
        mc3.metric("Warnings",    n_warnings,
                   delta=f"{n_warnings} ⚠️" if n_warnings else None,
                   delta_color="off")
        mc4.metric("Errors",      n_errors,
                   delta=f"{n_errors} ✗" if n_errors else None,
                   delta_color="inverse" if n_errors else "off")

        st.divider()

        # Filter controls
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            level_filter = st.selectbox(
                "Filter by level",
                ["All", "ERROR / CRITICAL", "WARNING", "INFO"],
                key="log_level_filter",
            )
        with search_col:
            log_search = st.text_input(
                "Search log text",
                placeholder="e.g. Harmony, DGE, cluster …",
                key="log_search",
            )

        # Apply filters
        filtered = log_lines
        if level_filter == "ERROR / CRITICAL":
            filtered = [l for l in filtered if " ERROR " in l or " CRITICAL " in l]
        elif level_filter == "WARNING":
            filtered = [l for l in filtered if " WARNING " in l]
        elif level_filter == "INFO":
            filtered = [l for l in filtered if " INFO " in l]
        if log_search:
            filtered = [l for l in filtered if log_search.lower() in l.lower()]

        # Colour-coded log viewer (last 500 lines to keep the DOM manageable)
        SHOW_TAIL = 500
        display_lines = filtered[-SHOW_TAIL:] if len(filtered) > SHOW_TAIL else filtered
        if len(filtered) > SHOW_TAIL:
            st.caption(f"Showing last {SHOW_TAIL} of {len(filtered):,} matching lines.")

        def _log_colour(line: str) -> str:
            ll = line.lower()
            if " error " in ll or " critical " in ll:
                return "#F44747"
            if " warning " in ll:
                return "#CE9178"
            if "====" in line or "----" in line:
                return "#6A9955"
            if " info " in ll and ("saved" in ll or "complete" in ll or "done" in ll or "✅" in line):
                return "#4EC9B0"
            return "#D4D4D4"

        html_lines = []
        for line in display_lines:
            colour  = _log_colour(line)
            escaped = (line.replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;"))
            html_lines.append(
                f'<span style="color:{colour};display:block;'
                f'white-space:pre;line-height:1.55">{escaped}</span>'
            )

        st.markdown(
            f'<div style="background:#1E1E1E; padding:14px 16px; border-radius:8px; '
            f'font-family:\'IBM Plex Mono\',monospace; font-size:11.5px; '
            f'height:480px; overflow-y:auto; border:1px solid #333;">'
            + "\n".join(html_lines)
            + "</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        _download_button(log_path, "⬇️ Download pipeline_run.log")

        # Session-start anchors for easy navigation
        sessions = [(i, l) for i, l in enumerate(log_lines) if "SESSION START" in l]
        if len(sessions) > 1:
            with st.expander(f"📋 {len(sessions)} pipeline runs in this log file"):
                for idx, (lineno, line) in enumerate(sessions):
                    # Extract timestamp from the marker line
                    ts = line.strip().replace("=", "").replace("SESSION START", "").strip()
                    st.caption(f"Run {idx+1}: line {lineno+1}  —  {ts}")
    else:
        st.info(
            "No log file found yet. "
            "`pipeline_run.log` is created automatically in your output directory "
            "when the pipeline runs."
        )

# ── Output directory browser ─────────────────────────────────────────────────
st.divider()
with st.expander("All output files"):
    all_files = sorted(out.glob("*")) if out.exists() else []
    for f in all_files:
        size = f.stat().st_size
        sz_str = (f"{size/1e6:.1f} MB" if size > 1e6
                  else f"{size/1e3:.0f} KB" if size > 1e3
                  else f"{size} B")
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"`{f.name}`  <span style='color:#888;font-size:11px'>{sz_str}</span>",
                    unsafe_allow_html=True)
        with c2:
            _download_button(f, "⬇️")
