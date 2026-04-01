"""
pages/8_leiden_optimizer.py
Leiden Resolution Optimizer — automated sweep with silhouette + modularity scoring.
"""

import sys
from pathlib import Path

import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(
    page_title="Leiden Optimizer · Xenium DGE",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Defaults ────────────────────────────────────────────────────────────────
for k, v in {
    "leiden_resolution": 0.6,
    "optimizer_results": None,
    "optimizer_best": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

page_header(
    "🔎 Leiden Resolution Optimizer",
    "Automatically find the optimal clustering resolution by sweeping silhouette score and modularity",
)

st.markdown(
    "This tool runs Leiden clustering at multiple resolutions on your "
    "pre-processed data, then scores each resolution using **silhouette score** "
    "(cluster separation in PCA/Harmony space) and **modularity** "
    "(community structure in the KNN graph).  The combined score identifies "
    "the resolution that best balances cluster quality and granularity."
)
st.divider()

# ── Check for preprocessed data ──────────────────────────────────────────────
out_dir = Path(st.session_state.get("output_dir", ""))
adata_path = out_dir / "adata_preprocessed.h5ad"
adata_final_path = out_dir / "adata_final.h5ad"

# Try to find usable AnnData
_candidate_paths = [adata_path, adata_final_path]
_found_path = None
for _p in _candidate_paths:
    if _p.exists():
        _found_path = _p
        break

if _found_path is None:
    st.warning(
        "No preprocessed AnnData found.  Run the main pipeline first "
        "(steps 1–4) so that QC, normalisation, PCA, Harmony, and the "
        "KNN graph are computed.  The optimizer needs the neighbour graph "
        "to evaluate clustering quality."
    )
    st.info(
        f"Expected file at: `{adata_path}`\n\n"
        "Alternatively, you can upload an `.h5ad` file below."
    )

# ── Optional upload ──────────────────────────────────────────────────────────
with st.expander("Upload AnnData (.h5ad) manually", expanded=_found_path is None):
    uploaded = st.file_uploader(
        "Upload a pre-processed .h5ad file with a KNN graph",
        type=["h5ad"],
        help="The file must have obsp['connectivities'] (from sc.pp.neighbors) "
             "and obsm['X_pca'] or obsm['X_pca_harmony'].",
    )
    if uploaded is not None:
        import tempfile
        _tmp = Path(tempfile.mkdtemp()) / "uploaded.h5ad"
        _tmp.write_bytes(uploaded.read())
        _found_path = _tmp
        st.success(f"Uploaded file loaded ({_tmp.stat().st_size / 1e6:.1f} MB)")

if _found_path is None:
    st.stop()

st.success(f"Using: `{_found_path}`")
st.divider()

# ── Sweep configuration ─────────────────────────────────────────────────────
st.subheader("Sweep configuration")

c1, c2, c3 = st.columns(3)
with c1:
    res_min = st.number_input("Min resolution", 0.05, 5.0, 0.1, 0.05, format="%.2f")
with c2:
    res_max = st.number_input("Max resolution", 0.1, 5.0, 2.0, 0.1, format="%.2f")
with c3:
    res_step = st.number_input("Step size", 0.05, 1.0, 0.1, 0.05, format="%.2f")

if res_min >= res_max:
    st.error("Min resolution must be less than max resolution.")
    st.stop()

import numpy as np
n_steps = int(round((res_max - res_min) / res_step)) + 1
resolutions = [round(res_min + i * res_step, 2) for i in range(n_steps)]
st.caption(f"Will test **{len(resolutions)}** resolutions: {resolutions[0]} – {resolutions[-1]}")

c4, c5 = st.columns(2)
with c4:
    n_sample = st.number_input(
        "Max cells for silhouette score",
        1000, 200_000, 50_000, 5000,
        help="Silhouette score is O(n^2). Subsampling speeds up the sweep "
             "with minimal impact on ranking.",
    )
with c5:
    st.info(
        "**Scoring:** 60% silhouette (penalises over-fragmentation) + "
        "40% modularity (rewards community structure). Both are normalised "
        "to [0, 1] before combining."
    )

st.divider()

# ── Run sweep ────────────────────────────────────────────────────────────────
run_clicked = st.button(
    "Run Resolution Sweep",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.get("optimizer_running", False),
)

if run_clicked:
    import anndata as ad
    from src.preprocessing import optimize_leiden_resolution

    with st.spinner("Loading AnnData..."):
        adata = ad.read_h5ad(_found_path)

    # Verify prerequisites
    if "connectivities" not in adata.obsp:
        st.error(
            "The AnnData file is missing `obsp['connectivities']`. "
            "Run `sc.pp.neighbors()` (step 5 of the pipeline) first."
        )
        st.stop()

    has_embedding = any(k in adata.obsm for k in ("X_pca_harmony", "X_pca"))
    if not has_embedding:
        st.error(
            "No PCA embedding found (`X_pca` or `X_pca_harmony`). "
            "Run PCA before optimisation."
        )
        st.stop()

    st.info(
        f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes.  "
        f"Sweeping {len(resolutions)} resolutions..."
    )

    progress_bar = st.progress(0, text="Starting sweep...")

    def _progress_callback(step, total, res, metrics):
        progress_bar.progress(
            step / total,
            text=f"Resolution {res:.2f} — "
                 f"{metrics['n_clusters']} clusters, "
                 f"silhouette {metrics['silhouette']:.4f}  "
                 f"({step}/{total})",
        )

    result = optimize_leiden_resolution(
        adata,
        resolutions=resolutions,
        random_state=42,
        n_sample=n_sample,
        callback=_progress_callback,
    )

    progress_bar.progress(1.0, text="Sweep complete!")

    st.session_state["optimizer_results"] = result["results"]
    st.session_state["optimizer_best"] = result["best_resolution"]
    st.session_state["optimizer_best_row"] = result["best_row"]

    # Clean up memory
    del adata

    st.rerun()


# ── Display results ──────────────────────────────────────────────────────────
df = st.session_state.get("optimizer_results")
best_res = st.session_state.get("optimizer_best")
best_row = st.session_state.get("optimizer_best_row")

if df is not None and best_res is not None:
    st.subheader("Results")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Optimal resolution", f"{best_res:.2f}")
    with m2:
        st.metric("Clusters at optimum", int(best_row["n_clusters"]))
    with m3:
        st.metric("Silhouette score", f"{best_row['silhouette']:.4f}")
    with m4:
        st.metric("Modularity", f"{best_row['modularity']:.4f}")

    st.divider()

    # Charts
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Combined Score",
            "Number of Clusters",
            "Silhouette Score",
            "Modularity",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Combined score
    fig.add_trace(
        go.Scatter(
            x=df["resolution"], y=df["combined_score"],
            mode="lines+markers", name="Combined",
            line=dict(color="#1B4F8A", width=2.5),
            marker=dict(size=7),
        ),
        row=1, col=1,
    )
    fig.add_vline(
        x=best_res, line_dash="dash", line_color="#E74C3C",
        annotation_text=f"Best: {best_res:.2f}",
        annotation_position="top right",
        row=1, col=1,
    )

    # Number of clusters
    fig.add_trace(
        go.Bar(
            x=df["resolution"], y=df["n_clusters"],
            name="Clusters",
            marker_color=["#E74C3C" if abs(r - best_res) < 1e-6 else "#90C8F0"
                          for r in df["resolution"]],
        ),
        row=1, col=2,
    )

    # Silhouette
    fig.add_trace(
        go.Scatter(
            x=df["resolution"], y=df["silhouette"],
            mode="lines+markers", name="Silhouette",
            line=dict(color="#0A7E6E", width=2),
            marker=dict(size=6),
        ),
        row=2, col=1,
    )
    fig.add_vline(
        x=best_res, line_dash="dash", line_color="#E74C3C",
        row=2, col=1,
    )

    # Modularity
    fig.add_trace(
        go.Scatter(
            x=df["resolution"], y=df["modularity"],
            mode="lines+markers", name="Modularity",
            line=dict(color="#CE9178", width=2),
            marker=dict(size=6),
        ),
        row=2, col=2,
    )
    fig.add_vline(
        x=best_res, line_dash="dash", line_color="#E74C3C",
        row=2, col=2,
    )

    fig.update_layout(
        height=600, showlegend=False,
        template="plotly_white",
        title_text="Leiden Resolution Sweep Results",
        title_x=0.5,
    )
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Resolution", row=i, col=j)

    fig.update_yaxes(title_text="Combined Score", row=1, col=1)
    fig.update_yaxes(title_text="Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette", row=2, col=1)
    fig.update_yaxes(title_text="Modularity", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Full table
    with st.expander("Full results table", expanded=False):
        st.dataframe(
            df.style.highlight_max(subset=["combined_score"], color="#D4EDDA"),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Apply button
    st.subheader("Apply optimal resolution")
    col_apply, col_current = st.columns(2)
    with col_current:
        st.info(
            f"**Current setting:** {st.session_state['leiden_resolution']:.2f}\n\n"
            f"**Recommended:** {best_res:.2f} "
            f"({int(best_row['n_clusters'])} clusters, "
            f"silhouette {best_row['silhouette']:.4f})"
        )
    with col_apply:
        if st.button(
            f"Apply resolution {best_res:.2f} to pipeline settings",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["leiden_resolution"] = best_res
            st.success(
                f"Leiden resolution updated to **{best_res:.2f}**.  "
                "Go to **Settings** or **Run Pipeline** to use it."
            )
            st.balloons()

        # Also allow picking any resolution from the sweep
        manual_pick = st.selectbox(
            "Or pick a different resolution from the sweep",
            options=df["resolution"].tolist(),
            index=df["resolution"].tolist().index(best_res),
            format_func=lambda r: (
                f"{r:.2f}  —  {int(df.loc[df['resolution']==r, 'n_clusters'].iloc[0])} clusters, "
                f"silhouette {df.loc[df['resolution']==r, 'silhouette'].iloc[0]:.4f}"
            ),
        )
        if st.button(f"Apply resolution {manual_pick:.2f}"):
            st.session_state["leiden_resolution"] = float(manual_pick)
            st.success(f"Leiden resolution updated to **{manual_pick:.2f}**.")

else:
    st.info(
        "Configure the sweep parameters above and click **Run Resolution Sweep** "
        "to find the optimal Leiden resolution for your data."
    )
