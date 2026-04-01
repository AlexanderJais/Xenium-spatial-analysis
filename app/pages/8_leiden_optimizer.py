"""
pages/8_leiden_optimizer.py
Leiden Resolution Optimizer — automated sweep with multi-metric scoring
and clustree visualisation.
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
    "optimizer_cluster_assignments": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

page_header(
    "🔎 Leiden Resolution Optimizer",
    "Multi-metric sweep to find the optimal clustering resolution",
)

st.markdown(
    "This tool runs Leiden clustering at multiple resolutions on your "
    "pre-processed data, then scores each resolution using five complementary "
    "metrics:\n\n"
    "- **Silhouette score** — cluster separation in PCA/Harmony space (higher = better)\n"
    "- **Calinski-Harabasz index** — between- vs within-cluster variance ratio (higher = better)\n"
    "- **Davies-Bouldin index** — average similarity to most-similar cluster (lower = better)\n"
    "- **Spatial coherence** — fraction of spatial neighbours in the same cluster (higher = better)\n"
    "- **Modularity** — community structure quality on the KNN graph (higher = better)\n\n"
    "A weighted combined score identifies the resolution that best balances "
    "cluster quality and granularity. A **clustree** plot shows how clusters "
    "split and merge across resolutions."
)
st.divider()

# ── Check for preprocessed data ──────────────────────────────────────────────
out_dir = Path(st.session_state.get("output_dir", ""))
adata_path = out_dir / "adata_preprocessed.h5ad"
adata_final_path = out_dir / "adata_final.h5ad"
adata_mbh_final_path = out_dir / "adata_mbh_final.h5ad"

# Try to find usable AnnData
_candidate_paths = [adata_path, adata_final_path, adata_mbh_final_path]
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
        f"Expected one of:\n"
        f"- `{adata_path}`\n"
        f"- `{adata_final_path}`\n"
        f"- `{adata_mbh_final_path}`\n\n"
        "Alternatively, you can upload an `.h5ad` file below."
    )

# ── Optional upload ──────────────────────────────────────────────────────────
with st.expander("Upload AnnData (.h5ad) manually", expanded=_found_path is None):
    uploaded = st.file_uploader(
        "Upload a pre-processed .h5ad file with a KNN graph",
        type=["h5ad"],
        help="The file must have obsp['connectivities'] (from sc.pp.neighbors) "
             "and obsm['X_pca'] or obsm['X_pca_harmony'].  "
             "For spatial coherence, obsm['spatial'] is also needed.",
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
# Guard against floating-point overshoot
resolutions = [r for r in resolutions if r <= res_max + 1e-9]
st.caption(f"Will test **{len(resolutions)}** resolutions: {resolutions[0]} – {resolutions[-1]}")

c4, c5 = st.columns(2)
with c4:
    n_sample = st.number_input(
        "Max cells for metric computation",
        1000, 200_000, 50_000, 5000,
        help="Silhouette score is O(n^2). Subsampling speeds up the sweep "
             "with minimal impact on ranking. CH and DB also use the subsample.",
    )
with c5:
    st.info(
        "**Scoring weights (with spatial data):**\n"
        "- 30% silhouette\n"
        "- 15% Calinski-Harabasz\n"
        "- 15% Davies-Bouldin (inverted)\n"
        "- 20% spatial coherence\n"
        "- 20% modularity\n\n"
        "Without spatial coordinates, silhouette and modularity each get 35%."
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

    has_spatial = "spatial" in adata.obsm
    spatial_msg = "spatial coherence enabled" if has_spatial else "no spatial coords — spatial coherence disabled"

    st.info(
        f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes  ({spatial_msg}).  "
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
    st.session_state["optimizer_cluster_assignments"] = result["cluster_assignments"]

    # Clean up memory
    del adata

    st.rerun()


# ── Display results ──────────────────────────────────────────────────────────
df = st.session_state.get("optimizer_results")
best_res = st.session_state.get("optimizer_best")
best_row = st.session_state.get("optimizer_best_row")
cluster_assignments = st.session_state.get("optimizer_cluster_assignments")

if df is not None and best_res is not None:
    st.subheader("Results")

    # Metrics row
    has_spatial = df["spatial_coherence"].notna().all()
    if has_spatial:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m6 = None

    with m1:
        st.metric("Optimal resolution", f"{best_res:.2f}")
    with m2:
        st.metric("Clusters", int(best_row["n_clusters"]))
    with m3:
        st.metric("Silhouette", f"{best_row['silhouette']:.4f}")
    with m4:
        st.metric("Calinski-Harabasz", f"{best_row['calinski_harabasz']:.1f}")
    with m5:
        _db = best_row['davies_bouldin']
        st.metric("Davies-Bouldin", f"{_db:.4f}" if _db == _db else "N/A")
    if m6 is not None:
        with m6:
            _sc = best_row['spatial_coherence']
            st.metric("Spatial coherence", f"{_sc:.4f}" if _sc == _sc else "N/A")

    st.divider()

    # ── Metric plots ─────────────────────────────────────────────────────────
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_metric_rows = 3 if has_spatial else 2
    fig = make_subplots(
        rows=n_metric_rows, cols=2,
        subplot_titles=(
            "Combined Score",
            "Number of Clusters",
            "Silhouette Score (higher = better)",
            "Modularity (higher = better)",
            "Calinski-Harabasz Index (higher = better)",
            "Davies-Bouldin Index (lower = better)",
        ) if not has_spatial else (
            "Combined Score",
            "Number of Clusters",
            "Silhouette Score (higher = better)",
            "Spatial Coherence (higher = better)",
            "Calinski-Harabasz Index (higher = better)",
            "Davies-Bouldin Index (lower = better)",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    _best_color = "#E74C3C"

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
        x=best_res, line_dash="dash", line_color=_best_color,
        annotation_text=f"Best: {best_res:.2f}",
        annotation_position="top right",
        row=1, col=1,
    )

    # Number of clusters
    fig.add_trace(
        go.Bar(
            x=df["resolution"], y=df["n_clusters"],
            name="Clusters",
            marker_color=[_best_color if abs(r - best_res) < 1e-6 else "#90C8F0"
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
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=2, col=1)

    if has_spatial:
        # Spatial coherence
        fig.add_trace(
            go.Scatter(
                x=df["resolution"], y=df["spatial_coherence"],
                mode="lines+markers", name="Spatial Coherence",
                line=dict(color="#8E44AD", width=2),
                marker=dict(size=6),
            ),
            row=2, col=2,
        )
        fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=2, col=2)
    else:
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
        fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=2, col=2)

    # Calinski-Harabasz
    fig.add_trace(
        go.Scatter(
            x=df["resolution"], y=df["calinski_harabasz"],
            mode="lines+markers", name="Calinski-Harabasz",
            line=dict(color="#D4AC0D", width=2),
            marker=dict(size=6),
        ),
        row=n_metric_rows, col=1,
    )
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=n_metric_rows, col=1)

    # Davies-Bouldin
    fig.add_trace(
        go.Scatter(
            x=df["resolution"], y=df["davies_bouldin"],
            mode="lines+markers", name="Davies-Bouldin",
            line=dict(color="#E67E22", width=2),
            marker=dict(size=6),
        ),
        row=n_metric_rows, col=2,
    )
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=n_metric_rows, col=2)

    fig.update_layout(
        height=300 * n_metric_rows,
        showlegend=False,
        template="plotly_white",
        title_text="Leiden Resolution Sweep — Cluster Quality Metrics",
        title_x=0.5,
    )
    for i in range(1, n_metric_rows + 1):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Resolution", row=i, col=j)

    fig.update_yaxes(title_text="Combined Score", row=1, col=1)
    fig.update_yaxes(title_text="Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette", row=2, col=1)
    if has_spatial:
        fig.update_yaxes(title_text="Spatial Coherence", row=2, col=2)
    else:
        fig.update_yaxes(title_text="Modularity", row=2, col=2)
    fig.update_yaxes(title_text="Calinski-Harabasz", row=n_metric_rows, col=1)
    fig.update_yaxes(title_text="Davies-Bouldin", row=n_metric_rows, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # ── Clustree plot ────────────────────────────────────────────────────────
    if cluster_assignments is not None and len(cluster_assignments.columns) >= 2:
        st.divider()
        st.subheader("Clustree — cluster lineage across resolutions")
        st.caption(
            "Each node is a cluster at a given resolution. Edges show what "
            "fraction of cells in a higher-resolution cluster came from each "
            "lower-resolution cluster. Edge width encodes cell proportion."
        )

        _build_clustree(df, cluster_assignments, best_res)

    # ── Modularity plot (if spatial took its slot above) ─────────────────────
    if has_spatial:
        with st.expander("Modularity across resolutions"):
            fig_mod = go.Figure()
            fig_mod.add_trace(go.Scatter(
                x=df["resolution"], y=df["modularity"],
                mode="lines+markers", name="Modularity",
                line=dict(color="#CE9178", width=2),
                marker=dict(size=6),
            ))
            fig_mod.add_vline(x=best_res, line_dash="dash", line_color=_best_color)
            fig_mod.update_layout(
                height=300, template="plotly_white",
                xaxis_title="Resolution", yaxis_title="Modularity",
            )
            st.plotly_chart(fig_mod, use_container_width=True)

    # Full table
    with st.expander("Full results table", expanded=False):
        styler = (
            df.style
            .highlight_max(subset=["combined_score"], color="#D4EDDA")
            .highlight_max(subset=["silhouette"], color="#D4EDDA")
            .highlight_max(subset=["calinski_harabasz"], color="#D4EDDA")
        )
        # highlight_min on DB only if no NaN values present
        if df["davies_bouldin"].notna().all():
            styler = styler.highlight_min(subset=["davies_bouldin"], color="#D4EDDA")
        st.dataframe(styler, use_container_width=True, hide_index=True)

    st.divider()

    # Apply button
    st.subheader("Apply optimal resolution")
    col_apply, col_current = st.columns(2)
    with col_current:
        _db_str = f"{best_row['davies_bouldin']:.4f}" if best_row['davies_bouldin'] == best_row['davies_bouldin'] else "N/A"
        detail_lines = (
            f"**Current setting:** {st.session_state['leiden_resolution']:.2f}\n\n"
            f"**Recommended:** {best_res:.2f} "
            f"({int(best_row['n_clusters'])} clusters, "
            f"silhouette {best_row['silhouette']:.4f}, "
            f"CH {best_row['calinski_harabasz']:.1f}, "
            f"DB {_db_str})"
        )
        st.info(detail_lines)
    with col_apply:
        if st.button(
            f"Apply recommended resolution ({best_res:.2f})",
            type="primary",
            use_container_width=True,
            key="apply_best_resolution",
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
                f"sil {df.loc[df['resolution']==r, 'silhouette'].iloc[0]:.4f}, "
                f"CH {df.loc[df['resolution']==r, 'calinski_harabasz'].iloc[0]:.1f}"
            ),
        )
        if st.button("Apply selected resolution", key="apply_manual_resolution"):
            st.session_state["leiden_resolution"] = float(manual_pick)
            st.success(f"Leiden resolution updated to **{manual_pick:.2f}**.")

else:
    st.info(
        "Configure the sweep parameters above and click **Run Resolution Sweep** "
        "to find the optimal Leiden resolution for your data."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Clustree helper
# ═══════════════════════════════════════════════════════════════════════════════

def _build_clustree(results_df, cluster_df, best_res):
    """
    Build and display a clustree Sankey diagram.

    For each consecutive pair of resolutions, compute the overlap matrix:
    for every cluster at resolution r_{i+1}, count how many cells came from
    each cluster at resolution r_i.  Render as a Plotly Sankey diagram.
    """
    import plotly.graph_objects as go

    all_cols = sorted(cluster_df.columns, key=lambda c: float(c.split("_")[1]))
    if len(all_cols) < 2:
        st.info("Need at least 2 resolutions for a clustree plot.")
        return

    # Limit to avoid overly dense diagrams — subsample if > 12 levels
    if len(all_cols) > 12:
        step = max(1, len(all_cols) // 12)
        cols = all_cols[::step]
        # Always include the last resolution
        if cols[-1] != all_cols[-1]:
            cols.append(all_cols[-1])
    else:
        cols = all_cols

    # Build node list and edge list
    node_labels = []
    node_x = []
    node_y = []
    node_colors = []
    node_map = {}  # (col_name, cluster_id) -> node_index

    n_levels = len(cols)

    for level_i, col in enumerate(cols):
        res_val = float(col.split("_")[1])
        clusters = sorted(cluster_df[col].unique(), key=lambda x: int(x))
        n_cl = len(clusters)
        for ci, cl in enumerate(clusters):
            idx = len(node_labels)
            node_map[(col, cl)] = idx
            node_labels.append(f"r{res_val:.1f} c{cl}")
            # x: spread across resolutions
            node_x.append((level_i + 0.5) / (n_levels + 0.5))
            # y: spread clusters vertically
            node_y.append((ci + 0.5) / max(n_cl + 0.5, 1))
            # Colour best resolution's nodes differently
            if abs(res_val - best_res) < 0.005:
                node_colors.append("rgba(231, 76, 60, 0.85)")
            else:
                node_colors.append("rgba(52, 152, 219, 0.65)")

    sources = []
    targets = []
    values = []
    edge_colors = []

    for i in range(len(cols) - 1):
        col_lo = cols[i]
        col_hi = cols[i + 1]

        # Cross-tabulate
        lo = cluster_df[col_lo].values
        hi = cluster_df[col_hi].values

        clusters_lo = sorted(set(lo), key=lambda x: int(x))
        clusters_hi = sorted(set(hi), key=lambda x: int(x))

        for cl_hi in clusters_hi:
            mask_hi = hi == cl_hi
            total_hi = mask_hi.sum()
            if total_hi == 0:
                continue
            for cl_lo in clusters_lo:
                overlap = ((lo == cl_lo) & mask_hi).sum()
                if overlap == 0:
                    continue
                src = node_map.get((col_lo, cl_lo))
                tgt = node_map.get((col_hi, cl_hi))
                if src is not None and tgt is not None:
                    sources.append(src)
                    targets.append(tgt)
                    values.append(int(overlap))
                    frac = overlap / total_hi
                    edge_colors.append(
                        f"rgba(100, 100, 100, {max(0.08, min(0.6, frac))})"
                    )

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels,
            x=node_x,
            y=node_y,
            color=node_colors,
            pad=4,
            thickness=14,
            line=dict(color="black", width=0.3),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=edge_colors,
        ),
    )])

    fig.update_layout(
        title_text="Clustree: Cluster Lineage Across Resolutions",
        title_x=0.5,
        font_size=10,
        height=max(450, 35 * len(set(node_labels))),
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)
