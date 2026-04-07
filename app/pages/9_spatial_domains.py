"""
pages/9_spatial_domains.py
Spatial Domain Detection — interactive spatially-aware clustering
that integrates expression and physical tissue coordinates.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(
    page_title="Spatial Domains · Xenium DGE",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Defaults ────────────────────────────────────────────────────────────────
for k, v in {
    "sd_lambda": 0.3,
    "sd_resolution": 0.5,
    "sd_n_neighbors": 15,
    "sd_min_cells": 30,
    "sd_sweep_results": None,
    "sd_adata_path": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

page_header(
    "🗺️ Spatial Domain Detection",
    "Identify tissue regions using combined expression + spatial information",
)

st.markdown(
    "Standard Leiden clustering groups cells by **expression similarity** "
    "alone, ignoring their physical positions in tissue. Spatial domain "
    "detection combines both:\n\n"
    "1. **Expression graph** — KNN in PCA/Harmony space (from preprocessing)\n"
    "2. **Spatial graph** — KNN in physical tissue coordinates\n"
    "3. **Joint graph** — weighted blend of both\n"
    "4. **Leiden clustering** on the joint graph\n\n"
    "The **lambda (λ)** parameter controls the balance:\n"
    "- λ = 0 → pure expression clustering (same as standard Leiden)\n"
    "- λ = 1 → pure spatial clustering (ignores gene expression)\n"
    "- λ = 0.2–0.5 → recommended range for Xenium data\n\n"
    "After detecting domains, you can run **domain-specific DEG analysis** "
    "to identify marker genes for each tissue region."
)
st.divider()

# ── Find preprocessed AnnData ───────────────────────────────────────────────
_raw_dir = st.session_state.get("output_dir", "")
out_dir = Path(_raw_dir) if _raw_dir else None
_found_path = None
if out_dir is not None:
    _candidate_paths = [
        out_dir / "adata_preprocessed.h5ad",
        out_dir / "adata_final.h5ad",
        out_dir / "adata_mbh_final.h5ad",
    ]
    for _p in _candidate_paths:
        if _p.exists():
            _found_path = _p
            break

if _found_path is None:
    st.warning(
        "No preprocessed AnnData found. Run the main pipeline first "
        "(Steps 1–4) so that QC, normalisation, PCA, Harmony, and the "
        "KNN graph are computed."
    )

# ── Optional upload ──────────────────────────────────────────────────────────
with st.expander("Upload AnnData (.h5ad) manually", expanded=_found_path is None):
    uploaded = st.file_uploader(
        "Upload a pre-processed .h5ad file",
        type=["h5ad"],
        help="Must have obsp['connectivities'] (expression KNN graph) "
             "and obsm['spatial'] (cell centroids).",
    )
    if uploaded is not None:
        _tmp = Path(tempfile.mkdtemp()) / "uploaded.h5ad"
        _tmp.write_bytes(uploaded.read())
        _found_path = _tmp
        st.success(f"Uploaded file loaded ({_tmp.stat().st_size / 1e6:.1f} MB)")

if _found_path is None:
    st.stop()

st.success(f"Using: `{_found_path}`")
st.divider()

# ── Parameters ──────────────────────────────────────────────────────────────
st.subheader("Parameters")

c1, c2, c3, c4 = st.columns(4)
with c1:
    lam = st.slider(
        "λ spatial weight",
        0.0, 1.0,
        st.session_state["sd_lambda"], 0.05,
        help="0 = expression-only, 1 = spatial-only. Recommended: 0.2–0.5.",
    )
    st.session_state["sd_lambda"] = lam
with c2:
    res = st.slider(
        "Leiden resolution",
        0.05, 3.0,
        st.session_state["sd_resolution"], 0.05,
        help="Higher = more domains.",
    )
    st.session_state["sd_resolution"] = res
with c3:
    k_spatial = st.number_input(
        "Spatial KNN neighbours",
        5, 100,
        st.session_state["sd_n_neighbors"], 5,
    )
    st.session_state["sd_n_neighbors"] = k_spatial
with c4:
    min_cells = st.number_input(
        "Min fragment size",
        5, 500,
        st.session_state["sd_min_cells"], 5,
        help="Small disconnected fragments below this size are merged into "
             "neighbouring domains.",
    )
    st.session_state["sd_min_cells"] = min_cells

st.divider()

# Helper for consistent domain sorting
def _sort_key(x):
    try:
        return (0, int(x))
    except (ValueError, TypeError):
        return (1, str(x))

# ── Run detection ───────────────────────────────────────────────────────────
tab_detect, tab_sweep, tab_degs = st.tabs([
    "Run Detection", "Lambda Sweep", "Domain DEGs"
])

with tab_detect:
    run_clicked = st.button(
        "Run Spatial Domain Detection",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        # Clear stale results from previous runs
        st.session_state.pop("sd_deg_df", None)

        import anndata as ad
        from src.spatial_domain_detection import run_spatial_domain_pipeline

        with st.spinner("Loading AnnData..."):
            adata = ad.read_h5ad(_found_path)

        # Validate prerequisites
        _prereq_ok = True
        if "connectivities" not in adata.obsp:
            st.error("Missing expression KNN graph (obsp['connectivities']). Run preprocessing first.")
            _prereq_ok = False
        if "spatial" not in adata.obsm:
            st.error("Missing spatial coordinates (obsm['spatial']). Load Xenium data with coordinates.")
            _prereq_ok = False

        if _prereq_ok:
            with st.spinner(f"Detecting spatial domains (λ={lam:.2f}, res={res:.2f})..."):
                adata, deg_df = run_spatial_domain_pipeline(
                    adata,
                    lambda_spatial=lam,
                    resolution=res,
                    n_spatial_neighbors=k_spatial,
                    min_fragment_cells=min_cells,
                    domain_key="spatial_domain",
                    run_degs=True,
                    random_state=42,
                )

            n_domains = adata.obs["spatial_domain"].nunique()
            coherence = adata.uns.get("spatial_domain_coherence", 0)

            st.success(f"Detected **{n_domains} spatial domains** (coherence = {coherence:.3f})")

            # Save updated AnnData and DEGs
            if out_dir is not None and out_dir.exists():
                save_path = out_dir / "adata_spatial_domains.h5ad"
                adata.write_h5ad(save_path)
                st.info(f"Saved to `{save_path}`")

                if deg_df is not None:
                    deg_path = out_dir / "spatial_domain_degs.csv"
                    deg_df.to_csv(deg_path, index=False)

            # --- Visualisation ---
            st.subheader("Spatial Domain Map")
            import matplotlib.pyplot as plt
            from src.figures_spatial_domains import get_domain_colours

            domain_colours = get_domain_colours(adata, "spatial_domain")
            domains = sorted(domain_colours.keys(), key=_sort_key)

            # Pick a representative slide or use all cells
            slide_col = "slide_id" if "slide_id" in adata.obs.columns else None
            if slide_col:
                slides = adata.obs[slide_col].unique().tolist()
            else:
                slides = ["all"]

            cols = st.columns(min(4, len(slides)))
            for idx, slide in enumerate(slides):
                with cols[idx % len(cols)]:
                    if slide_col:
                        mask = adata.obs[slide_col] == slide
                    else:
                        mask = np.ones(adata.n_obs, dtype=bool)
                    sub = adata[mask]
                    xy = sub.obsm["spatial"]

                    fig, ax = plt.subplots(figsize=(4, 4))
                    for dom in domains:
                        dm = sub.obs["spatial_domain"].values == dom
                        if dm.sum() == 0:
                            continue
                        ax.scatter(
                            xy[dm, 0], xy[dm, 1],
                            c=domain_colours[dom], s=1, alpha=0.6,
                            edgecolors="none", rasterized=True,
                        )
                    ax.set_title(str(slide), fontsize=8)
                    ax.set_aspect("equal")
                    ax.invert_yaxis()
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

            # Domain composition
            st.subheader("Domain Composition")
            comp = adata.obs["spatial_domain"].value_counts().sort_index()
            st.bar_chart(comp)

            # Store DEGs for the DEG tab
            if deg_df is not None:
                st.session_state["sd_deg_df"] = deg_df

with tab_sweep:
    st.markdown(
        "Sweep across multiple λ values to find the optimal balance between "
        "expression and spatial information. The combined score weighs "
        "silhouette score (cluster separation) and spatial coherence "
        "(contiguity of domains) equally."
    )

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sweep_res = st.slider(
            "Resolution (held constant)", 0.1, 3.0, 0.5, 0.1,
            key="sweep_res",
        )
    with c_s2:
        sweep_lambdas_str = st.text_input(
            "Lambda values (comma-separated)",
            "0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7",
        )

    sweep_clicked = st.button(
        "Run Lambda Sweep", type="secondary", use_container_width=True,
    )

    if sweep_clicked:
        import anndata as ad
        from src.spatial_domain_detection import sweep_lambda

        try:
            sweep_lambdas = [float(x.strip()) for x in sweep_lambdas_str.split(",")]
        except ValueError:
            st.error("Invalid lambda values. Use comma-separated numbers.")
            sweep_lambdas = None

        if sweep_lambdas is not None:
            with st.spinner("Loading AnnData..."):
                adata = ad.read_h5ad(_found_path)

            if "connectivities" not in adata.obsp or "spatial" not in adata.obsm:
                st.error("AnnData missing expression graph or spatial coordinates.")
            else:
                with st.spinner(f"Sweeping {len(sweep_lambdas)} lambda values..."):
                    sweep_df = sweep_lambda(
                        adata,
                        lambdas=sweep_lambdas,
                        resolution=sweep_res,
                        n_spatial_neighbors=k_spatial,
                        random_state=42,
                    )

                st.session_state["sd_sweep_results"] = sweep_df

    # Display results
    if st.session_state["sd_sweep_results"] is not None:
        sweep_df = st.session_state["sd_sweep_results"]
        st.dataframe(sweep_df, use_container_width=True)

        if sweep_df["combined_score"].isna().all():
            st.warning("All sweep runs produced invalid scores. Try different parameters.")
        else:
            best = sweep_df.loc[sweep_df["combined_score"].idxmax()]
            sil_str = f"{best['silhouette_score']:.3f}" if not np.isnan(best["silhouette_score"]) else "N/A"
            st.success(
                f"Best λ = **{best['lambda']:.2f}** — "
                f"{int(best['n_domains'])} domains, "
                f"coherence = {best['spatial_coherence']:.3f}, "
                f"silhouette = {sil_str}"
            )

            # Plot
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].plot(sweep_df["lambda"], sweep_df["spatial_coherence"], "o-")
            axes[0].set_xlabel("λ"); axes[0].set_ylabel("Spatial Coherence")
            axes[1].plot(sweep_df["lambda"], sweep_df["silhouette_score"], "s-")
            axes[1].set_xlabel("λ"); axes[1].set_ylabel("Silhouette Score")
            axes[2].plot(sweep_df["lambda"], sweep_df["combined_score"], "D-")
            axes[2].axvline(best["lambda"], color="red", ls="--", alpha=0.5)
            axes[2].set_xlabel("λ"); axes[2].set_ylabel("Combined Score")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if st.button("Apply best λ"):
                st.session_state["sd_lambda"] = float(best["lambda"])
                st.rerun()

with tab_degs:
    st.markdown(
        "Marker genes per spatial domain — identified via Wilcoxon rank-sum "
        "test (domain vs. all other cells). These are the **spatially variable "
        "genes (SVGs)** that define each tissue region."
    )

    deg_df = st.session_state.get("sd_deg_df", None)
    if deg_df is None and out_dir is not None:
        # Try loading from disk
        deg_path = out_dir / "spatial_domain_degs.csv"
        if deg_path.exists():
            import pandas as pd
            deg_df = pd.read_csv(deg_path)
            st.session_state["sd_deg_df"] = deg_df

    if deg_df is None:
        st.info("Run spatial domain detection first to generate domain DEGs.")
    elif "domain" not in deg_df.columns or "log2fc" not in deg_df.columns or "pval_adj" not in deg_df.columns:
        st.error("DEG table missing expected columns (domain, log2fc, pval_adj).")
    else:
        # Filter controls
        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            domain_filter = st.selectbox(
                "Domain",
                ["All"] + sorted(deg_df["domain"].unique().tolist(), key=_sort_key),
            )
        with c_d2:
            lfc_min = st.slider("Min |log2FC|", 0.0, 5.0, 0.5, 0.1)
        with c_d3:
            pval_max = st.slider("Max adj. p-value", 0.001, 0.1, 0.05, 0.001, format="%.3f")

        filtered = deg_df.copy()
        if domain_filter != "All":
            filtered = filtered[filtered["domain"] == domain_filter]
        filtered = filtered[
            (filtered["log2fc"].abs() > lfc_min) & (filtered["pval_adj"] < pval_max)
        ]

        st.metric("Significant genes", len(filtered))
        st.dataframe(
            filtered.sort_values("log2fc", ascending=False).head(200),
            use_container_width=True,
        )

        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Download filtered DEGs (CSV)",
            csv, "spatial_domain_degs_filtered.csv",
            mime="text/csv",
        )
