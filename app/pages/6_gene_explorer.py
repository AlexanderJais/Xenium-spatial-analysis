"""
pages/6_gene_explorer.py
Spatial Gene Expression Explorer — generate a spatial expression map
for any gene across all 8 MBH slides on demand.

Loads from the preprocessed AnnData cache — no pipeline rerun needed.
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(
    page_title="Gene Explorer · Xenium DGE",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


inject_css()
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in {
    "output_dir": str(Path.home() / "xenium_dge_output"),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Colormap ──────────────────────────────────────────────────────────────────
GREY_RED = mcolors.LinearSegmentedColormap.from_list(
    "grey_red",
    ["#2A2A2A", "#7B1010", "#CC2222", "#FF6B35", "#FFD166"],
    N=256,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AnnData from cache…")
def _load_adata(cache_path: str, _mtime: float):
    """Load AnnData. _mtime is the file modification time — changing it
    automatically invalidates this cache when the file is updated on disk
    (e.g. after a new pipeline run). Users no longer need to restart the app."""
    import anndata as ad
    return ad.read_h5ad(cache_path)


def _get_lognorm(adata):
    import scipy.sparse as sp
    X = adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X
    return X


def _resolve_cache() -> Path | None:
    """Find the best available AnnData cache."""
    output_dir = Path(st.session_state["output_dir"])
    cache_dir  = output_dir.parent / (output_dir.name + "_cache")
    candidates = [
        cache_dir / "adata_mbh_preprocessed.h5ad",
        cache_dir / "adata_mbh_raw.h5ad",
        output_dir / "adata_mbh_final.h5ad",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _plot_gene(adata, gene: str, spot_size: float, vmax_pct: int) -> bytes:
    """Render spatial expression figure and return as PNG bytes."""
    gi   = list(adata.var_names).index(gene)
    X    = _get_lognorm(adata)
    _xi  = X[:, gi]
    expr = np.array(_xi.todense()).ravel() if hasattr(_xi, "todense") else np.array(_xi).ravel()
    vmax = float(np.percentile(expr[expr > 0], vmax_pct)) if (expr > 0).any() else 1.0

    cond_key  = "condition"
    slide_key = "slide_id" if "slide_id" in adata.obs.columns else cond_key
    conditions = sorted(adata.obs[cond_key].unique().tolist())
    slides_per_cond = {
        cond: sorted(adata.obs.loc[adata.obs[cond_key] == cond, slide_key].unique())
        for cond in conditions
    }
    all_cols = [(cond, sid) for cond in conditions for sid in slides_per_cond[cond]]
    n_cols   = len(all_cols)

    panel_w = 2.4
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(panel_w * n_cols, panel_w * 0.90),
        squeeze=False,
    )
    fig.patch.set_facecolor("#111111")

    sc = None
    for col, (cond, sid) in enumerate(all_cols):
        ax   = axes[0, col]
        mask = (adata.obs[cond_key] == cond) & (adata.obs[slide_key] == sid)
        sub  = adata[mask]

        if sub.n_obs == 0 or "spatial" not in sub.obsm:
            ax.set_facecolor("#111111")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="#888")
            ax.axis("off")
            continue

        xy = sub.obsm["spatial"]
        _e = X[mask, :][:, gi]
        e  = np.array(_e.todense()).ravel() if hasattr(_e, "todense") else np.array(_e).ravel()

        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=e, cmap=GREY_RED,
            vmin=0, vmax=vmax,
            s=spot_size, alpha=0.85,
            linewidths=0, rasterized=True,
        )
        ax.set_facecolor("#1A1A1A")
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines[:].set_visible(False)
        ax.set_title(f"{cond}\n{sid}", fontsize=5.5, color="#DDDDDD", pad=3)

    if sc is not None:
        cax  = axes[0, -1].inset_axes([1.03, 0.1, 0.06, 0.8])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("log-norm", fontsize=6, color="#CCCCCC", labelpad=3)
        cbar.ax.tick_params(labelsize=5, colors="#CCCCCC", width=0.4, length=2)
        cbar.outline.set_edgecolor("#444444")
        cbar.outline.set_linewidth(0.4)

    fig.suptitle(
        f"{gene}  —  spatial expression across all MBH slides",
        fontsize=9, color="#EEEEEE", y=1.02,
    )
    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ── Page ──────────────────────────────────────────────────────────────────────

page_header("🔬 Gene Explorer", "On-demand spatial expression maps for any gene across all 8 MBH slides")
st.markdown(
    "Generate a spatial expression map for any gene across all 8 MBH slides. "
    "Loads from the preprocessed cache — no pipeline rerun needed."
)

# ── Cache status ──────────────────────────────────────────────────────────────
cache_path = _resolve_cache()
if cache_path is None:
    st.error(
        "No AnnData cache found. "
        "Run the pipeline first (📊 Run Pipeline) to generate the cache, "
        "then return here."
    )
    st.stop()

col_cache, col_reload = st.columns([5, 1])
with col_cache:
    st.success(f"Cache: `{cache_path}`  (modified {__import__('datetime').datetime.fromtimestamp(cache_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})")
with col_reload:
    if st.button("🔄 Reload", help="Force reload after a new pipeline run"):
        st.cache_resource.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
# Pass file mtime so cache auto-invalidates after a new pipeline run
_mtime = cache_path.stat().st_mtime
adata = _load_adata(str(cache_path), _mtime)

# ── Gene input ────────────────────────────────────────────────────────────────
st.divider()

col_gene, col_opts = st.columns([2, 1])

with col_gene:
    gene_input = st.text_input(
        "Gene name",
        placeholder="e.g. Wfs1",
        help="Type any gene from the 307-gene panel. Case-insensitive.",
    ).strip()

    # Autocomplete suggestions as user types
    if gene_input and len(gene_input) >= 2:
        matches = [g for g in adata.var_names
                   if g.lower().startswith(gene_input.lower())]
        if matches and gene_input not in adata.var_names:
            st.caption(
                f"Suggestions: {',  '.join(matches[:8])}"
                + (" …" if len(matches) > 8 else "")
            )

with col_opts:
    spot_size = st.slider("Dot size", 0.5, 6.0, 2.5, 0.5,
                          help="Increase for sparser tissues, decrease if dots overlap.")
    vmax_pct  = st.slider("Colour scale max (percentile)", 90, 100, 99, 1,
                          help="99 = scale to 99th percentile of expressing cells.")

# ── Plot ──────────────────────────────────────────────────────────────────────
if gene_input:
    # Resolve gene name
    if gene_input in adata.var_names:
        gene = gene_input
    else:
        hits = [g for g in adata.var_names if g.lower() == gene_input.lower()]
        if hits:
            gene = hits[0]
            st.info(f"Using '{gene}' (case-corrected).")
        else:
            close = [g for g in adata.var_names if gene_input.lower() in g.lower()]
            st.error(
                f"**'{gene_input}' not found** in the 307-gene panel."
                + (f"  Did you mean: **{', '.join(close[:5])}**?" if close else "")
            )
            st.stop()

    with st.spinner(f"Rendering {gene} across all 8 slides…"):
        png_bytes = _plot_gene(adata, gene, spot_size, vmax_pct)

    st.image(png_bytes, use_container_width=True)

    # ── Download buttons ──────────────────────────────────────────────────────
    dl_col1, dl_col2, _ = st.columns([1, 1, 3])
    with dl_col1:
        st.download_button(
            "⬇️ Download PNG",
            data=png_bytes,
            file_name=f"{gene}_spatial_all_slides.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        # Also render a high-res PDF for publication
        if st.button("⬇️ Download PDF (300 dpi)", use_container_width=True):
            with st.spinner("Rendering high-res PDF…"):
                gi   = list(adata.var_names).index(gene)
                X    = _get_lognorm(adata)
                _xi  = X[:, gi]
                expr = np.array(_xi.todense()).ravel() if hasattr(_xi, "todense") else np.array(_xi).ravel()
                vmax = float(np.percentile(expr[expr > 0], vmax_pct)) if (expr > 0).any() else 1.0

                cond_key   = "condition"
                slide_key2 = "slide_id" if "slide_id" in adata.obs.columns else cond_key
                conditions = sorted(adata.obs[cond_key].unique().tolist())
                slides_per_cond = {
                    cond: sorted(adata.obs.loc[adata.obs[cond_key] == cond, slide_key2].unique())
                    for cond in conditions
                }
                all_cols   = [(cond, sid) for cond in conditions for sid in slides_per_cond[cond]]
                n_cols     = len(all_cols)
                panel_w    = 2.4

                fig2, axes2 = plt.subplots(1, n_cols,
                                           figsize=(panel_w * n_cols, panel_w * 0.90),
                                           squeeze=False)
                fig2.patch.set_facecolor("#111111")
                sc2 = None
                for col, (cond, sid) in enumerate(all_cols):
                    ax = axes2[0, col]
                    mask = (adata.obs[cond_key] == cond) & (adata.obs[slide_key2] == sid)
                    sub  = adata[mask]
                    if sub.n_obs == 0 or "spatial" not in sub.obsm:
                        ax.axis("off"); continue
                    xy = sub.obsm["spatial"]
                    _e = X[mask, :][:, gi]
                    e  = np.array(_e.todense()).ravel() if hasattr(_e, "todense") else np.array(_e).ravel()
                    sc2 = ax.scatter(xy[:,0], xy[:,1], c=e, cmap=GREY_RED,
                                     vmin=0, vmax=vmax, s=spot_size, alpha=0.85,
                                     linewidths=0, rasterized=True)
                    ax.set_facecolor("#1A1A1A"); ax.set_aspect("equal")
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.spines[:].set_visible(False)
                    ax.set_title(f"{cond}\n{sid}", fontsize=5.5, color="#DDDDDD", pad=3)
                if sc2 is not None:
                    cax2 = axes2[0,-1].inset_axes([1.03, 0.1, 0.06, 0.8])
                    cbar2 = fig2.colorbar(sc2, cax=cax2)
                    cbar2.set_label("log-norm", fontsize=6, color="#CCCCCC", labelpad=3)
                    cbar2.ax.tick_params(labelsize=5, colors="#CCCCCC", width=0.4, length=2)
                    cbar2.outline.set_edgecolor("#444444"); cbar2.outline.set_linewidth(0.4)
                fig2.suptitle(f"{gene}  —  spatial expression across all MBH slides",
                              fontsize=9, color="#EEEEEE", y=1.02)
                fig2.tight_layout(pad=0.3)
                pdf_buf = io.BytesIO()
                fig2.savefig(pdf_buf, format="pdf", dpi=300, bbox_inches="tight",
                             facecolor=fig2.get_facecolor())
                plt.close(fig2)
                pdf_buf.seek(0)
                st.download_button(
                    "📄 Save PDF",
                    data=pdf_buf.getvalue(),
                    file_name=f"{gene}_spatial_all_slides.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

# ── Quick gene info ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Panel gene reference")
st.caption("Browse all 307 genes in the harmonised panel.")

search = st.text_input("Filter genes", placeholder="Type to search…",
                        key="panel_search").strip().lower()
if "slides_present" in adata.var.columns or "panel_type" in adata.var.columns:
    var_df = adata.var.copy().reset_index()
    var_df.columns = [c if c != "gene_name" else "Gene" for c in var_df.columns]
    if "Gene" not in var_df.columns and "index" in var_df.columns:
        var_df = var_df.rename(columns={"index": "Gene"})
    show_cols = [c for c in ["Gene","panel_type","ensembl_id","zero_filled","slides_present"]
                 if c in var_df.columns]
    if search:
        var_df = var_df[var_df["Gene"].str.lower().str.contains(search, na=False)]
    st.dataframe(var_df[show_cols].reset_index(drop=True),
                 use_container_width=True, hide_index=True)
else:
    genes_list = pd.DataFrame({"Gene": list(adata.var_names)})
    if search:
        genes_list = genes_list[genes_list["Gene"].str.lower().str.contains(search, na=False)]
    st.dataframe(genes_list, use_container_width=True, hide_index=True)


