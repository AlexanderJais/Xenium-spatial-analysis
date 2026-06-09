"""
defaults.py
-----------
Single source of truth for the web app's Streamlit session-state defaults.

Every page calls ``ensure_defaults(st.session_state)`` near the top, so adding
a new pipeline parameter only requires editing this file instead of the default
block in each of the ~8 pages. ``ensure_defaults`` never overwrites a value that
is already set — it only fills in missing keys — so pages can be opened in any
order and a value entered on one page is preserved when navigating to another.

Page-local widget state (e.g. the Leiden optimizer's ``optimizer_*`` keys or the
ROI manager's ``roi_last_slide``) is intentionally NOT defined here; it stays in
the page that owns it. This module holds only the shared pipeline configuration.
"""

from pathlib import Path

# This file lives in app/, so its parent is the app/ directory. Paths are
# computed exactly as the previous per-page default blocks computed them.
_APP_DIR = Path(__file__).parent


def _initial_slides() -> list[dict]:
    """The default 4 AGED + 4 ADULT slide template."""
    return [
        {"slide_id": f"AGED_{i}", "condition": "AGED", "run_dir": ""}
        for i in range(1, 5)
    ] + [
        {"slide_id": f"ADULT_{i}", "condition": "ADULT", "run_dir": ""}
        for i in range(1, 5)
    ]


def build_defaults() -> dict:
    """Return a fresh dict of every shared session-state default.

    A new dict (with a freshly-built ``slides`` list and ``roi_polygons`` dict)
    is returned on each call so callers never share mutable default objects.
    """
    return {
        # ── Study setup ──────────────────────────────────────────────────────
        "slides"        : _initial_slides(),
        "base_panel_csv": str(_APP_DIR / "data" / "Xenium_mBrain_v1_1_metadata.csv"),
        "output_dir"    : str(Path.home() / "xenium_dge_output"),
        "roi_cache_dir" : str(_APP_DIR / "roi_cache"),
        # ── Panel harmonisation ──────────────────────────────────────────────
        "panel_mode"    : "partial_union",
        "min_slides"    : 2,
        # ── Quality control ──────────────────────────────────────────────────
        "min_counts"    : 10,
        "max_counts"    : 2000,
        "min_genes"     : 10,
        "max_genes"     : 300,
        "filter_control_probes"    : True,
        "filter_control_codewords" : True,
        "normalize_by_cell_area"   : False,
        # ── Preprocessing / integration ──────────────────────────────────────
        "leiden_resolution" : 0.6,
        "n_neighbors"       : 12,
        "n_top_genes"       : 0,
        "harmony_max_iter"  : 30,
        # ── Differential expression ──────────────────────────────────────────
        "dge_method"      : "stringent_wilcoxon",
        "log2fc_threshold": 1.0,
        "pval_threshold"  : 0.01,
        # ── Spatial domain detection ─────────────────────────────────────────
        "run_spatial_domains"       : False,
        "lambda_spatial"            : 0.3,
        "spatial_domain_resolution" : 0.5,
        # ── Figure export ────────────────────────────────────────────────────
        "figure_format" : "pdf",
        "dpi"           : 300,
        # ── ROI ──────────────────────────────────────────────────────────────
        "roi_mode"      : "polygon",
        "roi_polygons"  : {},
        # ── Pipeline run state ───────────────────────────────────────────────
        "pipeline_running"    : False,
        "pipeline_log"        : [],
        "pipeline_returncode" : None,
        "pipeline_proc"       : None,
        "pipeline_log_queue"  : None,
    }


def ensure_defaults(session_state) -> None:
    """Populate ``session_state`` with any missing default keys.

    Idempotent: existing values are left untouched, so this is safe to call at
    the top of every page on every rerun.
    """
    for key, value in build_defaults().items():
        if key not in session_state:
            session_state[key] = value
