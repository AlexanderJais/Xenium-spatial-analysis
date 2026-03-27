"""
roi_selector.py
---------------
Interactive Region-of-Interest (ROI) selector for Xenium spatial data.

Designed for selecting the mediobasal hypothalamus (MBH) — or any other
anatomical ROI — from a tissue spatial scatter plot.

Features
--------
* Three drawing modes:
    - 'polygon'   : click to add vertices, close with right-click or Enter
    - 'lasso'     : freehand lasso (click-drag)
    - 'rectangle' : click-drag bounding box
* Per-slide JSON persistence:
    roi_cache/<slide_id>_roi.json
  Re-running the pipeline re-uses saved ROIs without re-drawing.
* Programmatic ROI: pass coordinates directly (for scripted use or CI).
* MBH anatomical hint overlay (approximate atlas reference lines).
* apply_roi(): subset an AnnData to cells inside a saved ROI polygon.

Usage — interactive
-------------------
    from src.roi_selector import ROISelector

    selector = ROISelector(cache_dir="roi_cache")

    # Draw ROI for one slide (opens a matplotlib window)
    selector.draw(
        adata      = adata_slide1,
        slide_id   = "AGED_1",
        colour_key = "leiden",      # or None for density
        mode       = "polygon",
    )

    # Apply saved ROI to filter cells
    adata_mbh = selector.apply_roi(adata_slide1, slide_id="AGED_1")

Usage — batch (apply all saved ROIs)
-------------------------------------
    for adata, sid in zip(adatas, slide_ids):
        adata_mbh = selector.apply_roi(adata, sid)

Usage — programmatic (no GUI)
------------------------------
    selector.save_roi(
        slide_id = "AGED_1",
        vertices = [(x1,y1), (x2,y2), ...],   # polygon in µm
        roi_name = "MBH",
    )
"""

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from datetime import datetime

import anndata as ad
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approximate mediobasal hypothalamus atlas hint (placeholder centroids in
# normalised coordinates 0-1; real µm coords are section-dependent).
# Drawn as a dashed overlay to guide placement.
# ---------------------------------------------------------------------------
_MBH_ATLAS_HINT_LABEL = (
    "MBH region (atlas hint)\n"
    "Ventromedial + arcuate nucleus\n"
    "Adjust polygon to match your section"
)


class ROISelector:
    """
    Interactive and reproducible ROI tool for Xenium spatial data.

    Parameters
    ----------
    cache_dir:
        Directory where per-slide ROI JSON files are stored/loaded.
        Created automatically if it does not exist.
    """

    def __init__(self, cache_dir: Path | str = "roi_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ROISelector: cache directory = %s", self.cache_dir.absolute())

    # ------------------------------------------------------------------
    # Drawing entry point
    # ------------------------------------------------------------------

    def draw(
        self,
        adata: ad.AnnData,
        slide_id: str,
        colour_key: Optional[str] = None,
        mode: str = "polygon",
        show_mbh_hint: bool = True,
        figsize: tuple = (8, 7),
        force_redraw: bool = False,
        roi_name: str = "MBH",
    ) -> dict:
        """
        Open an interactive matplotlib window for ROI drawing.

        If a saved ROI already exists for this slide_id and force_redraw
        is False, the existing ROI is loaded and shown without prompting
        re-drawing.

        Parameters
        ----------
        adata:
            AnnData with .obsm['spatial'] (cell centroids in µm).
        slide_id:
            Unique identifier for this slide (used as filename key).
        colour_key:
            Optional adata.obs column to colour cells by (e.g. 'leiden').
            If None, cells are coloured by density.
        mode:
            'polygon' | 'lasso' | 'rectangle'
        show_mbh_hint:
            Overlay a dashed ellipse as anatomical placement guidance.
        figsize:
            Matplotlib figure size.
        force_redraw:
            If True, ignore any existing saved ROI and re-draw.
        roi_name:
            Label stored in the JSON for this ROI.

        Returns
        -------
        ROI dict with keys:
            slide_id, roi_name, vertices (list of [x, y]),
            n_cells_selected, created_at
        """
        roi_path = self._roi_path(slide_id)

        if roi_path.exists() and not force_redraw:
            logger.info(
                "ROI for '%s' already exists (%s). "
                "Pass force_redraw=True to re-draw.",
                slide_id, roi_path,
            )
            roi = self._load_roi(roi_path)
            self._preview_roi(adata, roi, colour_key=colour_key,
                              figsize=figsize, slide_id=slide_id)
            return roi

        if "spatial" not in adata.obsm:
            raise ValueError("adata.obsm['spatial'] required for ROI selection.")

        xy = adata.obsm["spatial"].astype(np.float64)

        if mode == "polygon":
            roi = self._draw_polygon(
                xy, slide_id, roi_name, adata, colour_key,
                show_mbh_hint, figsize,
            )
        elif mode == "lasso":
            roi = self._draw_lasso(
                xy, slide_id, roi_name, adata, colour_key,
                show_mbh_hint, figsize,
            )
        elif mode == "rectangle":
            roi = self._draw_rectangle(
                xy, slide_id, roi_name, adata, colour_key,
                show_mbh_hint, figsize,
            )
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose 'polygon', 'lasso', or 'rectangle'."
            )

        return roi

    # ------------------------------------------------------------------
    # Apply saved ROI
    # ------------------------------------------------------------------

    def apply_roi(
        self,
        adata: ad.AnnData,
        slide_id: str,
        invert: bool = False,
    ) -> ad.AnnData:
        """
        Subset adata to cells inside the saved ROI polygon for slide_id.

        Parameters
        ----------
        adata:
            AnnData with .obsm['spatial'].
        slide_id:
            Identifier matching the saved ROI.
        invert:
            If True, keep cells OUTSIDE the ROI instead.

        Returns
        -------
        Filtered AnnData. Original adata is not modified.
        """
        roi_path = self._roi_path(slide_id)
        if not roi_path.exists():
            raise FileNotFoundError(
                f"No ROI found for slide '{slide_id}' at {roi_path}. "
                "Run draw() first or use save_roi() for programmatic setup."
            )
        roi = self._load_roi(roi_path)
        vertices = np.array(roi["vertices"], dtype=np.float64).reshape(-1, 2)

        if "spatial" not in adata.obsm:
            raise ValueError("adata.obsm['spatial'] required.")

        xy = adata.obsm["spatial"].astype(np.float64)
        inside = _points_in_polygon(xy, vertices)

        if invert:
            mask = ~inside
        else:
            mask = inside

        result = adata[mask].copy()
        result.obs["roi_name"] = roi.get("roi_name", "ROI")
        logger.info(
            "ROI '%s' applied to '%s': %d / %d cells selected (%.1f%%)",
            roi.get("roi_name"), slide_id,
            mask.sum(), adata.n_obs, 100 * mask.sum() / adata.n_obs,
        )
        return result

    def has_roi(self, slide_id: str) -> bool:
        """Return True if a saved ROI exists for this slide_id."""
        return self._roi_path(slide_id).exists()

    # ------------------------------------------------------------------
    # Programmatic ROI saving (no GUI)
    # ------------------------------------------------------------------

    def save_roi(
        self,
        slide_id: str,
        vertices: Sequence[tuple[float, float]],
        roi_name: str = "MBH",
        n_cells_selected: Optional[int] = None,
    ) -> dict:
        """
        Save a polygon ROI programmatically (no GUI required).

        Parameters
        ----------
        slide_id:
            Slide identifier.
        vertices:
            List of (x, y) tuples in µm defining the polygon boundary.
        roi_name:
            Human-readable label stored in the JSON.
        n_cells_selected:
            Optionally store the expected cell count for documentation.

        Returns
        -------
        The saved ROI dict.
        """
        roi = {
            "slide_id"          : slide_id,
            "roi_name"          : roi_name,
            "vertices"          : [list(v) for v in vertices],
            "n_cells_selected"  : n_cells_selected,
            "created_at"        : datetime.now().isoformat(),
            "method"            : "programmatic",
        }
        self._save_roi(roi, self._roi_path(slide_id))
        return roi

    def list_rois(self) -> pd.DataFrame:
        """List all saved ROIs in the cache directory."""
        rows = []
        for f in sorted(self.cache_dir.glob("*_roi.json")):
            roi = self._load_roi(f)
            rows.append({
                "slide_id"        : roi.get("slide_id"),
                "roi_name"        : roi.get("roi_name"),
                "n_vertices"      : len(roi.get("vertices", [])),
                "n_cells_selected": roi.get("n_cells_selected"),
                "created_at"      : roi.get("created_at"),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def apply_all(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
    ) -> list[ad.AnnData]:
        """
        Apply saved ROIs to every slide in the list.
        Slides without a saved ROI are returned unchanged (with a warning).
        """
        results = []
        for adata, sid in zip(adatas, slide_ids):
            if self.has_roi(sid):
                results.append(self.apply_roi(adata, sid))
            else:
                logger.warning(
                    "No ROI found for '%s'; returning full slide. "
                    "Run draw() to define an ROI.", sid
                )
                results.append(adata.copy())
        return results

    # ------------------------------------------------------------------
    # Interactive drawing implementations
    # ------------------------------------------------------------------

    def _draw_polygon(
        self, xy, slide_id, roi_name, adata, colour_key, show_hint, figsize
    ) -> dict:
        """Click-to-add-vertex polygon selector with keyboard controls."""
        fig, ax = plt.subplots(figsize=figsize)
        self._scatter_background(ax, adata, xy, colour_key)
        if show_hint:
            self._draw_mbh_hint(ax, xy)
        ax.set_title(
            f"POLYGON ROI  |  slide: {slide_id}\n"
            "Left-click: add vertex   Right-click / Enter: close polygon   "
            "Escape: cancel",
            fontsize=9,
        )

        vertices = []
        polygon_line, = ax.plot([], [], "r-o", lw=1.5, ms=4, zorder=5)
        close_line,   = ax.plot([], [], "r--",  lw=1,   zorder=4)
        count_text    = ax.text(
            0.02, 0.97, "Vertices: 0",
            transform=ax.transAxes, fontsize=8, va="top",
            color="red", bbox=dict(fc="white", alpha=0.7, pad=2, ec="none"),
        )

        def _update_display():
            if vertices:
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                polygon_line.set_data(xs + [xs[0]], ys + [ys[0]])
                if len(vertices) > 1:
                    close_line.set_data([xs[-1], xs[0]], [ys[-1], ys[0]])
            count_text.set_text(f"Vertices: {len(vertices)}")
            fig.canvas.draw_idle()

        def _on_click(event):
            if event.inaxes is not ax:
                return
            if event.button == 1:     # left-click: add vertex
                vertices.append((event.xdata, event.ydata))
                _update_display()
            elif event.button == 3 and len(vertices) >= 3:  # right-click: close
                _close_polygon()

        def _close_polygon():
            plt.close(fig)

        def _on_key(event):
            if event.key == "enter" and len(vertices) >= 3:
                _close_polygon()
            elif event.key == "escape":
                vertices.clear()
                _close_polygon()
            elif event.key == "backspace" and vertices:
                vertices.pop()
                _update_display()

        fig.canvas.mpl_connect("button_press_event", _on_click)
        fig.canvas.mpl_connect("key_press_event", _on_key)
        plt.tight_layout()
        plt.show(block=True)

        return self._finalise_roi(vertices, slide_id, roi_name, xy, method="polygon")

    def _draw_lasso(
        self, xy, slide_id, roi_name, adata, colour_key, show_hint, figsize
    ) -> dict:
        """Freehand lasso selector."""
        from matplotlib.widgets import LassoSelector
        from matplotlib.path import Path as MplPath

        fig, ax = plt.subplots(figsize=figsize)
        self._scatter_background(ax, adata, xy, colour_key)
        if show_hint:
            self._draw_mbh_hint(ax, xy)
        ax.set_title(
            f"LASSO ROI  |  slide: {slide_id}\n"
            "Click and drag to draw freehand region. Release to confirm.",
            fontsize=9,
        )

        selected_vertices = []

        def _on_select(verts):
            selected_vertices.clear()
            selected_vertices.extend(verts)
            plt.close(fig)

        lasso = LassoSelector(ax, _on_select, useblit=True, lineprops={"color": "red", "lw": 1.5})
        plt.tight_layout()
        plt.show(block=True)

        return self._finalise_roi(selected_vertices, slide_id, roi_name, xy, method="lasso")

    def _draw_rectangle(
        self, xy, slide_id, roi_name, adata, colour_key, show_hint, figsize
    ) -> dict:
        """Click-drag rectangle selector converted to polygon vertices."""
        from matplotlib.widgets import RectangleSelector

        fig, ax = plt.subplots(figsize=figsize)
        self._scatter_background(ax, adata, xy, colour_key)
        if show_hint:
            self._draw_mbh_hint(ax, xy)
        ax.set_title(
            f"RECTANGLE ROI  |  slide: {slide_id}\n"
            "Click and drag to draw rectangle. Release to confirm.",
            fontsize=9,
        )

        rect_coords = []

        def _on_select(eclick, erelease):
            x0, x1 = sorted([eclick.xdata, erelease.xdata])
            y0, y1 = sorted([eclick.ydata, erelease.ydata])
            rect_coords.clear()
            rect_coords.extend([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            plt.close(fig)

        rs = RectangleSelector(
            ax, _on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="data",
            interactive=True,
            props=dict(facecolor="none", edgecolor="red", lw=1.5, alpha=0.8),
        )
        plt.tight_layout()
        plt.show(block=True)

        return self._finalise_roi(rect_coords, slide_id, roi_name, xy, method="rectangle")

    # ------------------------------------------------------------------
    # Preview of existing ROI
    # ------------------------------------------------------------------

    def _preview_roi(self, adata, roi, colour_key, figsize, slide_id):
        """Show saved ROI overlaid on spatial scatter (no interaction)."""
        if "spatial" not in adata.obsm:
            return
        xy = adata.obsm["spatial"].astype(np.float64)
        vertices = np.array(roi["vertices"], dtype=np.float64).reshape(-1, 2)
        inside   = _points_in_polygon(xy, vertices) if len(vertices) >= 3 \
                   else np.zeros(len(xy), dtype=bool)

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))

        for ax, mask, title in [
            (axes[0], np.ones(len(xy), dtype=bool), "All cells"),
            (axes[1], inside, f"ROI: {roi.get('roi_name', 'ROI')} ({inside.sum()} cells)"),
        ]:
            self._scatter_background(ax, adata, xy[mask], colour_key,
                                     use_subset_idx=np.where(mask)[0] if not mask.all() else None,
                                     full_adata=adata)
            if len(vertices) >= 2:
                closed = np.vstack([vertices, vertices[0]])
                ax.plot(closed[:, 0], closed[:, 1], "r-", lw=1.5, zorder=5,
                        label="ROI boundary")
                ax.fill(vertices[:, 0], vertices[:, 1],
                        color="red", alpha=0.08, zorder=3)
            ax.set_title(f"{slide_id}\n{title}", fontsize=8)

        plt.suptitle(
            f"Saved ROI for '{slide_id}' — {roi.get('roi_name', 'ROI')}\n"
            "Pass force_redraw=True to redraw",
            fontsize=8,
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Shared scatter background
    # ------------------------------------------------------------------

    def _scatter_background(
        self, ax, adata, xy, colour_key,
        use_subset_idx=None, full_adata=None,
    ):
        """Render the spatial scatter background for drawing."""
        effective_adata = full_adata if full_adata is not None else adata
        effective_xy    = effective_adata.obsm["spatial"] if full_adata is not None else xy

        if colour_key is not None and colour_key in effective_adata.obs.columns:
            labels = effective_adata.obs[colour_key].values
            unique = sorted(set(labels), key=str)
            cmap   = mpl.colormaps.get_cmap("tab20")
            colour_map = {u: cmap(i / max(len(unique), 1)) for i, u in enumerate(unique)}
            colours = [colour_map[l] for l in labels]

            plot_xy  = effective_xy[use_subset_idx] if use_subset_idx is not None else effective_xy
            plot_col = np.array(colours)[use_subset_idx] if use_subset_idx is not None else colours
        else:
            plot_xy  = effective_xy[use_subset_idx] if use_subset_idx is not None else effective_xy
            plot_col = "#3A3A3A"

        ax.scatter(
            plot_xy[:, 0], plot_xy[:, 1],
            c=plot_col, s=1.5, alpha=0.4,
            linewidths=0, rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)", fontsize=8)
        ax.set_ylabel("y (µm)", fontsize=8)
        _style_ax_minimal(ax)

    def _draw_mbh_hint(self, ax, xy):
        """
        Draw a dashed ellipse as an anatomical placement hint for the MBH.
        The ellipse is centred on the tissue midline, 30% down from the top
        of the bounding box — a rough heuristic. Users should adjust their
        polygon to match the actual section.
        """
        x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
        y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
        cx = (x_min + x_max) / 2
        cy = y_min + (y_max - y_min) * 0.35
        w  = (x_max - x_min) * 0.18
        h  = (y_max - y_min) * 0.18

        ellipse = mpatches.Ellipse(
            (cx, cy), width=w, height=h,
            fill=False, edgecolor="#FF8800", lw=1.2,
            linestyle="--", alpha=0.8, zorder=6,
        )
        ax.add_patch(ellipse)
        ax.text(
            cx, cy - h / 2 - (y_max - y_min) * 0.02,
            "↑ MBH atlas hint",
            ha="center", va="top", fontsize=6.5,
            color="#FF8800", style="italic",
            bbox=dict(fc="white", alpha=0.6, pad=1, ec="none"),
        )

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    def _roi_path(self, slide_id: str) -> Path:
        safe_id = slide_id.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_id}_roi.json"

    def _load_roi(self, path: Path) -> dict:
        with open(path) as fh:
            return json.load(fh)

    def _save_roi(self, roi: dict, path: Path):
        with open(path, "w") as fh:
            json.dump(roi, fh, indent=2)
        logger.info("ROI saved: %s", path)

    def _finalise_roi(
        self,
        vertices: list,
        slide_id: str,
        roi_name: str,
        xy: np.ndarray,
        method: str,
    ) -> dict:
        if len(vertices) < 3:
            logger.warning(
                "ROI for '%s' has fewer than 3 vertices; nothing saved.", slide_id
            )
            return {"slide_id": slide_id, "vertices": [], "n_cells_selected": 0}

        verts_arr = np.array(vertices)
        inside = _points_in_polygon(xy, verts_arr)
        n_sel  = int(inside.sum())

        roi = {
            "slide_id"         : slide_id,
            "roi_name"         : roi_name,
            "vertices"         : verts_arr.tolist(),
            "n_cells_selected" : n_sel,
            "created_at"       : datetime.now().isoformat(),
            "method"           : method,
            "bbox"             : {
                "x_min": float(verts_arr[:, 0].min()),
                "x_max": float(verts_arr[:, 0].max()),
                "y_min": float(verts_arr[:, 1].min()),
                "y_max": float(verts_arr[:, 1].max()),
            },
        }
        self._save_roi(roi, self._roi_path(slide_id))
        logger.info(
            "ROI '%s' for slide '%s': %d cells selected (%.1f%%)",
            roi_name, slide_id, n_sel, 100 * n_sel / max(len(xy), 1),
        )
        return roi


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Ray-casting algorithm: return boolean mask of points inside a polygon.

    Parameters
    ----------
    points:   (N, 2) array of (x, y) coordinates
    vertices: (M, 2) array of polygon vertices (does not need to be closed)

    Returns
    -------
    Boolean array of shape (N,).
    """
    from matplotlib.path import Path as MplPath
    poly = MplPath(vertices)
    return poly.contains_points(points)


def _style_ax_minimal(ax):
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ===========================================================================
# Convenience: run the selector on all slides in a study
# ===========================================================================

def interactive_roi_session(
    adatas: list[ad.AnnData],
    slide_ids: list[str],
    cache_dir: Path | str = "roi_cache",
    mode: str = "polygon",
    colour_key: Optional[str] = "leiden",
    force_redraw: bool = False,
) -> ROISelector:
    """
    Walk through all slides and draw (or show) an ROI for each one.

    Parameters
    ----------
    adatas:
        List of per-slide AnnData objects.
    slide_ids:
        List of slide identifiers (same length as adatas).
    cache_dir:
        Directory for storing ROI JSON files.
    mode:
        Drawing mode passed to ROISelector.draw().
    colour_key:
        obs column to colour cells by during drawing.
    force_redraw:
        If True, re-draw even if an ROI already exists.

    Returns
    -------
    The ROISelector instance (with all ROIs saved to cache_dir).
    """
    selector = ROISelector(cache_dir=cache_dir)

    for adata, sid in zip(adatas, slide_ids):
        print(f"\n{'='*55}")
        print(f"  Slide: {sid}  ({adata.n_obs} cells)")
        status = "EXISTS" if selector.has_roi(sid) else "NEW"
        print(f"  ROI status: {status}")
        print(f"{'='*55}")
        selector.draw(
            adata=adata, slide_id=sid,
            colour_key=colour_key,
            mode=mode,
            force_redraw=force_redraw,
        )

    print("\nROI session complete.")
    print(selector.list_rois().to_string(index=False))
    return selector
