"""
multislide_loader.py
--------------------
Multi-slide loader for the AGED vs ADULT mouse brain Xenium study.

Handles:
  - 4 AGED + 4 ADULT brain sections (8 slides total)
  - A shared base panel (Xenium_mBrain_v1_1) across all slides
  - Per-slide custom gene addons (varying between runs)
  - Panel harmonisation via PanelRegistry
  - ROI-based spatial subsetting via ROISelector
  - Proper replicate labelling for PyDESeq2 pseudobulk

Outputs a single concatenated AnnData with:
  .obs['condition']   : "AGED" | "ADULT"
  .obs['replicate']   : "AGED_1" ... "AGED_4" / "ADULT_1" ... "ADULT_4"
  .obs['slide_id']    : original slide identifier
  .obs['roi_name']    : ROI label (e.g. "MBH") if ROI was applied
  .var['panel_type']  : "base" | "custom"
  .var['cell_type_annotation'] : from Xenium metadata CSV
"""

import logging
from pathlib import Path
from typing import Optional, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.xenium_loader import load_xenium_run
from src.panel_registry import PanelRegistry
from src.roi_selector import ROISelector

logger = logging.getLogger(__name__)


# ===========================================================================
# SlideManifest: describes all 8 slides
# ===========================================================================

class SlideManifest:
    """
    Manifest of all slides in the AGED vs ADULT study.

    Can be built programmatically or loaded from a CSV.

    CSV format (no header row):
        slide_id, condition, run_dir, [replicate_id (optional)]

    Example
    -------
        manifest = SlideManifest()
        manifest.add("AGED_1",  "AGED",  "data/aged_run_1")
        manifest.add("AGED_2",  "AGED",  "data/aged_run_2")
        manifest.add("ADULT_1", "ADULT", "data/adult_run_1")
        ...
    """

    def __init__(self):
        self._slides: list[dict] = []

    def add(
        self,
        slide_id: str,
        condition: str,
        run_dir: Path | str,
        replicate_id: Optional[str] = None,
    ) -> "SlideManifest":
        """Register one slide."""
        entry = {
            "slide_id"    : slide_id,
            "condition"   : condition,
            "run_dir"     : Path(run_dir),
            "replicate_id": replicate_id or slide_id,
        }
        self._slides.append(entry)
        return self

    @classmethod
    def from_csv(cls, csv_path: Path | str) -> "SlideManifest":
        """
        Load manifest from a CSV file.

        Expected columns (order matters, no header needed if exactly 3 cols):
            slide_id, condition, run_dir
        Optional 4th column: replicate_id

        A header row is detected automatically by checking whether the
        first cell looks like a path or an identifier.
        """
        csv_path = Path(csv_path)
        df_raw = pd.read_csv(csv_path, header=None)

        # Auto-detect whether the first row is a header.
        # Heuristic: if the third cell of the first row is NOT a valid path
        # on disk (i.e. it looks like a column name such as "run_dir"), treat
        # the first row as a header.  This handles absolute paths, relative
        # paths, and non-standard directory names correctly.
        first_cell_path = Path(str(df_raw.iloc[0, 2]))
        has_header = not (first_cell_path.exists() or first_cell_path.is_absolute())

        if has_header:
            df = pd.read_csv(csv_path)
            df.columns = [c.lower().strip() for c in df.columns]
            logger.info("SlideManifest: detected header row in %s", csv_path)
        else:
            df = df_raw
            logger.info("SlideManifest: no header row detected in %s", csv_path)

        col_names = ["slide_id", "condition", "run_dir"]
        if df.shape[1] >= 4:
            col_names.append("replicate_id")
        df.columns = col_names + list(df.columns[len(col_names):])

        manifest = cls()
        for _, row in df.iterrows():
            manifest.add(
                slide_id     = str(row["slide_id"]),
                condition    = str(row["condition"]),
                run_dir      = Path(row["run_dir"]),
                replicate_id = str(row["replicate_id"]) if "replicate_id" in row else None,
            )
        logger.info("SlideManifest: loaded %d slides from %s", len(manifest), csv_path)
        return manifest

    @classmethod
    def from_dict(cls, slides: list[dict]) -> "SlideManifest":
        """Build from a list of dicts with keys: slide_id, condition, run_dir."""
        m = cls()
        for s in slides:
            m.add(**s)
        return m

    def __len__(self):
        return len(self._slides)

    def __iter__(self):
        return iter(self._slides)

    @property
    def slide_ids(self) -> list[str]:
        return [s["slide_id"] for s in self._slides]

    @property
    def conditions(self) -> list[str]:
        return [s["condition"] for s in self._slides]

    @property
    def run_dirs(self) -> list[Path]:
        return [s["run_dir"] for s in self._slides]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self._slides)


# ===========================================================================
# MultiSlideLoader
# ===========================================================================

class MultiSlideLoader:
    """
    Loads, harmonises, and optionally ROI-filters all slides in a study.

    Parameters
    ----------
    manifest:
        SlideManifest describing all slides.
    panel_registry:
        PanelRegistry loaded from the base panel CSV.
    roi_selector:
        Optional ROISelector. If provided, saved ROIs are applied per slide.
    panel_mode:
        'intersection' : keep only base panel genes present in all slides.
        'union'        : include all genes, zero-fill missing custom genes.
    apply_roi:
        If True and a roi_selector is given, filter cells to ROI per slide.
    """

    def __init__(
        self,
        manifest: SlideManifest,
        panel_registry: PanelRegistry,
        roi_selector: Optional[ROISelector] = None,
        panel_mode: Literal["intersection", "partial_union", "union"] = "partial_union",
        min_slides: int = 2,
        apply_roi: bool = True,
        output_dir: Optional[Path] = None,
    ):
        self.manifest        = manifest
        self.registry        = panel_registry
        self.roi_selector    = roi_selector
        self.panel_mode      = panel_mode
        self.min_slides      = min_slides
        self.apply_roi_flag  = apply_roi
        self.output_dir      = Path(output_dir) if output_dir else None

        self._per_slide: list[ad.AnnData] = []   # raw, pre-harmonisation
        self._harmonised: list[ad.AnnData] = []  # after panel harmonisation
        self._roi_filtered: list[ad.AnnData] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load_all(self) -> ad.AnnData:
        """
        Execute the full load-harmonise-filter-concatenate pipeline.

        Returns
        -------
        Single AnnData, all slides concatenated and annotated.
        """
        logger.info(
            "MultiSlideLoader: loading %d slides [panel_mode=%s, apply_roi=%s]",
            len(self.manifest), self.panel_mode, self.apply_roi_flag,
        )

        # 1. Load each slide
        self._per_slide = self._load_slides()

        # 2. Validate panels + print cross-slide custom gene overlap
        #    validate_slides() checks every slide has the expected base genes
        #    and returns a summary table.
        validation = self.registry.validate_slides(
            self._per_slide,
            self.manifest.slide_ids,
            raise_on_missing_base=False,
        )
        self.registry.print_overlap_summary(self._per_slide, self.manifest.slide_ids)

        # Save validation table — use the configured output_dir if set,
        # otherwise fall back to the current working directory.
        _out_dir = getattr(self, "output_dir", None) or Path.cwd()
        val_path = Path(_out_dir) / "panel_validation.csv"
        try:
            validation.drop(
                columns=["missing_base_genes", "custom_genes"], errors="ignore"
            ).to_csv(val_path, index=False)
            logger.info("Panel validation table saved: %s", val_path)
        except Exception as e:
            logger.warning("Could not save panel_validation.csv: %s", e)

        # 3. Harmonise panels to a common gene set
        #
        # Mode: partial_union (default)
        #   Base panel genes (247, identical on every slide):
        #     -> kept as-is, no zero-filling needed
        #   Custom genes (~50 per slide, partially overlapping):
        #     -> kept only if present in >= min_slides slides
        #     -> slides missing a kept custom gene receive a zero-filled column,
        #        flagged in adata.var["zero_filled"] = True
        #     -> custom genes unique to one slide are dropped (pure zero columns
        #        would only add noise to DGE)
        self._harmonised = self.registry.harmonise(
            self._per_slide,
            slide_ids  = self.manifest.slide_ids,
            mode       = self.panel_mode,
            min_slides = self.min_slides,
        )

        # 4. Apply ROIs
        if self.apply_roi_flag and self.roi_selector is not None:
            self._roi_filtered = self.roi_selector.apply_all(
                self._harmonised, self.manifest.slide_ids
            )
        else:
            self._roi_filtered = [a.copy() for a in self._harmonised]
            if self.apply_roi_flag and self.roi_selector is None:
                logger.warning(
                    "apply_roi=True but no roi_selector provided. "
                    "Returning full slides. Pass a ROISelector to enable ROI filtering."
                )

        # 5. Concatenate
        combined = self._concatenate(self._roi_filtered)
        return combined

    # ------------------------------------------------------------------
    # Per-slide loading
    # ------------------------------------------------------------------

    def _load_slides(self) -> list[ad.AnnData]:
        """
        Load each slide's full count matrix (base + custom genes combined)
        and add study-level metadata columns to .obs.
        """
        adatas = []
        for entry in self.manifest:
            sid   = entry["slide_id"]
            cond  = entry["condition"]
            rep   = entry["replicate_id"]
            d     = entry["run_dir"]

            adata = load_xenium_run(d, condition_label=cond, slide_id=sid)

            # Study-level obs columns
            adata.obs["replicate"] = rep
            adata.obs["replicate"] = adata.obs["replicate"].astype("category")

            adatas.append(adata)

        # Log a per-slide gene breakdown table using the registry
        self._log_panel_breakdown(adatas)
        return adatas

    def _log_panel_breakdown(self, adatas: list[ad.AnnData]) -> None:
        """
        Print a clear per-slide table showing base genes, custom genes,
        and total genes for every slide.  Flags any missing base genes.
        """
        base_set = self.registry.base_gene_set

        logger.info("=" * 65)
        logger.info("Per-slide gene panel composition")
        logger.info(
            "  %-12s  %-7s  %6s  %6s  %6s  %s",
            "Slide", "Cond", "Total", "Base", "Custom", "Status",
        )
        logger.info("  " + "-" * 60)

        for entry, adata in zip(self.manifest, adatas):
            sid      = entry["slide_id"]
            cond     = entry["condition"]
            genes    = set(adata.var_names)
            n_base   = len(genes & base_set)
            n_custom = len(genes - base_set)
            n_total  = adata.n_vars
            missing  = base_set - genes

            if missing:
                status = f"WARN: {len(missing)} base genes missing: "                          f"{sorted(missing)[:5]}"                          + (" ..." if len(missing) > 5 else "")
            else:
                status = "OK - all base genes present"

            logger.info(
                "  %-12s  %-7s  %6d  %6d  %6d  %s",
                sid, cond, n_total, n_base, n_custom, status,
            )

            # Store counts in uns for downstream use
            if "slide_info" in adata.uns:
                adata.uns["slide_info"]["n_genes_base"]   = n_base
                adata.uns["slide_info"]["n_genes_custom"] = n_custom

        logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    def _concatenate(self, adatas: list[ad.AnnData]) -> ad.AnnData:
        """
        Concatenate harmonised per-slide AnnData objects.

        Barcodes are prefixed with slide_id to avoid collisions.
        """
        prefixed = []
        for adata, entry in zip(adatas, self.manifest):
            sid = entry["slide_id"]
            adata = adata.copy()
            adata.obs_names = [f"{sid}__{bc}" for bc in adata.obs_names]
            prefixed.append(adata)

        combined = ad.concat(
            prefixed,
            axis=0,
            join="outer",
            merge="first",
            fill_value=0,
        )
        combined.obs_names_make_unique()
        combined.var_names_make_unique()

        # Re-cast to float32 sparse
        combined.X = sp.csr_matrix(combined.X.astype(np.float32))
        combined.layers["counts"] = combined.X.copy()

        # Rebuild categorical columns
        for col in ["condition", "replicate", "slide_id"]:
            if col in combined.obs.columns:
                combined.obs[col] = combined.obs[col].astype("category")

        # Re-build obsm["spatial"] from centroid columns if concat dropped it.
        # ad.concat preserves obsm only when all slides have identical obsm keys.
        if "spatial" not in combined.obsm:
            if "centroid_x" in combined.obs.columns and "centroid_y" in combined.obs.columns:
                combined.obsm["spatial"] = (
                    combined.obs[["centroid_x", "centroid_y"]].values.astype(np.float32)
                )
                logger.info("Rebuilt obsm['spatial'] from centroid_x/y after concat.")
            else:
                logger.warning(
                    "obsm['spatial'] missing after concat and no centroid columns found. "
                    "Spatial figures will be unavailable."
                )

        # Store run-level summary in uns
        combined.uns["study"] = {
            "conditions"  : self.manifest.conditions,
            "slide_ids"   : self.manifest.slide_ids,
            "panel_mode"  : self.panel_mode,
            "n_slides"    : len(self.manifest),
            "roi_applied" : self.apply_roi_flag,
        }

        n_by_cond = combined.obs.groupby("condition", observed=True).size().to_dict()
        logger.info(
            "Combined: %d cells x %d genes | %s",
            combined.n_obs, combined.n_vars,
            "  ".join(f"{k}={v}" for k, v in sorted(n_by_cond.items())),
        )
        return combined

    # ------------------------------------------------------------------
    # Access to intermediate results
    # ------------------------------------------------------------------

    def get_per_slide(self) -> list[ad.AnnData]:
        """Return raw per-slide AnnData objects (pre-harmonisation)."""
        return self._per_slide

    def get_harmonised(self) -> list[ad.AnnData]:
        """Return panel-harmonised per-slide AnnData objects (pre-ROI)."""
        return self._harmonised

    def get_roi_filtered(self) -> list[ad.AnnData]:
        """Return ROI-filtered per-slide AnnData objects."""
        return self._roi_filtered


# ===========================================================================
# Convenience function
# ===========================================================================

def load_aged_adult_study(
    aged_dirs: list[Path | str],
    adult_dirs: list[Path | str],
    base_panel_csv: Path | str,
    roi_cache_dir: Optional[Path | str] = None,
    panel_mode: Literal["intersection", "partial_union", "union"] = "partial_union",
    aged_ids: Optional[list[str]] = None,
    adult_ids: Optional[list[str]] = None,
    run_roi_session: bool = False,
    roi_mode: str = "polygon",
) -> tuple[ad.AnnData, MultiSlideLoader]:
    """
    One-call loader for the 4 AGED + 4 ADULT study.

    Parameters
    ----------
    aged_dirs:
        List of 4 Xenium run directories for AGED condition.
    adult_dirs:
        List of 4 Xenium run directories for ADULT condition.
    base_panel_csv:
        Path to Xenium_mBrain_v1_1_metadata.csv.
    roi_cache_dir:
        Directory for ROI JSON files. If None, ROI filtering is skipped.
    panel_mode:
        'partial_union' (recommended, default), 'intersection', or 'union'.
    aged_ids:
        Optional slide identifiers for AGED slides. Defaults to AGED_1..4.
    adult_ids:
        Optional slide identifiers for ADULT slides. Defaults to ADULT_1..4.
    run_roi_session:
        If True, open the interactive ROI selection GUI for slides that
        do not yet have a saved ROI. Requires a display.
    roi_mode:
        Drawing mode for the interactive session ('polygon' recommended).

    Returns
    -------
    (combined_adata, loader) tuple.
    combined_adata is the ready-to-analyse AnnData.
    loader gives access to per-slide and harmonised intermediates.
    """
    assert len(aged_dirs)  == 4, f"Expected 4 AGED dirs, got {len(aged_dirs)}"
    assert len(adult_dirs) == 4, f"Expected 4 ADULT dirs, got {len(adult_dirs)}"

    aged_ids  = aged_ids  or [f"AGED_{i+1}"  for i in range(4)]
    adult_ids = adult_ids or [f"ADULT_{i+1}" for i in range(4)]

    # Build manifest
    manifest = SlideManifest()
    for sid, d in zip(aged_ids, aged_dirs):
        manifest.add(slide_id=sid, condition="AGED", run_dir=d, replicate_id=sid)
    for sid, d in zip(adult_ids, adult_dirs):
        manifest.add(slide_id=sid, condition="ADULT", run_dir=d, replicate_id=sid)

    # Panel registry
    registry = PanelRegistry(base_panel_csv)

    # ROI selector
    roi_selector = None
    if roi_cache_dir is not None:
        roi_selector = ROISelector(cache_dir=roi_cache_dir)

    # Optionally run interactive ROI session
    if run_roi_session and roi_selector is not None:
        from src.roi_selector import interactive_roi_session
        # Load slides temporarily for the GUI
        temp_adatas = [
            load_xenium_run(d, condition_label=c)
            for d, c in zip(
                list(aged_dirs) + list(adult_dirs),
                ["AGED"] * 4 + ["ADULT"] * 4,
            )
        ]
        interactive_roi_session(
            adatas    = temp_adatas,
            slide_ids = aged_ids + adult_ids,
            cache_dir = roi_cache_dir,
            mode      = roi_mode,
            colour_key= None,
        )

    # Full load
    loader = MultiSlideLoader(
        manifest      = manifest,
        panel_registry= registry,
        roi_selector  = roi_selector,
        panel_mode    = panel_mode,
        apply_roi     = roi_cache_dir is not None,
    )
    combined = loader.load_all()
    return combined, loader
