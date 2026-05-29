# Codebase Audit Report

**Date:** 2026-05-29
**Scope:** Full repository audit — correctness, statistical validity, security, performance, robustness, documentation
**Method:** Every `.py` source file (~20,400 lines) was read in full and cross-checked. Key findings (crash bugs and statistical-validity claims) were independently re-verified against the source before inclusion.

---

## Executive Summary

The Xenium spatial-analysis pipeline remains a well-architected scientific application: Xenium-tuned defaults, three-mode panel harmonisation, pseudoreplication-aware DGE, graceful library-version fallbacks, and thorough logging. Most issues from the previous (2026-03-30) audit have been **fixed** — DPI/gene `.index()` crashes, the run-page double-launch race, gene-explorer guards, and the ROI "success message" flicker are resolved (see §5).

This audit found **3 critical issues** (2 are newly-introduced crash bugs), **statistical-validity concerns in the spatial-statistics module**, plus a set of medium/low robustness, performance, and reproducibility items. Two of the criticals are guaranteed crashes on a normal code path and should be fixed first.

> **Correction to a tempting-but-wrong finding:** `spatial_stats._compute_s1_s2` was suspected of "double-counting" the weight symmetrisation. It does **not** — `0.5·Σ(w_ij+w_ji)²` is the general Cliff–Ord definition of S₁ and is correctly implemented for any input matrix. The genuine problem is *which* matrix is passed (§2.1), not the function itself.

---

## 1. Critical Issues

### 1.1 `launcher.py:645` — call to undefined method `_set_running` (guaranteed crash)
In `_launch()`'s exception handler, a failed `subprocess.Popen` calls `self._set_running(False)`, but **no `_set_running` method exists** anywhere in the class (verified). So when a launch fails (missing `run_xenium_mbh.py`, bad `sys.executable`, unwritable dir), the error handler itself raises `AttributeError`, masking the real error and leaving the buttons in the wrong state.
**Fix:** Replace with the explicit resets already used in `_stop()`:
```python
self._launch_btn.config(state="normal")
self._stop_btn.config(state="disabled")
```

### 1.2 `app/pages/8_leiden_optimizer.py:407` — `_build_clustree` called before it is defined (`NameError`)
The page calls `_build_clustree(df, cluster_assignments, best_res)` at line 407, but `def _build_clustree` is at line 495 — *after* the call site in module top-to-bottom execution order (verified). Whenever `cluster_assignments` has ≥2 columns (the normal success path after a sweep), the page crashes with `NameError: name '_build_clustree' is not defined`.
**Fix:** Hoist the `def _build_clustree(...)` block above the results-display section (before line ~228).

### 1.3 `src/preprocessing.py:79` — raw counts layer aliased into `.X` (latent data corruption)
`adata.X = adata.layers["counts"]` assigns **by reference**. Downstream in-place operations on `.X` (notably `sc.pp.log1p`, which mutates `.data` in place for sparse matrices) can corrupt the preserved `layers["counts"]` — the very matrix relied upon for pseudobulk DGE. Whether corruption actually fires depends on scanpy's reassignment order, which makes this a fragile latent bug rather than a guaranteed one.
**Fix:** `adata.X = adata.layers["counts"].copy()`.

---

## 2. Statistical-Validity Issues (scientific correctness)

These affect the *numbers reported to users*, not whether the code runs. They warrant careful review by someone who owns the methodology.

### 2.1 `src/spatial_stats.py:108–139` — Moran's I variance uses a different weight matrix than the statistic
The statistic is computed with the **row-normalised** matrix `W_norm` (line 135: `I = (N / W_norm.sum()) * z'W_norm z / z'z`), but the variance components `S0, S1, S2` (lines 108–112) are computed from the **binary symmetric** `W_raw`. Under Cliff & Ord, the normality variance must be derived from the *same* weights used in the statistic. Mixing row-standardised weights in `I` with binary weights in `Var(I)` makes the z-score (line 139) and p-value (line 140) miscalibrated. The in-code comment ("S1, S2 must be computed from the RAW binary W … per Cliff & Ord") is only correct if `I` is *also* computed with binary `W` — which it is not.
**Fix:** Compute `I`, `S0`, `S1`, `S2` from a single consistent weight matrix (either binary `W_raw` throughout, or `W_norm` throughout).

### 2.2 `src/spatial_stats.py:358–361` — two-sided permutation p-value adds the wrong correction term
`p = min(count_up, count_low)·2/(B+1) + 2/(B+1) = (2·min + 2)/(B+1)`. The Phipson–Smyth bias-corrected two-sided value is `(2·min + 1)/(B+1)`; the observed statistic should also be included in the counts. As written the floor is `2/(B+1)` instead of `1/(B+1)`, and ties are double-counted (both `>=` and `<=`). Slightly over-inflates small p-values.
**Fix:** `p = np.clip((2*np.minimum(c_up, c_low) + 1) / (B + 1), 0, 1)` with the observed value included in the counts.

### 2.3 `src/spatial_stats.py:139` — `sqrt(abs(var_I_base))` hides negative variance
Taking `abs()` of a variance silently converts a degenerate/erroneous negative variance into a plausible z-score instead of surfacing it.
**Fix:** guard `var <= 0 → z=NaN` rather than `abs()`.

### 2.4 `src/composition_analysis.py:107` — `reference_cell_type='auto'` picks the *most abundant* type
scCODA requires a **stable** (low-variance) reference, and the docstring says as much, but `auto` selects by abundance (`idxmax`). If the largest type itself shifts with condition, every relative effect size is biased.
**Fix:** select the reference by minimal relative-abundance variance across replicates (scCODA's own heuristic).

### 2.5 `src/dge_analysis.py:469–497` — replicate-consistency filter compares against the opposite condition's grand mean
Each replicate's sign is tested against the *other* condition's grand mean, so any globally-shifted gene passes trivially, providing little pseudoreplication protection. A genuine consistency test compares each B-replicate to the distribution of A-replicates (or tests per-replicate LFC sign).

### 2.6 `src/cell_type_annotation.py:572–603` — scoring biases & missing absolute floor
- Specificity weighting is applied only when `len(available) >= 2` (line 572), so single-marker types (Hcrt/Oxt/Gal) keep an *unweighted* score while multi-marker types are scaled down — biasing argmax toward single-marker types. (Applying the weight unconditionally is harmless since a unique marker's weight is 1.0.)
- A cell whose scores are all ≤0 still receives a confident label as long as the top-two delta exceeds `min_score_delta` (lines 595–603). Add a `top1_score <= 0 → Unknown` gate.

### 2.7 `src/galanin_resistance.py:90` — resistance index silently returns 0 for missing genes
`_get_expr_vector` returns all-zeros for an absent gene, so `GRI = Gal/(Galr1+Galr3+1)` becomes `0` everywhere (with only a warning) when *Gal* isn't in the panel — indistinguishable from "no signal".
**Fix:** return `NaN` (or raise) when the GAL gene is absent.

---

## 3. Medium Issues (robustness, performance, reproducibility)

### Robustness / correctness
- **`app/pages/4_run.py:179–191` — no subprocess timeout (STILL PRESENT).** A hung child blocks the streamer thread on `proc.wait()` indefinitely. Add a timeout / heartbeat.
- **`app/pages/4_run.py:269` — `proc.terminate()` only.** A child blocked in native code (Harmony/scanpy) may ignore SIGTERM; escalate to `proc.kill()` after `wait(timeout=…)`.
- **`src/pipeline.py` galanin figure block — no try/except.** Spatial-domain and most figure blocks are guarded, but a galanin-figure failure aborts a run *after* DGE has already succeeded. Wrap it like the others.
- **`src/pipeline.py:171–179` — cache keyed only on a fixed filename.** Changing `min_counts`/`target_sum`/QC params silently reuses a stale `.h5ad`. Store a config hash in `uns` (or the filename) and compare before loading.
- **`src/preprocessing.py:142` — no zero-cell / zero-gene guard after filtering.** An over-aggressive filter yields an empty AnnData that fails with opaque errors downstream. Raise a clear error when `n_obs == 0` or `n_vars == 0`.
- **`src/multislide_loader.py:101` — header detection via filesystem existence.** `has_header = not (path.exists() or path.is_absolute())` misclassifies a not-yet-mounted relative `run_dir` as a header and silently drops the first slide. Detect by column-name keywords instead.
- **`src/dge_analysis.py:158–164` — PyDESeq2 contrast direction unverified.** Some pydeseq2 versions ignore `ref_level`; an unflagged reference flip inverts every LFC sign. Assert the realised contrast equals B-vs-A.
- **`src/roi_selector.py:414` — `LassoSelector(lineprops=…)`.** `lineprops` was renamed to `props` and removed in matplotlib ≥3.8; lasso drawing raises `TypeError` on current matplotlib (the rectangle path already uses `props`). Use `props={...}`.
- **`launcher.py:711–719` — numeric GUI params not validated.** `_validate()` checks paths/IDs only; a non-numeric resolution/threshold reaches the subprocess and surfaces as an opaque traceback. Coerce with `float()`/`int()` in `_validate()` and report inline.
- **`launcher.py:823` — `MPLBACKEND=MacOSX` set unconditionally.** Breaks matplotlib on Linux/Windows/headless. Guard with `if sys.platform == "darwin"`.
- **`app/pages/3_roi_manager.py:328–331` — slider clamp desync.** Clamped `x1/y1` are used for drawing while the slider widget still shows the unclamped value. Write back via session_state before render.
- **`app/pages/5_results.py:142` — thumbnail click is a dead state write.** Clicking a "Fig N" thumbnail sets `selected_fig_idx` then reruns, but the selectbox (line ~114) never reads it. Seed the selectbox index/key from `selected_fig_idx`.
- **`src/cluster_dge.py:214` — `groupby(...).apply(..., include_groups=False)` is pandas ≥2.2 only.** Raises `TypeError` on 2.1 after all DGE work is done. Use explicit aggregation.

### Performance / memory
- **`src/figures.py` & `src/figures_extended.py` — full-matrix densification.** `_get_lognorm(adata)` and several figure builders call `.toarray()` on the entire `(n_cells × n_genes)` matrix (e.g. `figures.py:1034`, `figures_extended.py:1829`); on a full Xenium dataset this is multiple GB. Slice to the needed genes before densifying.
- **`src/figures_extended.py:1842` — O(n_genes × n_cells) correlation loop.** `plot_gal_coexpression` computes `np.corrcoef` against Gal one gene at a time, re-densifying each column. Vectorise into a single masked correlation.
- **`src/multislide_loader.py:269` — full deep-copy of every slide when ROI is disabled.** `[a.copy() for a in self._harmonised]` copies all slide matrices needlessly; pass references through when no ROI filtering occurs.
- **`src/xenium_loader.py:286` / `multislide_loader.py:377` — redundant matrix copies.** `csr_matrix(combined.X.astype(np.float32))` plus `layers["counts"] = combined.X.copy()` makes ~3 transient copies of the combined matrix. Convert once.
- **`app/pages/1_study_setup.py:135` — uncached per-slide CSV reads.** `_xenium_dir_status()` re-reads gzipped features/barcodes for every slide on every rerun. Add `@st.cache_data` keyed on path+mtime.
- **`app/pages/5_results.py` — whole-file reads into memory (STILL PRESENT).** `.h5ad`/PDF/log/CSV are read in full for download/preview with no size cap; large outputs spike memory.

### Reproducibility / portability
- **`requirements.txt` — lower-bound-only pins, no lockfile.** `numpy>=1.26` etc. can pull breaking majors (numpy 2.x, pandas 2.x). Add upper bounds or ship a pinned lockfile / `environment.yml`.
- **`run_xenium_mbh.py:83–101` — data paths relative to CWD.** `ROOT_DATA = Path("data")` breaks when run from another directory even though imports are anchored to `__file__`. Anchor data paths to `Path(__file__).parent`.
- **`run_xenium_mbh.py:962` — `CACHE_DIR = Path(str(OUTPUT_DIR) + "_cache")`.** A trailing slash in `output_dir` yields `out/_cache` (a child) instead of sibling `out_cache`. Use `OUTPUT_DIR.parent / (OUTPUT_DIR.name + "_cache")`.

---

## 4. Low-Severity Issues

- **`src/figures.py` — `WONG[i]` indexing** at lines 225/367/933/975 raises `IndexError` with >8 conditions; use `WONG[i % len(WONG)]` (done elsewhere).
- **`src/figures.py:266` — `row[slide_col].replace(...)`** assumes a `str` slide id; cast with `str(...)`.
- **`src/figures_galanin_resistance.py:208` — unguarded `violinplot`** on `vals[vals>0]`: identical sparse values trigger `LinAlgError` from the internal KDE. Check `np.unique(...).size >= 2` or wrap in try/except (as `plot_galanin_panel` does).
- **Empty-data guards before `.max()`/`np.percentile`/`conditions[0]`** in figures (`figures.py:689`, `figures_extended.py:768`, galanin panels): add early `len()`/`.size` checks to emit a placeholder instead of `ValueError`/`IndexError` on empty or single-condition inputs.
- **`src/figures_spatial_domains.py:75,322` — `representative_slides` parameter accepted but unused** in `plot_spatial_domains`; remove from the signature or honour it.
- **`app/pages/1_study_setup.py:160,187,225,288` — broad `except Exception` (PARTIALLY FIXED).** `app.py` now uses narrow catches; these remain broad.
- **`app/pages/3_roi_manager.py:239,247,423,446` — `st.success` then `st.rerun` (PARTIALLY FIXED).** The save path now uses a `roi_just_saved` flag, but these other toasts still never render.
- **Path sanitisation (`3_roi_manager.py:88`, `4_run.py:126`)** sanitises only `/` and space; `..`, `\`, leading `~` in a user-edited slide id are not neutralised (mild path-traversal for ROI JSON writes). Use an allowlist sanitiser.
- **`app/pages/8_leiden_optimizer.py:473` — `tolist().index(best_res)`** raises `ValueError` on float-rounding mismatch; guard.
- **`run_galanin_resistance.py:43` — `--fmt` accepts any string** (unlike `plot_gene.py` which uses `choices=`); restrict to `{pdf,png,svg}`.
- **`xenium_analysis.ipynb` — placeholder paths** `data/condition_A` / `data/condition_B` fail immediately on a clean checkout with no guard/instruction cell.
- **`install_mac.sh`** — no free-disk-space pre-flight (~1.8 GB needed) and no Miniforge installer checksum verification; `[[ $? -eq 0 ]]` after a heredoc is fragile (capture into a variable).
- **Duplicated helpers in figures** — the cell-type label-shortening block, the `grey_red` colormap, `_clean_ax`/`_clean_umap_ax`, and representative-slide resolution are copy-pasted across `figures.py`, `figures_extended.py`, and `figures_galanin_resistance.py`. Factor into shared helpers.

---

## 5. Status of Previous (2026-03-30) Findings

| Prev. | Issue | Status |
|------|-------|--------|
| 1.1 | Run-page double-launch race | **FIXED** — `pipeline_running=True` set before launch (`4_run.py:262`); Streamlit serialises button events per rerun. |
| 1.2 | Untracked daemon thread | **PARTIALLY FIXED** — thread now stored in session state (`4_run.py:194`) but still never `join()`ed and lifecycle unmanaged on Stop. |
| 1.3 | Session-state defaults duplicated | **STILL PRESENT** — overlapping defaults in `app.py:27`, `1_study_setup.py:21`, `2_settings.py:17`, `4_run.py:31`; divergence already visible. |
| 1.4 | Broad `except Exception` | **PARTIALLY FIXED** — `app.py` narrowed; `1_study_setup.py` still broad. |
| 1.5 | DPI selectbox `.index()` crash | **FIXED** — guarded at `2_settings.py:396`. |
| 1.6 | Gene-explorer unguarded `.index()` | **FIXED** — clean `ValueError` raised before `.index()` (`6_gene_explorer.py:83`). |
| 1.7 | No subprocess timeout | **STILL PRESENT** — `4_run.py:179–191`. |
| 1.8 | `st.success` then `st.rerun` | **PARTIALLY FIXED** — save path uses a flag; other toasts still hidden. |
| 2.x | Unpinned deps / hardcoded manifest / relative paths / cache cleanup | **STILL PRESENT** — see §3. |

---

## 6. Architecture & Test Coverage

**Strengths** (unchanged): domain-tuned defaults, panel harmonisation with zero-fill tracking, pseudoreplication-aware DGE with loud warnings, robust library-version fallbacks, comprehensive logging.

**Still no automated tests.** The two critical crash bugs (§1.1, §1.2) and the statistical issues (§2) are exactly what a minimal test layer would catch. Recommended first tests:
1. Import-and-instantiate smoke tests for `launcher.py` and a `streamlit.testing` render of page 8 (would catch §1.1, §1.2 immediately).
2. Unit test for `panel_registry` harmonisation (intersection/partial_union/union + zero-fill alignment).
3. Numerical test for `spatial_stats.morans_i_scan` against a tiny grid with a known analytic Moran's I and variance (would catch §2.1/§2.2).
4. `dge_analysis` column-normalisation + contrast-direction test.

---

## 7. Security Assessment

**Risk Level: LOW** (local-only scientific pipeline). No network-exposed endpoints, no eval/exec/SQL injection surfaces, no secrets in-repo, `.gitignore` correctly excludes `.env`/`*.h5ad`/outputs.

Minor concerns: ROI JSON import accepts unbounded array sizes (memory exhaustion); ROI/output path sanitisation does not neutralise `..`/`\`/`~` (§4); the Miniforge installer is executed without checksum verification (§4). None are remotely exploitable.
