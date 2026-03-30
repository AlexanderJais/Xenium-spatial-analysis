# Codebase Audit Report

**Date:** 2026-03-30
**Scope:** Full repository audit — code quality, bugs, security, performance, documentation

---

## Executive Summary

The Xenium spatial analysis pipeline is a well-architected scientific computing application with strong domain-specific design (Xenium-tuned defaults, panel harmonisation, pseudoreplication-aware DGE). The codebase demonstrates good logging practices, defensive fallbacks for library version mismatches, and thorough docstrings.

This audit identified **8 critical issues**, **15 medium issues**, and **12 low-severity items** across the `src/`, `app/`, and project configuration files.

---

## 1. Critical Issues

### 1.1 Pipeline race condition — double-launch on rapid clicks
**File:** `app/pages/4_run.py:253-256`
The pipeline running flag is checked but not set before `_launch_pipeline()`. Rapid double-clicks can launch two concurrent pipelines overwriting each other's process handle.

**Fix:** Set `pipeline_running = True` in session state *before* calling `_launch_pipeline()`.

### 1.2 Untracked daemon thread — resource leak
**File:** `app/pages/4_run.py:188`
`threading.Thread(target=_stream, daemon=True).start()` — the thread object is never stored. If the subprocess hangs, there is no way to join or kill the thread programmatically.

**Fix:** Store the thread in session state for cleanup.

### 1.3 Session state defaults duplicated across 5+ files
**Files:** `app/app.py:27-66`, `app/pages/1_study_setup.py:21-33`, `app/pages/2_settings.py:17-26`, `app/pages/3_roi_manager.py:46-53`, `app/pages/4_run.py:31-50`
Adding a new parameter requires updating 5 files. If any is missed, the app uses stale defaults.

**Recommendation:** Centralise all defaults in `app.py` DEFAULTS dict; pages should read from session state only.

### 1.4 Broad exception handling hides real errors
**Files:** `app/app.py:327-335`, `app/pages/1_study_setup.py:187-188`
Bare `except Exception` catches everything including `KeyError`, `MemoryError`, etc. Real failures are silently swallowed.

### 1.5 DPI selectbox crash on corrupted session state
**File:** `app/pages/2_settings.py:337-343`
If `st.session_state["dpi"]` is set to a value not in `[150, 300, 600]`, `.index()` raises `ValueError` and crashes the page.

### 1.6 Gene explorer — unguarded `.index()` call
**File:** `app/pages/6_gene_explorer.py:82`
`list(adata.var_names).index(gene)` raises `ValueError` if gene not found. The outer try/except catches it but the error message is generic.

### 1.7 No subprocess timeout
**File:** `app/pages/4_run.py:173-188`
If the pipeline subprocess hangs (e.g. waiting for input), there is no timeout. The app will poll indefinitely with `time.sleep(0.5)` + `st.rerun()`.

### 1.8 Success messages cleared immediately by `st.rerun()`
**File:** `app/pages/3_roi_manager.py:349-356`
`st.success(...)` followed by `st.rerun()` — the success message is never visible to the user.

---

## 2. Medium Issues

### 2.1 Unpinned dependency versions
**File:** `requirements.txt`
All dependencies use `>=` without upper bounds. Different installs can produce different scientific results. No lock file exists for reproducibility.

**Recommendation:** Add `requirements-lock.txt` from `pip freeze` and use version ranges like `numpy >= 1.26, < 2.0`.

### 2.2 Hardcoded 8-slide manifest
**File:** `run_xenium_mbh.py:89-100`
`AGED_SLIDES` and `ADULT_SLIDES` are hardcoded lists. The pipeline cannot analyse a different number of slides without code changes.

### 2.3 Relative path assumptions
**Files:** `run_xenium_mbh.py:83-87`, `xenium_analysis.ipynb`
`ROOT_DATA = Path("data")` breaks if the script is run from a different working directory.

### 2.4 Hardcoded default cache path in plot_gene.py
**File:** `plot_gene.py:32-33`
`Path.home() / "xenium_dge_output" / "adata_mbh_final.h5ad"` — fails silently if user changed output_dir.

### 2.5 Home-directory restriction too strict
**File:** `app/pages/4_run.py:100-106`
Output and cache directories must be inside `Path.home()`. This blocks HPC scratch dirs, network mounts, and shared data drives.

### 2.6 Slider key collision with special characters
**File:** `app/pages/3_roi_manager.py:309-324`
Slide IDs with `/` or spaces create Streamlit widget keys like `sl_x0_Slide/1`. If a slide is renamed, all slider state is lost.

### 2.7 Repeated DataFrame loads in ROI summary
**File:** `app/pages/3_roi_manager.py:570-584`
`_load_cells(s["run_dir"])` is called for every slide on every render. Should be cached.

### 2.8 Log capped at 300 lines without notification
**File:** `app/pages/4_run.py:401`
`log_lines[-300:]` silently drops older output. Long runs lose important early messages.

### 2.9 maxUploadSize too low
**File:** `app/.streamlit/config.toml:13`
200 MB limit; `.h5ad` files from the pipeline can exceed this.

### 2.10 install_mac.sh — no disk space check
**File:** `install_mac.sh`
~1.8 GB required but no pre-flight check. Failure occurs mid-install with a confusing error.

### 2.11 install_mac.sh — no macOS version validation
**File:** `install_mac.sh:47`
Checks `Darwin` but not the macOS version. The README requires Ventura/Sonoma/Sequoia.

### 2.12 launcher.py — no subprocess error handling
**File:** `launcher.py:634-640`
`subprocess.Popen()` not wrapped in try/except. If Python binary not found, GUI crashes.

### 2.13 launcher.py — numeric parameters not validated
**File:** `launcher.py:437-442`
Spinbox/Entry fields accept any string; only validated at pipeline runtime.

### 2.14 No cleanup of partial cache on pipeline crash
**Files:** `run_xenium_mbh.py`, `src/pipeline.py:166-174`
If the pipeline crashes mid-write, partial `.h5ad` files remain and are loaded as valid cache on next run.

### 2.15 QUICKSTART_MAC.md Option C missing conda activate
**File:** `QUICKSTART_MAC.md:63-65`
CLI commands don't include `conda activate xenium_dge`.

---

## 3. Low-Severity Issues

### 3.1 Unused imports
- `app/pages/3_roi_manager.py:24` — `import sys` (reassigned on line 33)
- `app/pages/5_results.py:6` — `import base64` used inline but import is redundant pattern

### 3.2 Redundant session state reassignment
**File:** `app/pages/1_study_setup.py:193`
`st.session_state["slides"] = slides` — `slides` is already a reference to the session state dict.

### 3.3 Missing type hints
Most `app/` functions lack return type annotations and parameter types.

### 3.4 Inconsistent function naming
Mix of `_xenium_dir_status()` (returns tuple), `_xenium_dir_ok()` (returns bool), `_load_cells()` (returns tuple).

### 3.5 No pagination for large log viewer
**File:** `app/pages/5_results.py:309-312`
500-line tail renders slowly for very large logs.

### 3.6 Orphan installer temp file
**File:** `install_mac.sh:27,109`
`/tmp/Miniforge3-arm64.sh` is cleaned up on success but orphaned on early script exit.

### 3.7 Hardcoded representative slides
**File:** `run_xenium_mbh.py:312`
`CFG.representative_slides = {"AGED": "AGED_3", "ADULT": "ADULT_1"}` — silently uses wrong slides if manifest changes.

### 3.8 No NaN/Inf validation in ROI JSON import
**File:** `app/pages/3_roi_manager.py:167-188`
Vertex coordinate validation checks type but not `NaN`/`Inf`.

### 3.9 plot_gene.py — no range validation for --dpi and --spot-size
Accepts `--dpi 0` or negative `--spot-size`.

### 3.10 Notebook uses placeholder paths
**File:** `xenium_analysis.ipynb` cell 2
`Path('data/condition_A')` doesn't exist; notebook fails immediately.

### 3.11 Error log tag typo in launcher.py
**File:** `launcher.py:739`
Uses `"err"` tag instead of `"error"` — log line won't be colored correctly.

### 3.12 No download size warning for large files
**File:** `app/pages/5_results.py:46-57`
Files >100 MB are read entirely into memory with no user feedback.

---

## 4. Architecture & Design Notes

### Strengths
- **Domain-tuned defaults:** target_sum=100 (not 10k), n_top_genes=0 for targeted panels — backed by Xenium literature
- **Panel harmonisation:** Three-mode system (intersection/partial_union/union) with zero-fill tracking is well-designed
- **Pseudoreplication awareness:** `stringent_wilcoxon_dge` applies replicate consistency filters; clear warnings when using cell-level tests
- **Library compatibility:** Graceful fallbacks for scanpy API changes, harmonypy shape mismatches, and missing optional deps
- **Logging:** Comprehensive, readable log output with session markers and noisy-logger silencing

### Opportunities
- **Centralise app defaults** into a single dict in `app.py` to eliminate duplication
- **Add a requirements lock file** for reproducible environments
- **Parameterise the slide manifest** instead of hardcoding 8 slides
- **Cache validation** — write a `.complete` sentinel file alongside cache outputs; check for it before loading

---

## 5. Security Assessment

**Risk Level: LOW** (local-only scientific pipeline)

- No network-exposed endpoints (Streamlit runs locally with CORS disabled, XSRF enabled)
- No SQL/eval/exec injection surfaces
- File paths from user input are validated before use (existence checks, home-dir restriction)
- No secrets or credentials in the repository
- `.gitignore` properly excludes `.env`, `*.h5ad`, output directories

**Minor concerns:**
- JSON ROI import accepts unbounded array sizes (potential memory exhaustion)
- Exception messages containing user paths are displayed raw (minimal XSS risk in Streamlit context)

---

## 6. Test Coverage

No automated tests exist in the repository. Recommended additions:
1. Unit tests for `panel_registry.py` harmonisation logic
2. Unit tests for `dge_analysis.py` column normalisation
3. Integration test for `pipeline.py` with a small synthetic dataset
4. Streamlit page smoke tests using `streamlit.testing`
