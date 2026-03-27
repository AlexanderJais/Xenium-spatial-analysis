"""
cell_type_annotation.py
-----------------------
Automated cell type annotation for Xenium brain sections.

Three complementary strategies are provided:

1. MarkerScoring     -- score cells against curated marker gene lists
                        using scanpy's score_genes (based on Tirosh 2016).
                        Works with any gene panel; no reference needed.

2. CorrelationMapper -- correlate each cluster's mean expression profile
                        against a reference single-cell atlas average
                        (e.g. Allen Brain Atlas or user-supplied CSV).

3. ThresholdAnnotator -- hard-threshold approach: classify cells by
                         expression of a small set of canonical markers.
                         Useful as a sanity check.

All methods write the final label to adata.obs['cell_type'] (str) and
adata.obs['cell_type_confidence'] (float 0-1).
"""

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default brain cell-type marker dictionary
# (curated from MBH_cell_type_markers.csv — validated against
#  this specific AGED vs ADULT Xenium mBrain v1.1 dataset)
# ---------------------------------------------------------------------------
# ===========================================================================
# MBH-specific marker gene sets (mouse, Xenium panel gene names)
# Based on Xenium mBrain v1.1 base panel + custom hypothalamus panel.
# Updated from empirical dotplot analysis of this AGED vs ADULT dataset.
# ===========================================================================

MBH_MARKERS: dict[str, list[str]] = {
    "Orexigenic neuron": [
        "Agrp",
        "Npy",
        "Ghrh",
        "Ghsr",
    ],
    "Anorexigenic neuron": [
        "Pomc",
        "Cartpt",
        "Ppp1r17",
        "Mc3r",
        "Tac2",
        "Scgn",
        "Pnoc",
    ],
    "Dopaminergic neuron": [
        "Slc18a3",
        "Slc6a3",
        "Otp",
    ],
    "Peptidergic neuron": [
        "Cck",
        "Cckar",
        "Scgn",
        "Sst",
        "Sstr5",
        "Chrm2",
        "Chodl",
        "Npvf",
        "Brs3",
        "Crhr2",
        "Nmb",
        "Crh",
        "Nts",
        "Gal",
        "Lepr",
        "Galr1",
        "Galr3",
    ],
    "Glutamatergic neuron": [
        "Ucn3",
        "Esr1",
        "Adcyap1",
        "Bdnf",
        "Slc17a7",
    ],
    "Hcrt/Orexin neuron": [
        "Hcrt",
        "Gal",
        "Npy4r",
        "Glp1r",
    ],
    "GABAergic neuron": [
        "Synpr",
        "Prdm12",
        "Sstr5",
        "Pnoc",
        "Slc32a1",
        "Gad1",
        "Gad2",
        "Penk",
        "Tac1",
    ],
    "Tanycyte": [
        "Gpr50",
        "Slc2a2",
        "Mfsd2a",
        "Notch1",
        "Lef1",
        "Slc2a1",
    ],
    "Neuroendocrine neuron": [
        "Oxt",
        "Otp",
        "Scgn",
        "Avp",
    ],
    "Astrocyte": [
        "Aqp4",
        "Gfap",
        "Ntsr2",
        "Clmn",
        "Acsbg1",
        "Rfx4",
    ],
    "Oligodendrocyte": [
        "Sox10",
        "Opalin",
        "Sema3d",
        "Gjc3",
        "Gpr17",
    ],
    "OPC": [
        "Pdgfra",
        "Cspg4",
        "Gpr17",
    ],
    "Endothelial": [
        "Cldn5",
        "Pecam1",
        "Kdr",
        "Emcn",
        "Car4",
        "Ly6a",
        "Fgd5",
    ],
    "Pericyte": [
        "Acta2",
        "Cspg4",
        "Carmn",
        "Sncg",
        "Ano1",
    ],
    "Microglia": [
        "Siglech",
        "Spi1",
        "Ikzf1",
        "Arhgap25",
        "Trem2",
        "Cd68",
        "Cd300c2",
        "Laptm5",
        "Cd53",
    ],
    "VLMC": [
        "Col1a1",
        "Dcn",
        "Aldh1a2",
        "Igf2",
        "Fmod",
        "Slc13a4",
    ],
}


# ---------------------------------------------------------------------------
# Neuronal subtype markers for second-pass scoring within broad classes.
# Used by assign_labels_from_markers() to distinguish, e.g., GABAergic
# subtypes from each other once a cluster is classified as broadly GABAergic.
# ---------------------------------------------------------------------------
MBH_SUBTYPE_MARKERS: dict[str, list[str]] = {
    "AgRP / NPY neurons": [  # ARC: Agrp + Npy sufficient for identity
        "Agrp",
        "Npy",
        "Ghrh",
        "Ghsr",
    ],
    "POMC neurons": [  # ARC: Ppp1r17 strongly enriched in POMC
        "Pomc",
        "Cartpt",
        "Ppp1r17",
        "Mc3r",
    ],
    "KNDy neurons": [  # ARC: Kiss1 and Pdyn absent from panel
        "Tac2",
        "Scgn",
        "Pnoc",
    ],
    "Dopaminergic neurons (TIDA)": [  # ARC: Th absent; Slc6a3 annotated to Meis2 in panel
        "Slc18a3",
        "Slc6a3",
        "Otp",
    ],
    "CCK neurons": [  # ARC: CCK broadly expressed; Cckar autocrine
        "Cck",
        "Cckar",
        "Scgn",
    ],
    "Somatostatin neurons": [  # ARC: Sst annotated to Sst Chodl in base panel
        "Sst",
        "Sstr5",
        "Chrm2",
        "Chodl",
    ],
    "VMH neurons": [  # VMH: Nr5a1 (SF1) absent from panel
        "Ucn3",
        "Esr1",
        "Adcyap1",
        "Bdnf",
    ],
    "RFRP / NPVFergic neurons": [  # DMH: Npvf highly DMH-specific
        "Npvf",
        "Brs3",
        "Crhr2",
        "Nmb",
    ],
    "CRH-target / stress neurons": [  # DMH: Crh annotated to Vip interneurons in base panel
        "Crhr2",
        "Crh",
        "Nts",
    ],
    "Orexin / hypocretin neurons": [  # LHA: Hcrt primary; Gal co-expressed in ~80%
        "Hcrt",
        "Gal",
        "Npy4r",
        "Glp1r",
    ],
    "MCH neurons": [  # LHA: Pmch absent from panel; low specificity
        "Gal",
        "Nts",
        "Lepr",
    ],
    "GAL / metabolic neurons": [  # LHA: Overlaps with orexin neurons
        "Gal",
        "Galr1",
        "Galr3",
        "Lepr",
    ],
    "ZI GABAergic neurons": [  # ZI: Synpr most ZI-specific in panel
        "Synpr",
        "Prdm12",
        "Sstr5",
        "Pnoc",
    ],
    "ZI dopaminergic (A13)": [  # ZI: Th absent; shares markers with ARC TIDA
        "Slc18a3",
        "Slc6a3",
        "Otp",
    ],
    "Tanycytes (beta)": [  # 3V: Gpr50 most selective tanycyte marker
        "Gpr50",
        "Slc2a2",
        "Mfsd2a",
        "Notch1",
        "Lef1",
    ],
    "Tanycytes (alpha) / ependymal": [  # 3V: GLUT1 marks alpha-tanycytes and ependymal lining
        "Slc2a1",
        "Notch1",
        "Lef1",
    ],
    "Oxytocin neurons": [  # PVN: PVN-adjacent to 3V; Otp marks neuroendocrine lineage
        "Oxt",
        "Otp",
        "Scgn",
    ],
    "AVP neurons": [  # PVN: Avp also expressed in SCN; Otp shared with Oxt neurons
        "Avp",
        "Otp",
        "Scgn",
    ],
    "GABAergic neurons": [  # Pan-neuronal: Gad1/2 annotated to Lamp5 in base panel
        "Slc32a1",
        "Gad1",
        "Gad2",
    ],
    "Glutamatergic neurons": [  # Pan-neuronal: Slc17a6 (VGluT2) absent from panel
        "Slc17a7",
    ],
    "Enkephalin neurons": [  # Pan-neuronal: Broadly distributed; Penk annotated to Vip int. in panel
        "Penk",
        "Pnoc",
        "Tac1",
    ],
    "Astrocytes": [  # Glia: Ntsr2 enriched in hypothalamic astrocytes
        "Aqp4",
        "Gfap",
        "Ntsr2",
        "Clmn",
        "Acsbg1",
        "Rfx4",
    ],
    "Oligodendrocytes": [  # Glia: Sox10 pan-oligodendrocyte lineage
        "Sox10",
        "Opalin",
        "Sema3d",
        "Gjc3",
        "Gpr17",
    ],
    "OPCs": [  # Glia: Gpr17 marks transitional OPC-to-OL stage
        "Pdgfra",
        "Cspg4",
        "Gpr17",
    ],
    "Endothelial cells": [  # Vascular: Car4 / Ly6a help split arterial vs. venous
        "Cldn5",
        "Pecam1",
        "Kdr",
        "Emcn",
        "Car4",
        "Ly6a",
        "Fgd5",
    ],
    "Pericytes": [  # Vascular: Acta2 shared with SMC; Cspg4 more selective
        "Acta2",
        "Cspg4",
        "Carmn",
        "Sncg",
        "Ano1",
    ],
    "Homeostatic microglia": [  # Immune: Siglech downregulated upon activation
        "Siglech",
        "Spi1",
        "Ikzf1",
        "Arhgap25",
    ],
    "Activated / DAM microglia": [  # Immune: All significantly up in aged; classic DAM signature
        "Trem2",
        "Cd68",
        "Cd300c2",
        "Laptm5",
        "Cd53",
    ],
    "VLMC": [  # Vascular: Leptomeningeal / perivascular fibroblast-like
        "Col1a1",
        "Dcn",
        "Aldh1a2",
        "Igf2",
        "Fmod",
        "Slc13a4",
    ],
}

# Alias so existing code using BRAIN_MARKERS still works
BRAIN_MARKERS = MBH_MARKERS

# ===========================================================================
# Strategy 1: Marker gene scoring
# ===========================================================================

class MarkerScoring:
    """
    Score each cell against curated marker lists, then assign the
    cell type with the highest score.

    Parameters
    ----------
    markers:
        Dict mapping cell-type name to list of marker genes.
        Defaults to BRAIN_MARKERS.
    ctrl_size:
        Number of control genes per marker set (scanpy score_genes).
    min_score_delta:
        Minimum gap between top-1 and top-2 scores to assign a label
        (instead of "Unknown").
    """

    def __init__(
        self,
        markers: Optional[dict[str, list[str]]] = None,
        ctrl_size: Optional[int] = None,
        min_score_delta: float = 0.05,
    ):
        self.markers = markers or BRAIN_MARKERS
        # For Xenium panels (~100-500 genes), ctrl_size=50 is too large
        # (Tirosh 2016 used 50 for ~20k-gene scRNA-seq). Scale to panel
        # size: ~n_vars/25 genes per expression bin.
        self._ctrl_size_override = ctrl_size
        self.min_score_delta = min_score_delta

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Score all cells and assign cell types.

        Adds to adata.obs:
            '<cell_type>_score'  for every cell type
            'cell_type'          final label
            'cell_type_confidence'  score delta (top1 - top2)

        Returns the annotated AnnData.
        """
        import scanpy as sc

        # Use log-normalised values
        if "lognorm" in adata.layers:
            adata = adata.copy()
            adata.X = adata.layers["lognorm"]

        # Compute ctrl_size: scale to panel size if not overridden
        if self._ctrl_size_override is not None:
            ctrl_size = self._ctrl_size_override
        else:
            # ~n_vars/25 genes per expression bin (scanpy uses 25 bins)
            ctrl_size = max(1, min(50, adata.n_vars // 25))
        logger.info("score_genes ctrl_size = %d (panel has %d genes)", ctrl_size, adata.n_vars)

        score_keys = []
        for ct, gene_list in self.markers.items():
            available = [g for g in gene_list if g in adata.var_names]
            if len(available) < 2:
                logger.warning(
                    "Cell type '%s': only %d marker(s) in panel; skipping.",
                    ct, len(available)
                )
                continue
            key = ct.replace(" ", "_") + "_score"
            sc.tl.score_genes(
                adata,
                gene_list=available,
                ctrl_size=max(1, min(ctrl_size, adata.n_vars - len(available) - 1)),
                score_name=key,
            )
            score_keys.append((ct, key))
            logger.debug("Scored %d/%d genes for %s", len(available), len(gene_list), ct)

        if not score_keys:
            raise RuntimeError("No cell types could be scored (no marker genes found in panel).")

        # Build score matrix
        score_df = adata.obs[[k for _, k in score_keys]].copy()
        score_df.columns = [ct for ct, _ in score_keys]

        # Assign label: argmax with confidence gate
        top1_idx = score_df.values.argmax(axis=1)
        top1_scores = score_df.values[np.arange(len(score_df)), top1_idx]

        sorted_scores = np.sort(score_df.values, axis=1)[:, ::-1]
        delta = sorted_scores[:, 0] - sorted_scores[:, 1]

        cell_types_list = [ct for ct, _ in score_keys]
        labels = np.array(cell_types_list)[top1_idx].astype(str)
        labels[delta < self.min_score_delta] = "Unknown"

        adata.obs["cell_type"] = pd.Categorical(labels)
        adata.obs["cell_type_confidence"] = delta.astype(np.float32)

        n_unknown = (labels == "Unknown").sum()
        logger.info(
            "MarkerScoring: annotated %d cells; %d Unknown (%.1f%%)",
            adata.n_obs, n_unknown, 100 * n_unknown / adata.n_obs,
        )
        _log_type_counts(adata)
        return adata


# ===========================================================================
# Strategy 2: Correlation mapper (reference atlas)
# ===========================================================================

class CorrelationMapper:
    """
    Assign each Leiden cluster to a cell type by Pearson correlation
    between the cluster mean expression and a reference profile matrix.

    Parameters
    ----------
    reference_path:
        Path to a CSV with genes as rows and cell types as columns.
        Values should be mean log-normalised expression in the reference.
        If None, a built-in micro-reference is used (for demo only).
    cluster_key:
        obs column containing cluster assignments.
    min_correlation:
        Correlations below this threshold result in "Unknown".
    """

    def __init__(
        self,
        reference_path: Optional[Path] = None,
        cluster_key: str = "leiden",
        min_correlation: float = 0.3,
    ):
        self.reference_path = reference_path
        self.cluster_key = cluster_key
        self.min_correlation = min_correlation
        self._ref: Optional[pd.DataFrame] = None

    def load_reference(self, adata: ad.AnnData) -> pd.DataFrame:
        """Load and align reference to adata var_names."""
        if self.reference_path is not None:
            ref = pd.read_csv(self.reference_path, index_col=0)
        else:
            # Micro-reference built from BRAIN_MARKERS
            ref = _build_micro_reference(adata.var_names)

        # Align to panel
        shared = ref.index.intersection(adata.var_names)
        ref = ref.loc[shared]
        if len(shared) < 20:
            logger.warning(
                "Only %d genes shared between reference and panel.", len(shared)
            )
        return ref

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Assign cell types to clusters, then propagate to cells.

        Adds:
            adata.obs['cell_type']
            adata.obs['cell_type_confidence']
            adata.uns['cluster_to_celltype']
        """
        ref = self.load_reference(adata)
        self._ref = ref

        X = _get_lognorm(adata)
        var_map = {g: i for i, g in enumerate(adata.var_names)}

        clusters = sorted(adata.obs[self.cluster_key].unique(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
        cluster_to_ct: dict[str, str] = {}
        cluster_to_corr: dict[str, float] = {}

        for cl in clusters:
            mask = adata.obs[self.cluster_key] == cl
            _me = X[mask].mean(axis=0)  # sparse -> (1, n_genes) matrix; dense -> array
            mean_expr = np.asarray(_me).ravel()  # always 1-D

            # Align to reference genes
            gene_idx = [var_map[g] for g in ref.index if g in var_map]
            ref_genes = [g for g in ref.index if g in var_map]
            query = mean_expr[gene_idx]

            best_ct, best_corr = "Unknown", -1.0
            for ct in ref.columns:
                ref_vec = ref.loc[ref_genes, ct].values.astype(float)
                if np.std(ref_vec) < 1e-9 or np.std(query) < 1e-9:
                    continue
                corr = float(np.corrcoef(query, ref_vec)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_ct = ct

            if best_corr < self.min_correlation:
                best_ct = "Unknown"

            cluster_to_ct[cl] = best_ct
            cluster_to_corr[cl] = best_corr
            logger.debug(
                "Cluster %s -> %s  (r = %.3f)", cl, best_ct, best_corr
            )

        adata.obs["cell_type"] = pd.Categorical(
            adata.obs[self.cluster_key].map(cluster_to_ct)
        )
        adata.obs["cell_type_confidence"] = (
            adata.obs[self.cluster_key].map(cluster_to_corr).astype(np.float32)
        )
        adata.uns["cluster_to_celltype"] = cluster_to_ct

        logger.info("CorrelationMapper: cluster assignments:")
        for cl, ct in sorted(cluster_to_ct.items(),
                              key=lambda x: (0, int(x[0])) if str(x[0]).isdigit() else (1, str(x[0]))):
            logger.info("  Cluster %s -> %s (r=%.3f)", cl, ct, cluster_to_corr[cl])

        _log_type_counts(adata)
        return adata


# ===========================================================================
# Strategy 3: Threshold annotator
# ===========================================================================

class ThresholdAnnotator:
    """
    Simple hard-threshold cell type assignment.

    For each cell, iterate over canonical single-gene markers in priority
    order. Assign the first cell type whose marker gene exceeds the
    threshold in that cell. Useful as a fast sanity check.

    Parameters
    ----------
    canonical_markers:
        Ordered list of (cell_type, gene, threshold) tuples.
        Earlier entries take priority.
    """

    # Gene names must match the mouse MGI convention used by the Xenium mBrain
    # panel (lowercase, e.g. Gad1 not GAD1).  Using uppercase human HGNC names
    # here will cause every rule to be silently skipped because the gene will
    # not be found in var_map.
    DEFAULT_RULES = [
        ("Microglia",         "P2ry12",  0.5),
        ("Microglia",         "Cx3cr1",  0.5),
        ("Oligodendrocyte",   "Mbp",     1.0),
        ("Oligodendrocyte",   "Mog",     1.0),
        ("OPC",               "Pdgfra",  0.5),
        ("Astrocyte",         "Gfap",    0.5),
        ("Astrocyte",         "Aqp4",    0.5),
        ("Inhibitory Neuron", "Gad1",    0.5),
        ("Inhibitory Neuron", "Gad2",    0.5),
        ("Excitatory Neuron", "Slc17a7", 0.5),
        ("Excitatory Neuron", "Satb2",   0.5),
        ("Endothelial",       "Cldn5",   0.5),
        ("Pericyte",          "Pdgfrb",  0.5),
    ]

    def __init__(self, rules: Optional[list] = None):
        self.rules = rules or self.DEFAULT_RULES

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Apply rules sequentially and assign labels.

        Adds adata.obs['cell_type'] and 'cell_type_confidence'.
        Cells matching no rule are labelled 'Unknown'.
        """
        X = _get_lognorm(adata)
        var_map = {g: i for i, g in enumerate(adata.var_names)}

        # Warn if a large fraction of rules cannot be matched -- most likely
        # cause is a gene name case mismatch (e.g. human GAD1 vs mouse Gad1).
        n_skipped = sum(1 for _, g, _ in self.rules if g not in var_map)
        if self.rules and n_skipped / len(self.rules) > 0.5:
            missing = [g for _, g, _ in self.rules if g not in var_map]
            logger.warning(
                "ThresholdAnnotator: %d/%d rules skipped because the gene is "
                "not present in the panel. Check that gene names match the "
                "species convention (mouse: lowercase, e.g. 'Gad1'; human: "
                "uppercase, e.g. 'GAD1'). Missing genes: %s",
                n_skipped, len(self.rules), missing,
            )

        labels = np.full(adata.n_obs, "Unknown", dtype=object)
        confidence = np.zeros(adata.n_obs, dtype=np.float32)

        for ct, gene, thresh in self.rules:
            if gene not in var_map:
                continue
            gi = var_map[gene]
            expr = X[:, gi]
            mask = (expr > thresh) & (labels == "Unknown")
            labels[mask] = ct
            confidence[mask] = (expr[mask] / (expr[mask].max() + 1e-9)).astype(
                np.float32
            )

        adata.obs["cell_type"] = pd.Categorical(labels)
        adata.obs["cell_type_confidence"] = confidence

        n_unknown = (labels == "Unknown").sum()
        logger.info(
            "ThresholdAnnotator: %d Unknown (%.1f%%)",
            n_unknown, 100 * n_unknown / adata.n_obs,
        )
        _log_type_counts(adata)
        return adata


# ===========================================================================
# Convenience dispatcher
# ===========================================================================


def assign_labels_from_markers(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    broad_markers: Optional[dict] = None,
    subtype_markers: Optional[dict] = None,
    new_col: str = "cell_type",
    min_score: float = 0.0,
    fallback: str = "Unknown",
) -> ad.AnnData:
    """
    Data-driven cluster annotation: score each cluster's mean expression
    against MBH_MARKERS (broad types) then MBH_SUBTYPE_MARKERS (subtypes).

    This is the correct approach for targeted spatial panels because it
    derives labels from the actual expression data of the current run,
    making it robust to changes in cluster numbers across runs (which change
    with every different resolution, QC setting, or random seed).

    Workflow
    --------
    1.  Compute mean log-normalised expression per cluster.
    2.  For each cluster score against every broad cell type in
        ``broad_markers`` (default: MBH_MARKERS) by summing the z-scored
        mean expression of available marker genes.
    3.  Assign the broad type with the highest score.
    4.  Within clusters classified as GABAergic or Glutamatergic, run a
        second scoring pass using ``subtype_markers`` (default:
        MBH_SUBTYPE_MARKERS) to resolve the specific subtype.
    5.  Log all scores so the researcher can verify the assignments and
        override any cluster manually if needed.

    Parameters
    ----------
    adata:
        AnnData post-clustering with .obs[cluster_key] populated.
    cluster_key:
        Leiden (or other) cluster column.
    broad_markers:
        Dict of broad cell type -> marker gene list. Defaults to MBH_MARKERS.
    subtype_markers:
        Dict of subtype label -> marker gene list. Defaults to
        MBH_SUBTYPE_MARKERS. Only applied to neuronal clusters.
    new_col:
        Output obs column name.
    min_score:
        Minimum score for a cluster to receive a label (raw sum of z-scores).
        Clusters below this threshold are labelled ``fallback``.
    fallback:
        Label for clusters that fail the min_score gate.

    Returns
    -------
    AnnData with .obs[new_col] populated, .obs['cell_type_score' ] (float),
    and .uns['cluster_annotation_scores'] (DataFrame with all scores).
    """
    broad_markers   = broad_markers   or MBH_MARKERS
    subtype_markers = subtype_markers or MBH_SUBTYPE_MARKERS

    X    = _get_lognorm(adata)
    vn   = list(adata.var_names)
    vmap = {g: i for i, g in enumerate(vn)}

    clusters = sorted(
        adata.obs[cluster_key].astype(str).unique(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, x),
    )

    # ── Step 1: mean expression per cluster ──────────────────────────────────
    cluster_means: dict[str, np.ndarray] = {}
    for cl in clusters:
        mask = adata.obs[cluster_key].astype(str) == cl
        import scipy.sparse as _sp
        sub = X[mask.values]
        if _sp.issparse(sub):
            mu = np.asarray(sub.mean(axis=0)).ravel()
        else:
            mu = sub.mean(axis=0)
        cluster_means[cl] = mu

    # ── Step 2: score each cluster against cell-type markers ─────────────────
    # Build a (n_clusters x n_genes) matrix and z-score each gene across
    # clusters. This removes the broad-expression bias where genes like Aqp4
    # or Gfap, expressed at low levels in every cell type, inflate the score
    # of cell types that happen to have many such genes. After z-scoring, only
    # genes that are specifically elevated in a cluster relative to others
    # contribute a positive score.
    cl_list   = list(clusters)
    expr_mat  = np.stack([cluster_means[cl] for cl in cl_list], axis=0)  # (n_cl, n_genes)
    gene_mean = expr_mat.mean(axis=0)
    gene_std  = expr_mat.std(axis=0) + 1e-6
    zscore_mat = (expr_mat - gene_mean) / gene_std   # (n_cl, n_genes)
    cl_zscore  = {cl: zscore_mat[i] for i, cl in enumerate(cl_list)}

    def _score(cluster_id: str, gene_list: list[str]) -> float:
        """Sum of per-cluster z-scores for available marker genes."""
        z = cl_zscore[cluster_id]
        vals = [z[vmap[g]] for g in gene_list if g in vmap]
        return float(np.sum(vals)) if vals else 0.0

    broad_types = list(broad_markers.keys())
    score_rows  = []
    broad_labels: dict[str, str]  = {}
    broad_scores: dict[str, float] = {}

    for cl in clusters:
        mu  = cluster_means[cl]
        row = {"cluster": cl}
        for bt in broad_types:
            row[bt] = _score(cl, broad_markers[bt])
        score_rows.append(row)
        best_bt    = max(broad_types, key=lambda bt: row[bt])
        best_score = row[best_bt]
        broad_labels[cl] = best_bt if best_score >= min_score else fallback
        broad_scores[cl] = best_score

    score_df = pd.DataFrame(score_rows).set_index("cluster")

    # ── Step 3: second-pass subtype scoring ──────────────────────────────────
    # Maps each broad class to the list of subtypes it can resolve into.
    # All broad classes get a second pass — this resolves both neuronal
    # subtypes (e.g. Orexigenic -> AgRP/NPY vs POMC) and glial subtypes
    # (e.g. Microglia -> Homeostatic vs DAM).
    BROAD_TO_SUBTYPES: dict[str, list[str]] = {
        "Orexigenic neuron":    ["AgRP / NPY neurons"],
        "Anorexigenic neuron":  ["POMC neurons", "KNDy neurons"],
        "Peptidergic neuron":   [
            "CCK neurons", "Somatostatin neurons", "RFRP / NPVFergic neurons",
            "CRH-target / stress neurons", "MCH neurons", "GAL / metabolic neurons",
            "Enkephalin neurons",
        ],
        "Glutamatergic neuron": ["VMH neurons", "Glutamatergic neurons"],
        "Hcrt/Orexin neuron":   ["Orexin / hypocretin neurons"],
        "GABAergic neuron":     ["ZI GABAergic neurons", "GABAergic neurons"],
        "Dopaminergic neuron":  ["Dopaminergic neurons (TIDA)", "ZI dopaminergic (A13)"],
        "Neuroendocrine neuron":["Oxytocin neurons", "AVP neurons"],
        "Tanycyte":             ["Tanycytes (beta)", "Tanycytes (alpha) / ependymal"],
        "Astrocyte":            ["Astrocytes"],
        "Oligodendrocyte":      ["Oligodendrocytes"],
        "OPC":                  ["OPCs"],
        "Endothelial":          ["Endothelial cells"],
        "Pericyte":             ["Pericytes"],
        "Microglia":            ["Homeostatic microglia", "Activated / DAM microglia"],
        "VLMC":                 ["VLMC"],
    }

    subtypes = list(subtype_markers.keys())
    final_labels: dict[str, str]  = {}
    final_scores: dict[str, float] = {}

    for cl in clusters:
        broad   = broad_labels[cl]
        compat  = BROAD_TO_SUBTYPES.get(broad, [])

        if compat:
            srow = {st: _score(cl, subtype_markers.get(st, [])) for st in compat}
            best_st    = max(compat, key=lambda st: srow[st])
            best_score = srow[best_st]
            final_labels[cl] = best_st
            final_scores[cl] = best_score
        else:
            final_labels[cl] = broad
            final_scores[cl] = broad_scores[cl]

    # ── Step 4: log all assignments with scores ───────────────────────────────
    logger.info("="*65)
    logger.info("Cluster annotation (data-driven marker scoring):")
    logger.info("  %-6s  %-40s  %s", "ID", "Label", "Score")
    logger.info("  " + "-"*60)

    # Also log top-3 broad scores per cluster for transparency
    for cl in clusters:
        label = final_labels.get(cl, fallback)
        score = final_scores.get(cl, 0.0)
        row   = score_df.loc[cl] if cl in score_df.index else {}
        top3  = sorted(broad_types, key=lambda bt: row.get(bt, 0), reverse=True)[:3]
        top3_str = ", ".join(f"{bt.split()[0]}={row.get(bt,0):.2f}" for bt in top3)
        logger.info("  %-6s  %-40s  %.3f  [top3: %s]", cl, label, score, top3_str)

    logger.info("="*65)
    logger.info(
        "IMPORTANT: Review the scores above and the dotplot (fig4_dotplot.pdf) "
        "before proceeding. If any cluster assignment looks wrong, override it "
        "by passing a custom label_map to assign_cluster_labels()."
    )

    # ── Step 5: assign to adata ───────────────────────────────────────────────
    obs_labels = adata.obs[cluster_key].astype(str).map(final_labels).fillna(fallback)
    obs_scores = adata.obs[cluster_key].astype(str).map(final_scores).fillna(0.0)

    adata.obs[new_col]             = pd.Categorical(obs_labels)
    adata.obs["cell_type_score"]   = obs_scores.astype(np.float32)
    adata.uns["cluster_annotation_scores"] = score_df
    adata.uns["cluster_label_map"] = final_labels   # store for Supp Table 1

    n_unknown = (obs_labels == fallback).sum()
    if n_unknown > 0:
        logger.warning(
            "%d clusters could not be confidently annotated (labelled '%s'). "
            "Consider inspecting the dotplot and providing a manual label_map.",
            n_unknown, fallback,
        )
    _log_type_counts(adata)
    return adata

def annotate_cell_types(
    adata: ad.AnnData,
    method: str = "marker_scoring",
    markers: Optional[dict[str, list[str]]] = None,
    cluster_key: str = "leiden",
    reference_path: Optional[Path] = None,
    min_score_delta: float = 0.05,
    min_correlation: float = 0.3,
) -> ad.AnnData:
    """
    Unified cell type annotation dispatcher.

    Parameters
    ----------
    method:
        'marker_scoring' | 'correlation' | 'threshold'
    markers:
        Custom marker dict (marker_scoring only).
    cluster_key:
        Leiden cluster column (correlation only).
    reference_path:
        CSV reference atlas (correlation only).
    min_score_delta:
        Confidence gate (marker_scoring only).
    min_correlation:
        Confidence gate (correlation only).

    Returns
    -------
    AnnData with 'cell_type' and 'cell_type_confidence' in .obs.
    """
    if method == "marker_scoring":
        annotator = MarkerScoring(
            markers=markers or BRAIN_MARKERS,
            min_score_delta=min_score_delta,
        )
    elif method == "correlation":
        annotator = CorrelationMapper(
            reference_path=reference_path,
            cluster_key=cluster_key,
            min_correlation=min_correlation,
        )
    elif method == "threshold":
        annotator = ThresholdAnnotator()
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Choose 'marker_scoring', 'correlation', or 'threshold'."
        )

    return annotator.fit_transform(adata)


# ===========================================================================
# Helpers
# ===========================================================================

def _get_lognorm(adata: ad.AnnData) -> np.ndarray:
    if "lognorm" in adata.layers:
        X = adata.layers["lognorm"]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _log_type_counts(adata: ad.AnnData):
    if "cell_type" not in adata.obs:
        return
    counts = adata.obs["cell_type"].value_counts()
    logger.info("Cell type composition:")
    for ct, n in counts.items():
        logger.info("  %-25s %5d  (%.1f%%)", ct, n, 100 * n / adata.n_obs)


def _build_micro_reference(var_names) -> pd.DataFrame:
    """
    Build a minimal reference matrix from BRAIN_MARKERS for demo use.
    Each gene gets 1.0 in its cell type column, 0.0 elsewhere.
    """
    all_cts = list(BRAIN_MARKERS.keys())
    ref = pd.DataFrame(0.0, index=list(var_names), columns=all_cts)
    for ct, genes in BRAIN_MARKERS.items():
        for g in genes:
            if g in ref.index:
                ref.loc[g, ct] = 1.0
    # Remove all-zero rows
    ref = ref[ref.sum(axis=1) > 0]
    return ref
