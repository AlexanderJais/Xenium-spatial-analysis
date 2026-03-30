"""
cell_type_annotation.py
-----------------------
Automated cell type annotation for Xenium MBH (mediobasal hypothalamus) sections.

Marker gene sets curated from:
    - Campbell et al. (2017)     ARC-ME molecular census, 20,921 cells
    - Chen et al. (2017)         hypothalamic non-neuronal diversity
    - Romanov et al. (2017)      hypothalamic neuron subtypes (A12-A15)
    - Kim et al. (2019)           VMH transcriptomic atlas (17 types)
    - Affinati et al. (2021)     cross-species VMH classes
    - Steuernagel et al. (2022)  HypoMap unified atlas (384,925 cells)

Design principles for marker selection
---------------------------------------
1.  Use the FEWEST genes that uniquely separate each population from its
    neighbours. Most broad types need only 2-3 genes; subtypes need 1-2.
2.  Rank genes by specificity: tier-1 genes appear in exactly ONE marker
    list; tier-2 genes appear in two lists; tier-3 genes appear in three
    or more. Scoring weights genes inversely to their overlap count.
3.  Avoid "hub" genes that appear in 3+ lists (e.g. Lepr, Esr1, Gal,
    Scgn, Otp in the original script) as primary classifiers. These are
    included only as supporting evidence when no better alternative exists.
4.  Separate neuronal subtypes from glia/vascular FIRST (easy), then
    resolve within each compartment (hard).

Three complementary annotation strategies
------------------------------------------
1. MarkerScoring        per-cell specificity-weighted gene scoring
2. CorrelationMapper    cluster-mean Pearson correlation vs reference
3. ThresholdAnnotator   single-gene hard thresholds (sanity check)

All methods write to adata.obs['cell_type'] (str) and
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


# ============================================================================
# MARKER GENE DICTIONARIES
# ============================================================================
#
# Genes annotated with specificity tier:
#   T1 = unique to this cell type in the MBH (best discriminators)
#   T2 = shared with 1 other type (good, but needs combinatorial logic)
#   T3 = shared with 2+ other types (supporting only; down-weighted)
#
# Genes that are NOT in the Xenium mBrain v1.1 base panel or a typical
# custom hypothalamus add-on are flagged with [panel?]. The scoring
# functions gracefully skip missing genes, so including them costs nothing.
# ============================================================================


# ---------------------------------------------------------------------------
# BROAD MARKERS: first-pass classification into ~15 major cell classes.
# Each list is ordered: most specific gene first.
# ---------------------------------------------------------------------------

MBH_MARKERS: dict[str, list[str]] = {

    # ── ARC NEURONAL ──────────────────────────────────────────────────────

    "Orexigenic neuron": [
        "Agrp",         # T1: unique identifier for AgRP/NPY population
        "Npy",          # T2: co-expressed; also sparse in other GABA types
        "Ghsr",         # T2: ghrelin receptor; enriched in AgRP neurons
    ],

    "Anorexigenic neuron": [
        "Pomc",         # T1: canonical POMC; unique in MBH neurons
        "Cartpt",       # T2: CART peptide; co-expressed in all POMC subtypes
        "Mc3r",         # T2: melanocortin-3 receptor; ARC POMC-enriched
    ],

    "Dopaminergic neuron": [
        "Slc6a3",       # T1: DAT; dopaminergic transporter (TIDA + A13)
        "Slc18a3",      # T2: vesicular monoamine transporter
    ],

    "Peptidergic neuron": [
        "Sst",          # T2: somatostatin; multiple ARC clusters
        "Nts",          # T2: neurotensin; Sst/Nts + MCH subsets
        "Cck",          # T2: cholecystokinin; broadly expressed neuropeptide
        "Crh",          # T2: CRH; stress axis marker
    ],

    "GABAergic neuron": [
        "Slc32a1",      # T1: VGAT; pan-inhibitory
        "Gad1",         # T2: GAD67
        "Gad2",         # T2: GAD65
    ],

    "Glutamatergic neuron": [
        "Slc17a7",      # T1: VGluT1
        "Adcyap1",      # T2: PACAP; enriched in VMH glutamatergic
        "Bdnf",         # T2: VMH-enriched neurotrophin
    ],

    "Hcrt/Orexin neuron": [
        "Hcrt",         # T1: hypocretin; unique to LHA orexin neurons
    ],

    "Neuroendocrine neuron": [
        "Oxt",          # T1: oxytocin (PVN)
        "Avp",          # T1: vasopressin (PVN/SCN)
    ],

    # ── EPENDYMAL / TANYCYTE ──────────────────────────────────────────────

    "Tanycyte": [
        "Gpr50",        # T1: most selective tanycyte marker (also in DMH neurons
                        #     but in scRNA-seq ARC context, marks tanycytes)
        "Mfsd2a",       # T2: tanycyte-enriched over ependymocytes
        "Lef1",         # T2: tanycyte Wnt pathway [panel?]
        "Notch1",       # T2: Notch signaling in tanycytes [panel?]
    ],

    # ── GLIA ──────────────────────────────────────────────────────────────

    "Astrocyte": [
        "Aqp4",         # T1: aquaporin-4; canonical (Campbell: Gfap+ cluster)
        "Gfap",         # T2: broadly astrocytic; also marks reactive state
        "Ntsr2",        # T1: hypothalamic astrocyte-enriched (HypoMap)
    ],

    "Oligodendrocyte": [
        "Opalin",       # T1: mature oligodendrocyte-specific
        "Sox10",        # T2: pan-OL lineage (includes OPC; but OPC has Pdgfra)
        "Gjc3",         # T1: gap junction in mature OLs [panel?]
    ],

    "OPC": [
        "Pdgfra",       # T1: canonical OPC (NG2 glia)
        "Cspg4",        # T2: NG2 proteoglycan; OPC + pericyte overlap
        "Gpr17",        # T2: transitional OPC-to-OL
    ],

    "Microglia": [
        "Siglech",      # T1: homeostatic microglia-specific
        "Spi1",         # T2: PU.1; myeloid TF
        "Trem2",        # T2: DAM/activated microglia
        "Cd68",         # T2: lysosomal; activated microglia
    ],

    # ── VASCULAR / STROMAL ────────────────────────────────────────────────

    "Endothelial": [
        "Cldn5",        # T1: claudin-5; tight junction (BBB endothelium)
        "Pecam1",       # T1: CD31; pan-endothelial
    ],

    "Pericyte": [
        "Acta2",        # T2: alpha-SMA; pericyte + VSMC
        "Carmn",        # T1: pericyte-enriched lncRNA (Campbell 2017)
    ],

    "VLMC": [
        "Col1a1",       # T1: collagen I; leptomeningeal fibroblast
        "Dcn",          # T1: decorin; VLMC-specific in brain
    ],
}


# ---------------------------------------------------------------------------
# SUBTYPE MARKERS: second-pass scoring WITHIN broad classes.
#
# Each subtype is resolved only among clusters already assigned to its
# parent broad class. This means the subtype genes only need to
# distinguish siblings, not all 50+ cell types. Lists are therefore
# very short (1-3 genes).
#
# Parent broad class is encoded in BROAD_TO_SUBTYPES below.
# ---------------------------------------------------------------------------

MBH_SUBTYPE_MARKERS: dict[str, list[str]] = {

    # ── ARC orexigenic ────────────────────────────────────────────────────
    "AgRP / NPY neurons": [
        "Agrp",                     # gold standard; Agrp+Npy = sufficient
        "Npy",
    ],

    # ── ARC anorexigenic ──────────────────────────────────────────────────
    "POMC neurons": [
        "Pomc",                     # defines the population
        "Cartpt",                   # co-expressed in all 3 POMC subtypes
        "Mc3r",                     # melanocortin-3 receptor; POMC autocrine
    ],
    "KNDy neurons": [
        "Tac2",                     # best proxy; Kiss1 absent from most panels
        "Scgn",                     # secretagogin; KNDy co-marker
    ],

    # ── ARC peptidergic subtypes ──────────────────────────────────────────
    "Somatostatin neurons": [
        "Sst",                      # canonical; 3 Sst+ clusters in ARC
        "Chodl",                    # Sst/Chodl co-expression pattern
    ],
    "CCK neurons": [
        "Cck",                      # cholecystokinin
        "Cckar",                    # autocrine CCK-A receptor
    ],
    "CRH / stress neurons": [
        "Crh",                      # CRH; DMH/PVN stress axis
        "Crhr2",                    # CRH receptor type 2
    ],
    "RFRP / NPVFergic neurons": [
        "Npvf",                     # T1: RFRP-3 precursor; highly DMH-specific
        "Brs3",                     # bombesin receptor subtype 3
    ],
    "Enkephalin neurons": [
        "Penk",                     # proenkephalin
        "Tac1",                     # substance P; co-expressed subset
    ],
    "NTS neurons": [
        "Nts",                      # neurotensin
    ],

    # ── ARC misc. named clusters (Campbell 2017) ──────────────────────────
    # These are real populations identified by Campbell et al. but depend
    # on genes that may not be in every Xenium panel. The scoring functions
    # will silently skip them if the genes are absent.
    "TRH / Cxcl12 neurons": [
        "Trh",                      # [panel?] thyrotropin-releasing hormone
        "Cxcl12",                   # [panel?] highest Lepr cluster in ARC
    ],
    "Pnoc neurons": [
        "Pnoc",                     # prepronociceptin; calorie-dense food response
    ],
    "Tbx19 neurons": [
        "Tbx19",                    # [panel?] RIP-Cre population
        "Nmu",                      # [panel?] neuromedin U
    ],

    # ── ARC dopaminergic subtypes ─────────────────────────────────────────
    "Dopaminergic neurons (TIDA)": [
        "Slc6a3",                   # DAT defines TIDA
    ],
    "ZI dopaminergic (A13)": [
        "Slc6a3",                   # shared with TIDA; resolved by spatial gating
        "Slc18a3",                  # vesicular transporter
    ],

    # ── VMH subtypes ──────────────────────────────────────────────────────
    "VMH-Esr1 neurons": [
        "Esr1",                     # estrogen receptor alpha; VMHvl
        "Pgr",                      # [panel?] progesterone receptor
    ],
    "VMH-Fezf1 neurons": [
        "Fezf1",                    # [panel?] central VMH; panic/autonomic
        "Adcyap1",                  # PACAP; co-expressed
    ],
    "VMH-Ucn3 neurons": [
        "Ucn3",                     # T1: urocortin-3; VMH-specific
        "Bdnf",                     # BDNF; VMH-enriched
    ],
    "VMH generic (glutamatergic)": [
        "Slc17a7",                  # VGluT1; pan-glutamatergic VMH
    ],

    # ── LHA subtypes ──────────────────────────────────────────────────────
    "Orexin / hypocretin neurons": [
        "Hcrt",                     # T1: unique to LHA
    ],
    "MCH neurons": [
        "Nts",                      # neurotensin; MCH co-expression
        "Lepr",                     # leptin receptor; MCH subset (Pmch absent)
    ],
    "Galanin neurons": [
        "Gal",                      # T1: galanin neuropeptide defines this type
    ],

    # ── DMH subtypes ──────────────────────────────────────────────────────
    "DMH Ppp1r17 neurons": [
        "Ppp1r17",                  # T1: DMH-specific (Tokizane 2024, Friedman 2021)
                                    #     glutamatergic; regulates food restriction,
                                    #     aging, and WAT function via sympathetic NS
    ],

    # ── ZI subtypes ───────────────────────────────────────────────────────
    "ZI GABAergic neurons": [
        "Synpr",                    # T1: synaptoporin; ZI-specific
        "Prdm12",                   # ZI transcription factor
    ],

    # ── PVN subtypes ──────────────────────────────────────────────────────
    "Oxytocin neurons": [
        "Oxt",                      # T1: unique neuropeptide
    ],
    "AVP neurons": [
        "Avp",                      # T1: unique neuropeptide
    ],

    # ── Broad neurotransmitter identity ───────────────────────────────────
    "GABAergic neurons": [
        "Slc32a1",                  # VGAT
        "Gad1",
        "Gad2",
    ],
    "Glutamatergic neurons": [
        "Slc17a7",                  # VGluT1
    ],

    # ── Tanycyte subtypes (Campbell 2017 gradient) ────────────────────────
    "Tanycytes (beta)": [
        "Gpr50",                    # beta-tanycyte selective (also DMH neurons,
                                    # but resolved by broad-class gating)
        "Mfsd2a",                   # tanycyte-enriched lipid transporter
    ],
    "Tanycytes (alpha) / ependymal": [
        "Slc2a1",                   # GLUT1; alpha-tanycyte
        "Notch1",                   # [panel?] Notch signaling; alpha
    ],

    # ── Glial subtypes ────────────────────────────────────────────────────
    "Astrocytes": [
        "Aqp4",
        "Gfap",
        "Ntsr2",
    ],
    "Oligodendrocytes": [
        "Opalin",
        "Sox10",
    ],
    "OPCs": [
        "Pdgfra",
        "Gpr17",
    ],
    "Homeostatic microglia": [
        "Siglech",                  # downregulated upon DAM activation
        "Spi1",
    ],
    "Activated / DAM microglia": [
        "Trem2",                    # classic DAM signature gene
        "Cd68",
        "Laptm5",
    ],

    # ── Vascular subtypes ─────────────────────────────────────────────────
    "Endothelial cells": [
        "Cldn5",
        "Pecam1",
    ],
    "Pericytes": [
        "Acta2",
        "Carmn",
    ],
    "VLMC": [
        "Col1a1",
        "Dcn",
    ],
}


# ---------------------------------------------------------------------------
# Maps each broad type to the subtypes it can resolve into.
# Only listed subtypes are candidates during the second scoring pass.
# ---------------------------------------------------------------------------

BROAD_TO_SUBTYPES: dict[str, list[str]] = {
    "Orexigenic neuron": [
        "AgRP / NPY neurons",
    ],
    "Anorexigenic neuron": [
        "POMC neurons",
        "KNDy neurons",
    ],
    "Dopaminergic neuron": [
        "Dopaminergic neurons (TIDA)",
        "ZI dopaminergic (A13)",
    ],
    "Peptidergic neuron": [
        "Somatostatin neurons",
        "CCK neurons",
        "CRH / stress neurons",
        "RFRP / NPVFergic neurons",
        "Enkephalin neurons",
        "NTS neurons",
        "Galanin neurons",
        "MCH neurons",
        "TRH / Cxcl12 neurons",
        "Pnoc neurons",
        "Tbx19 neurons",
    ],
    "GABAergic neuron": [
        "ZI GABAergic neurons",
        "GABAergic neurons",
    ],
    "Glutamatergic neuron": [
        "VMH-Esr1 neurons",
        "VMH-Fezf1 neurons",
        "VMH-Ucn3 neurons",
        "VMH generic (glutamatergic)",
        "DMH Ppp1r17 neurons",
        "Glutamatergic neurons",
    ],
    "Hcrt/Orexin neuron": [
        "Orexin / hypocretin neurons",
    ],
    "Neuroendocrine neuron": [
        "Oxytocin neurons",
        "AVP neurons",
    ],
    "Tanycyte": [
        "Tanycytes (beta)",
        "Tanycytes (alpha) / ependymal",
    ],
    "Astrocyte": [
        "Astrocytes",
    ],
    "Oligodendrocyte": [
        "Oligodendrocytes",
    ],
    "OPC": [
        "OPCs",
    ],
    "Microglia": [
        "Homeostatic microglia",
        "Activated / DAM microglia",
    ],
    "Endothelial": [
        "Endothelial cells",
    ],
    "Pericyte": [
        "Pericytes",
    ],
    "VLMC": [
        "VLMC",
    ],
}


# Alias for backward compatibility
BRAIN_MARKERS = MBH_MARKERS

# ============================================================================
# SPECIFICITY WEIGHTS
# ============================================================================

def compute_specificity_weights(
    markers: dict[str, list[str]],
) -> dict[str, float]:
    """
    Compute inverse-frequency weight for every gene across all marker lists.

    A gene appearing in N marker lists gets weight 1/N. This means a gene
    unique to one cell type (N=1) contributes its full z-score, while a
    gene shared across 3 types (N=3) contributes only one-third.

    Returns dict mapping gene name to weight in [0, 1].
    """
    gene_counts: dict[str, int] = {}
    for gene_list in markers.values():
        for gene in gene_list:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
    return {gene: 1.0 / count for gene, count in gene_counts.items()}


def compute_tier_weights(
    markers: dict[str, list[str]],
) -> dict[str, float]:
    """
    Alternative weighting: first gene in each list gets weight 1.0,
    second gets 0.8, third gets 0.6, etc., reflecting the curation order
    (most specific listed first).

    Multiplied with specificity weights for the final score.
    """
    position_weights: dict[str, float] = {}
    for gene_list in markers.values():
        for i, gene in enumerate(gene_list):
            w = max(0.4, 1.0 - 0.2 * i)
            # Keep the highest positional weight if gene appears in multiple lists
            if gene not in position_weights or w > position_weights[gene]:
                position_weights[gene] = w
    return position_weights


# ============================================================================
# Strategy 1: Specificity-weighted marker gene scoring
# ============================================================================

class MarkerScoring:
    """
    Score each cell against curated marker lists using specificity-weighted
    scoring, then assign the cell type with the highest score.

    Improvement over the original: genes shared across many marker lists
    are down-weighted so that truly specific genes drive the classification.

    Parameters
    ----------
    markers:
        Dict mapping cell-type name to list of marker genes.
    ctrl_size:
        Number of control genes per marker set (scanpy score_genes).
    min_score_delta:
        Minimum gap between top-1 and top-2 scores to assign a label.
    use_specificity_weights:
        If True, multiply each gene's contribution by 1/(number of
        marker lists it appears in).
    """

    def __init__(
        self,
        markers: Optional[dict[str, list[str]]] = None,
        ctrl_size: Optional[int] = None,
        min_score_delta: float = 0.05,
        use_specificity_weights: bool = True,
    ):
        self.markers = markers or MBH_MARKERS
        self._ctrl_size_override = ctrl_size
        self.min_score_delta = min_score_delta
        self.use_specificity_weights = use_specificity_weights

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

        # Compute specificity weights
        spec_weights = compute_specificity_weights(self.markers)

        score_keys = []
        for ct, gene_list in self.markers.items():
            available = [g for g in gene_list if g in adata.var_names]
            if len(available) < 1:
                logger.warning(
                    "Cell type '%s': no marker genes in panel; skipping.", ct
                )
                continue
            key = ct.replace(" ", "_").replace("/", "_") + "_score"
            sc.tl.score_genes(
                adata,
                gene_list=available,
                ctrl_size=max(1, min(ctrl_size, adata.n_vars - len(available) - 1)),
                score_name=key,
            )

            # Apply specificity weights if requested
            if self.use_specificity_weights and len(available) >= 2:
                raw_scores = adata.obs[key].values.copy()
                weights = np.array([spec_weights.get(g, 1.0) for g in available])
                mean_weight = weights.mean()
                # Scale the score by the mean weight of available markers
                # This penalizes types whose markers are shared with many others
                adata.obs[key] = raw_scores * mean_weight

            score_keys.append((ct, key))
            logger.debug(
                "Scored %d/%d genes for %s (mean specificity weight: %.2f)",
                len(available), len(gene_list), ct,
                np.mean([spec_weights.get(g, 1.0) for g in available]),
            )

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


# ============================================================================
# Strategy 2: Correlation mapper (reference atlas)
# ============================================================================

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


# ============================================================================
# Strategy 3: Threshold annotator
# ============================================================================

class ThresholdAnnotator:
    """
    Hard-threshold cell type assignment using single canonical markers.

    For each cell, iterate over rules in priority order. Assign the first
    cell type whose marker gene exceeds the threshold. Useful as a fast
    sanity check against the scoring methods.

    Priority order matters: more specific markers are tested first so that,
    e.g., Agrp+ cells are labelled "AgRP/NPY" before the broad "GABAergic"
    rule fires on Gad1.
    """

    # Rules ordered: most specific first, broad neurotransmitter last.
    # Gene names must match mouse MGI convention (e.g. Agrp, not AGRP).
    DEFAULT_RULES = [
        # -- Specific neuronal populations --
        ("AgRP/NPY neuron",     "Agrp",    0.5),
        ("POMC neuron",         "Pomc",    0.5),
        ("Orexin neuron",       "Hcrt",    0.3),
        ("Oxytocin neuron",     "Oxt",     0.3),
        ("AVP neuron",          "Avp",     0.3),
        ("Dopaminergic neuron", "Slc6a3",  0.3),

        # -- Glia / ependymal --
        ("Tanycyte",            "Gpr50",   0.3),
        ("Microglia",           "Siglech", 0.3),
        ("Microglia",           "Cd68",    0.5),
        ("OPC",                 "Pdgfra",  0.5),
        ("Oligodendrocyte",     "Opalin",  0.5),
        ("Oligodendrocyte",     "Sox10",   0.5),
        ("Astrocyte",           "Aqp4",    0.5),
        ("Astrocyte",           "Gfap",    0.5),

        # -- Vascular --
        ("Endothelial",         "Cldn5",   0.5),
        ("Pericyte",            "Carmn",   0.3),
        ("VLMC",                "Col1a1",  0.3),

        # -- Broad neurotransmitter (last resort) --
        ("Inhibitory Neuron",   "Gad1",    0.5),
        ("Inhibitory Neuron",   "Gad2",    0.5),
        ("Excitatory Neuron",   "Slc17a7", 0.5),
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


# ============================================================================
# Data-driven cluster annotation (recommended primary method)
# ============================================================================


def assign_labels_from_markers(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    broad_markers: Optional[dict] = None,
    subtype_markers: Optional[dict] = None,
    broad_to_subtypes: Optional[dict] = None,
    new_col: str = "cell_type",
    min_score: float = 0.0,
    fallback: str = "Unknown",
    use_specificity_weights: bool = True,
) -> ad.AnnData:
    """
    Data-driven cluster annotation: score each cluster's mean expression
    against MBH_MARKERS (broad types) then MBH_SUBTYPE_MARKERS (subtypes).

    This is the recommended approach for targeted spatial panels because it
    derives labels from the actual expression data.

    Workflow
    --------
    1.  Compute mean log-normalised expression per cluster.
    2.  Z-score each gene across clusters (removes broadly-expressed bias).
    3.  For each cluster, compute weighted sum of z-scores for each broad
        cell type.  Weights are 1/(number of marker lists the gene appears
        in), so unique markers count more.
    4.  Assign the broad type with the highest weighted score.
    5.  Log all scores for researcher verification.

    Note: only broad-level labels from MBH_MARKERS are used.  Subtype
    resolution is not reliable with targeted Xenium panels (~300 genes).

    Parameters
    ----------
    adata:
        AnnData post-clustering.
    cluster_key:
        Cluster column in adata.obs.
    broad_markers:
        Dict of broad cell type -> marker gene list.
    subtype_markers:
        Dict of subtype label -> marker gene list.
    broad_to_subtypes:
        Dict mapping broad type -> list of eligible subtype names.
    new_col:
        Output column name in adata.obs.
    min_score:
        Minimum score for a cluster to receive a label.
    fallback:
        Label for clusters that fail the min_score gate.
    use_specificity_weights:
        If True, weight each gene by 1/(times it appears across lists).

    Returns
    -------
    AnnData with .obs[new_col], .obs['cell_type_score'],
    and .uns['cluster_annotation_scores'].
    """
    broad_markers    = broad_markers    or MBH_MARKERS

    X    = _get_lognorm(adata)
    vn   = list(adata.var_names)
    vmap = {g: i for i, g in enumerate(vn)}

    clusters = sorted(
        adata.obs[cluster_key].astype(str).unique(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, x),
    )

    # Compute specificity weights across broad marker dict
    spec_weights = compute_specificity_weights(broad_markers) if use_specificity_weights else {}

    # ── Step 1: mean expression per cluster ──────────────────────────────
    cluster_means: dict[str, np.ndarray] = {}
    for cl in clusters:
        mask = adata.obs[cluster_key].astype(str) == cl
        sub = X[mask.values] if hasattr(mask, 'values') else X[mask]
        if sp.issparse(sub):
            mu = np.asarray(sub.mean(axis=0)).ravel()
        else:
            mu = sub.mean(axis=0)
        cluster_means[cl] = mu

    # ── Step 2: z-score across clusters ──────────────────────────────────
    cl_list   = list(clusters)
    expr_mat  = np.stack([cluster_means[cl] for cl in cl_list], axis=0)
    gene_mean = expr_mat.mean(axis=0)
    gene_std  = expr_mat.std(axis=0) + 1e-6
    zscore_mat = (expr_mat - gene_mean) / gene_std
    cl_zscore  = {cl: zscore_mat[i] for i, cl in enumerate(cl_list)}

    def _score(cluster_id: str, gene_list: list[str]) -> float:
        """Weighted sum of z-scores for available marker genes."""
        z = cl_zscore[cluster_id]
        total = 0.0
        for g in gene_list:
            if g in vmap:
                w = spec_weights.get(g, 1.0) if use_specificity_weights else 1.0
                total += z[vmap[g]] * w
        return total

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

    # Use broad-level labels directly (appropriate for targeted Xenium panels
    # with ~300 genes — subtype resolution is not reliable at this depth).
    final_labels = dict(broad_labels)
    final_scores = dict(broad_scores)

    # ── Log assignments ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Cluster annotation (specificity-weighted marker scoring):")
    logger.info("  %-6s  %-40s  %s", "ID", "Label", "Score")
    logger.info("  " + "-" * 64)

    # Also log top-3 broad scores per cluster for transparency
    for cl in clusters:
        label = final_labels.get(cl, fallback)
        score = final_scores.get(cl, 0.0)
        row   = score_df.loc[cl] if cl in score_df.index else {}
        top3  = sorted(broad_types, key=lambda bt: row.get(bt, 0), reverse=True)[:3]
        top3_str = ", ".join(f"{bt.split()[0]}={row.get(bt,0):.2f}" for bt in top3)
        logger.info("  %-6s  %-40s  %.3f  [top3: %s]", cl, label, score, top3_str)

    logger.info("=" * 70)
    logger.info(
        "IMPORTANT: Review the scores above and the dotplot before proceeding. "
        "Override any cluster by passing a custom label_map."
    )

    # ── Write to adata ──────────────────────────────────────────────────
    obs_labels = adata.obs[cluster_key].astype(str).map(final_labels).fillna(fallback)
    obs_scores = adata.obs[cluster_key].astype(str).map(final_scores).fillna(0.0)

    adata.obs[new_col]             = pd.Categorical(obs_labels)
    adata.obs["cell_type_score"]   = obs_scores.astype(np.float32)
    adata.uns["cluster_annotation_scores"] = score_df
    adata.uns["cluster_label_map"] = final_labels   # store for Supp Table 1

    n_unknown = (obs_labels == fallback).sum()
    if n_unknown > 0:
        logger.warning(
            "%d cells could not be annotated (labelled '%s'). "
            "Inspect the dotplot and provide a manual label_map.",
            n_unknown, fallback,
        )
    _log_type_counts(adata)
    return adata

# ============================================================================
# Unified dispatcher
# ============================================================================

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
    """
    if method == "marker_scoring":
        annotator = MarkerScoring(
            markers=markers or MBH_MARKERS,
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


# ============================================================================
# Panel audit utility
# ============================================================================

def audit_panel_coverage(
    adata: ad.AnnData,
    markers: Optional[dict[str, list[str]]] = None,
    subtype_markers: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Check which marker genes are present in the panel and which are missing.

    Returns a DataFrame with columns:
        cell_type, gene, in_panel, tier (broad or subtype)

    Use this BEFORE annotation to verify your panel covers the markers
    needed for each population.
    """
    markers = markers or MBH_MARKERS
    subtype_markers = subtype_markers or MBH_SUBTYPE_MARKERS
    panel_genes = set(adata.var_names)

    rows = []
    for ct, genes in markers.items():
        for g in genes:
            rows.append({
                "cell_type": ct,
                "gene": g,
                "in_panel": g in panel_genes,
                "tier": "broad",
            })
    for ct, genes in subtype_markers.items():
        for g in genes:
            rows.append({
                "cell_type": ct,
                "gene": g,
                "in_panel": g in panel_genes,
                "tier": "subtype",
            })

    df = pd.DataFrame(rows)

    # Summary
    n_total = df["gene"].nunique()
    n_found = df.loc[df["in_panel"], "gene"].nunique()
    logger.info(
        "Panel coverage: %d / %d unique marker genes found (%.0f%%)",
        n_found, n_total, 100 * n_found / n_total,
    )

    # Per-type coverage
    for ct in df["cell_type"].unique():
        sub = df[df["cell_type"] == ct]
        n = sub["in_panel"].sum()
        total = len(sub)
        missing = sub.loc[~sub["in_panel"], "gene"].tolist()
        if missing:
            logger.warning(
                "  %-35s  %d/%d genes found  (missing: %s)",
                ct, n, total, ", ".join(missing),
            )

    return df


# ============================================================================
# Helpers
# ============================================================================

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
        logger.info("  %-35s %5d  (%.1f%%)", ct, n, 100 * n / adata.n_obs)


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
