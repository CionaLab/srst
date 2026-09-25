# %% [markdown]
# Stage-by-stage clusters of the pooled Cao, Sharma and Winkley cells, linked
# by optimal transport (moscot) from the 64-cell stage to larva. Mid-gastrula
# clusters are anchored to the ANISEED stage-12 neural plate map (pass_02.tsv)
# and checked against Winkley's labels at midG, backward to c64 and iniG, and
# forward against Cao's tissue labels. The whole process runs twice: on all
# cells, then on the clusters that the transport network places upstream or
# downstream of the midG neural plate clusters. Genes without ANISEED in situ
# data are predicted per midG cell and its descendants at midG, earN and latN
# from the per-stage DE and the transport. Analysis functions return
# objects; analyze() runs them in order and writes each step's tables and
# figures as soon as that step finishes.

# %% Setup
import re
from collections.abc import Iterable
from typing import Any

import anndata as ad
import decoupler as dc
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import scanpy as sc
import scvi
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from moscot.problems.time import TemporalProblem

scvi.settings.seed = 0

PREFIX = "integrated"
GENOME = "ky21"
OUT = f"{PREFIX}_{GENOME}_npisc"
SCVI_MODEL = "integrated_ky21_SCVI"
MARKERS = "npisc/pass_02.tsv"
GENE_MAP = "npisc/kh2012_ky2021_map.tsv"
HOMOLOGS = "npisc/ky2021_swissprot_map.csv"
STAGE_MAPS = {
    "midG": "npisc/mid_gastrula.geojson",
    "earN": "npisc/early_neurula.geojson",
    "latN": "npisc/late_neurula.geojson",
}
MARKER_STAGES = ["midG"]

BATCH_KEY = "source"
STAGE_KEY = "stage"
LATENT_KEY = "X_scVI"
CAO = "cao2019"
WINKLEY = "winkley2021"
WINKLEY_LABEL_KEY = "winkley_celltype"
CAO_TISSUE_KEY = "cao_tissue_type"
NODE_TISSUE_KEY = "cao_tissue_type_knn"
NEURAL_LABEL = "nervous system"

CHAIN = [
    "c64",
    "iniG",
    "midG",
    "earN",
    "latN",
    "iniT",
    "earT",
    "midT",
    "latTI",
    "latTII",
    "larva",
]
ANCHOR_STAGE = "midG"
BENCH_GEN = {
    "c64": 7,
    "iniG": 8,
}

RESOLUTION = 2.0
MIN_CLUSTER = 20
DE_DELTA = 0.25
DE_FDR = 0.05
IS_DE = f"is_de_fdr_{DE_FDR}"
MIN_DE_CELLS = 10
TMIN = 3
MAX_PADJ = 0.05
MIN_MARGIN = 1.0
EPSILON = 1e-3
TAU_A = 0.95
OT_BATCH = 1024
MIN_EDGE = 0.05
MIN_SHOW = 0.05
PLOT_MIN_EDGE = 0.1
IDENTITY_GEN = {
    "midG": 9,
    "earN": 10,
    "latN": 10,
}
LINEAGE_MIN = 0.5
# ANISEED stage labels in MARKERS for the internal stage names
EXPRESSION_STAGES = {
    "midG": "Stage 12 (mid gastrula)",
    "earN": "Stage 14 (early neurula)",
    "latN": "Stage 16 (late neurula)",
}
EXPRESSED_MIN = 0.5
SPECIFIC_MIN = 1.0
TOP_GENES = 3
MAP_PER_CLONE = 1
MAP_GENES: list[str] = []
DPI = 300
METRICS = [
    "precision",
    "recall",
    "jaccard",
    "bits",
    "score",
    "gain",
]
TISSUE_COLORS = {
    "nervous system": "#2a78d6",
    "epidermis": "#eb6834",
    "mesenchyme": "#1baf7a",
    "endoderm": "#eda100",
    "notochord": "#e87ba4",
    "muscle & heart": "#008300",
    "germ": "#4a3aa7",
}
OTHER_COLOR = "#b7b6b0"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
]
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging",
    [
        "#184f95",
        "#86b6ef",
        "#f0efec",
        "#ef9a99",
        "#b8302f",
    ],
)

NP_GRID = {
    "a9.40": ("VI", 1),
    "a9.36": ("VI", 2),
    "a9.52": ("VI", 3),
    "a9.39": ("V", 1),
    "a9.35": ("V", 2),
    "a9.51": ("V", 3),
    "a9.38": ("IV", 1),
    "a9.34": ("IV", 2),
    "a9.50": ("IV", 3),
    "a9.37": ("III", 1),
    "a9.33": ("III", 2),
    "a9.49": ("III", 3),
    "A9.14": ("II", 1),
    "A9.16": ("II", 2),
    "A9.30": ("II", 3),
    "A9.32": ("II", 4),
    "A9.13": ("I", 1),
    "A9.15": ("I", 2),
    "A9.29": ("I", 3),
    "A9.31": ("I", 4),
}
WINKLEY_NP_RE = re.compile(r"^NP \((A|a)\) ([\d/]+):([IV/]+)$")
WINKLEY_EARLY = {
    "Neural (A)": "A7.4/A7.8",
    "Neural (a)": "a7.9/a7.10/a7.13",
    "Neural (b)": "b7.9/b7.10",
    "Neural (A) C1/2": "A8.7/A8.8",
    "Neural (A) C3/4": "A8.15/A8.16",
}
BLASTOMERE_RE = re.compile(r"\b([AaBb])(\d+)\.(\d+)\b")
GRID_ROWS = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
]


# %% Helpers
def to_generation(label: str, gen: int) -> set[str]:
    """Move every blastomere named in a label to one generation: the ancestor
    of a later cell, all descendants of an earlier one.

    :param label: Text naming blastomeres, e.g. "A9.15/A9.16".
    :type label: str
    :param gen: Target generation.
    :type gen: int
    :returns: Blastomere names at generation gen.
    :rtype: set[str]
    """
    out = set()
    for line, g, k in BLASTOMERE_RE.findall(str(label)):
        g, k = int(g), int(k)
        ks = {k}
        while g > gen:
            ks, g = {(x + 1) // 2 for x in ks}, g - 1
        while g < gen:
            ks, g = {y for x in ks for y in (2 * x - 1, 2 * x)}, g + 1
        out |= {f"{line}{gen}.{x}" for x in ks}
    return out


def winkley_blastomeres(label: str) -> set[str]:
    """Find the NP_GRID cells covered by a Winkley "NP (A|a) columns:rows"
    grid label.

    :param label: Winkley cell type label.
    :type label: str
    :returns: NP_GRID cells in the label's rows and columns; empty for any
        other label.
    :rtype: set[str]
    """
    m = WINKLEY_NP_RE.match(str(label).strip())
    if not m:
        return set()
    cols = {int(c) for c in m.group(2).split("/")}
    rows = set(m.group(3).split("/"))
    return {
        b
        for b, (r, c) in NP_GRID.items()
        if b[0] == m.group(1) and r in rows and c in cols
    }


def set_scores(
    pick: set[str],
    truth: set[str],
    n: int = len(NP_GRID),
) -> dict[str, float]:
    """Score a picked blastomere set against a true set.

    :param pick: Predicted blastomeres.
    :type pick: set[str]
    :param truth: True blastomeres; must not be empty.
    :type truth: set[str]
    :param n: Number of cells on the plate.
    :type n: int
    :returns: precision, recall, jaccard, bits of the pick (log2 n/|pick|),
        score (precision x bits / log2 n) and gain (precision x bits minus the
        bits of the truth).
    :rtype: dict[str, float]
    """
    hit = len(pick & truth)
    truth_bits = np.log2(n / len(truth))
    if not pick:
        return {
            "precision": np.nan,
            "recall": 0.0,
            "jaccard": 0.0,
            "bits": 0.0,
            "score": 0.0,
            "gain": -truth_bits,
        }
    precision = hit / len(pick)
    bits = np.log2(n / len(pick))
    return {
        "precision": precision,
        "recall": hit / len(truth),
        "jaccard": hit / len(pick | truth),
        "bits": bits,
        "score": precision * bits / np.log2(n),
        "gain": precision * bits - truth_bits,
    }


def num(node: str) -> int:
    """Read the Leiden number of a "{stage}_{leiden}" cluster name.

    :param node: Cluster name, e.g. "midG_12".
    :type node: str
    :returns: The Leiden number.
    :rtype: int
    """
    return int(node.rsplit("_", 1)[-1])


def labels_of(adata: ad.AnnData, key: str) -> np.ndarray:
    """Read an obs column as strings.

    :param adata: Cells.
    :type adata: anndata.AnnData
    :param key: obs column.
    :type key: str
    :returns: One string per cell, "" where missing.
    :rtype: numpy.ndarray
    """
    return adata.obs[key].astype(object).fillna("").astype(str).to_numpy()


def share_list(e: pd.DataFrame, key: str, share: str) -> pd.Series:
    """Format each cluster's children or parents as text.

    :param e: Transport edges.
    :type e: pandas.DataFrame
    :param key: "from" for children, "to" for parents.
    :type key: str
    :param share: Share column, "fwd" or "bwd".
    :type share: str
    :returns: Per cluster, "cluster:share" pairs with a share of MIN_EDGE or
        more, largest first.
    :rtype: pandas.Series
    """
    e = e[e[share] >= MIN_EDGE].sort_values(share, ascending=False)
    other = "to" if key == "from" else "from"

    def pairs(g: pd.DataFrame) -> str:
        return ",".join(f"{n}:{v:.2f}" for n, v in zip(g[other], g[share]))

    return e.groupby(key)[[other, share]].apply(pairs)


def plain(v: object) -> object:
    """Convert a numpy scalar to a Python value.

    :param v: Any value.
    :type v: object
    :returns: The Python value, or v unchanged when it is not a numpy scalar.
    :rtype: object
    """
    return v.item() if hasattr(v, "item") else v


def edges_between(
    edges: pd.DataFrame,
    sources: list[str],
    targets: list[str],
) -> pd.DataFrame:
    """Select the edges from some clusters to others.

    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :param sources: Source clusters.
    :type sources: list[str]
    :param targets: Target clusters.
    :type targets: list[str]
    :returns: Edges from any source to any target.
    :rtype: pandas.DataFrame
    """
    keep = edges["from"].isin(sources) & edges["to"].isin(targets)
    return edges[keep]


def early_blastomeres(label: str, gen: int) -> set[str]:
    """Find the blastomeres of a Winkley label at one generation, reading the
    early neural labels through WINKLEY_EARLY.

    :param label: Winkley cell type label.
    :type label: str
    :param gen: Target generation.
    :type gen: int
    :returns: Blastomere names at generation gen.
    :rtype: set[str]
    """
    return to_generation(WINKLEY_EARLY.get(label, label), gen)


def fmt(weights: dict[str, float], keep: set[str] | None = None) -> str:
    """Format weights as "name:weight" text, largest first.

    :param weights: Weight of each name.
    :type weights: dict[str, float]
    :param keep: Names to show; None shows weights of MIN_SHOW or more.
    :type keep: set[str] or None
    :returns: Comma-separated pairs.
    :rtype: str
    """
    items = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    if keep is None:
        items = [(k, v) for k, v in items if v >= MIN_SHOW]
    else:
        items = [(k, v) for k, v in items if k in keep]
    return ",".join(f"{k}:{v:.2f}" for k, v in items)


def blastomere_key(b: str) -> tuple[str, int]:
    """Sort blastomere names by line, then cell number.

    :param b: Blastomere name, e.g. "a9.49".
    :type b: str
    :returns: Line letter and cell number.
    :rtype: tuple[str, int]
    """
    return b[0], int(b.split(".")[1])


def grid_key(b: str) -> tuple[int, int]:
    """Sort NP_GRID cells by grid row I to VI, then column.

    :param b: NP_GRID cell name.
    :type b: str
    :returns: Row index and column.
    :rtype: tuple[int, int]
    """
    row, col = NP_GRID[b]
    return GRID_ROWS.index(row), col


NP_ORDER = sorted(NP_GRID, key=grid_key)


def generation_of(b: str) -> int:
    """Read the generation of a blastomere name.

    :param b: Blastomere name, e.g. "A10.29".
    :type b: str
    :returns: The generation, 10 for A10.29.
    :rtype: int
    """
    return int(b.split(".")[0][1:])


def covered(name: str, cells: list[str]) -> bool:
    """Check whether a blastomere is one of some cells, or an ancestor or
    descendant of one.

    :param name: Blastomere to check.
    :type name: str
    :param cells: Blastomere names.
    :type cells: list[str]
    :returns: True when covered.
    :rtype: bool
    """
    gen = generation_of(name)
    return any(name in to_generation(c, gen) for c in cells)


def map_cells(
    b: str,
    st: str,
    stage_maps: dict[str, gpd.GeoDataFrame],
) -> set[str]:
    """Find the cells on a stage map that are a blastomere or descend from it.

    :param b: Blastomere name.
    :type b: str
    :param st: Stage.
    :type st: str
    :param stage_maps: Neural plate maps from load_stage_maps.
    :type stage_maps: dict[str, geopandas.GeoDataFrame]
    :returns: Map cells matching b; b's cells at IDENTITY_GEN[st] when the
        stage has no map or none match.
    :rtype: set[str]
    """
    m = stage_maps.get(st)
    if m is not None:
        g = generation_of(b)
        cells = set()
        for n in m["name"]:
            if generation_of(n) >= g and b in to_generation(n, g):
                cells.add(n)
        if cells:
            return cells
    return to_generation(b, IDENTITY_GEN[st])


def join_names(names: Iterable[str]) -> str:
    """Join blastomere names in line and cell-number order.

    :param names: Blastomere names.
    :type names: iterable[str]
    :returns: Names joined by "/".
    :rtype: str
    """
    return "/".join(sorted(names, key=blastomere_key))


def auroc(score: npt.ArrayLike, positive: npt.ArrayLike) -> float:
    """Compute the area under the ROC curve, ties counting half.

    :param score: Score of each item.
    :type score: array-like
    :param positive: Whether each item is positive.
    :type positive: array-like of bool
    :returns: Chance that a positive item outscores a negative one; NaN unless
        both classes are present.
    :rtype: float
    """
    positive = np.asarray(positive, dtype=bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if not n1 or not n0:
        return np.nan
    ranks = pd.Series(np.asarray(score, dtype=float)).rank().to_numpy()
    return (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def short_name(name: str) -> str:
    """Drop the parenthesised parts of a gene name.

    :param name: Gene name, e.g. "Ptf1a-r (PTF1A)".
    :type name: str
    :returns: The name without them, e.g. "Ptf1a-r".
    :rtype: str
    """
    return re.sub(r"\s*\([^()]*\)", "", str(name)).strip()


def gene_label(row: pd.Series | dict) -> str:
    """Label a gene for plots.

    :param row: Row with "gene" and optionally "gene_name".
    :type row: pandas.Series or dict
    :returns: short_name of the gene name, or the KY21 ID when there is no
        name.
    :rtype: str
    """
    name = row.get("gene_name")
    name = short_name(name) if isinstance(name, str) and name else ""
    return name or row["gene"]


def file_safe(text: str) -> str:
    """Make text usable in a file name.

    :param text: Any text.
    :type text: str
    :returns: The text with runs of other characters than letters, digits,
        ".", "_" and "-" replaced by "_".
    :rtype: str
    """
    return re.sub(r"[^\w.-]+", "_", str(text))


def style(ax: Axes, grid_axis: str | None = "y") -> None:
    """Style an axis: surface background, no top or right spine, muted ticks
    and a hairline grid.

    :param ax: Axis to style.
    :type ax: matplotlib.axes.Axes
    :param grid_axis: "x", "y", "both", or None for no grid.
    :type grid_axis: str or None
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig: Figure, out: str, name: str) -> None:
    """Save a figure as {out}_{name}.png at DPI and close it.

    :param fig: Figure to save.
    :type fig: matplotlib.figure.Figure
    :param out: Output file prefix.
    :type out: str
    :param name: File name suffix.
    :type name: str
    """
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(f"{out}_{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    table: pd.DataFrame,
    out: str,
    name: str,
    **kwargs: Any,
) -> None:
    """Write a table as {out}_{name}.csv, or .tsv when sep is a tab.

    :param table: Table to write.
    :type table: pandas.DataFrame
    :param out: Output file prefix.
    :type out: str
    :param name: File name suffix.
    :type name: str
    :param kwargs: Passed to DataFrame.to_csv.
    :type kwargs: dict
    """
    ext = "tsv" if kwargs.get("sep") == "\t" else "csv"
    table.to_csv(f"{out}_{name}.{ext}", **kwargs)


# %% Inputs
def load_territory_net() -> pd.DataFrame:
    """Read the MARKER_STAGES rows of MARKERS as a decoupler network.
    Territories with the same gene set become one source named like
    "a9.35/a9.36/a9.39/a9.40".

    :returns: Columns "source" (territory), "target" (KY21 gene) and "weight"
        (1).
    :rtype: pandas.DataFrame
    :raises ValueError: A MARKER_STAGES label is not in MARKERS.
    """
    kh2ky = pd.read_csv(GENE_MAP, sep="\t").set_index("KH2012")["KY2021"]
    mk = pd.read_csv(MARKERS, sep="\t")
    labels = [EXPRESSION_STAGES[st] for st in MARKER_STAGES]
    missing = sorted(set(labels) - set(mk["Stage"]))
    if missing:
        raise ValueError(
            f"{MARKERS} has no rows for {missing}; "
            f"its stages are {sorted(mk['Stage'].unique())}",
        )
    mk = mk[mk["Stage"].isin(labels)].copy()
    mk["gene"] = mk["Gene"].str.extract(r"KH2012:(\S+)")[0].map(kh2ky)
    mk["territory"] = mk["Territory_eq"].str.rstrip("*")
    sets = mk.groupby("territory")["gene"].apply(frozenset)
    merged = sets.groupby(sets).transform(lambda t: "/".join(sorted(t.index)))
    print(
        "territories with identical markers:",
        sorted(set(merged) - set(sets.index)),
    )
    return (
        pd.DataFrame({"source": merged.reindex(sets.index), "target": sets})
        .explode("target")
        .drop_duplicates()
        .assign(weight=1.0)
    )


def load_stage_markers() -> pd.DataFrame:
    """Read the MARKER_STAGES rows of MARKERS for the expression benchmark.

    :returns: One row per stage and KY21 gene: "aniseed" (ANISEED gene name),
        "truth" (NP_GRID cells at generation 9 covering the annotated cells)
        and "cells" (annotated cells).
    :rtype: pandas.DataFrame
    :raises ValueError: A MARKER_STAGES label is not in MARKERS.
    """
    kh2ky = pd.read_csv(GENE_MAP, sep="\t").set_index("KH2012")["KY2021"]
    gen = IDENTITY_GEN[ANCHOR_STAGE]
    mk = pd.read_csv(MARKERS, sep="\t")
    stage_of = {EXPRESSION_STAGES[st]: st for st in MARKER_STAGES}
    missing = sorted(set(stage_of) - set(mk["Stage"]))
    if missing:
        raise ValueError(f"{MARKERS} has no rows for {missing}")
    mk = mk[mk["Stage"].isin(stage_of)].copy()
    mk["stage"] = mk["Stage"].map(stage_of)
    mk["gene"] = mk["Gene"].str.extract(r"KH2012:(\S+)")[0].map(kh2ky)
    mk["aniseed"] = mk["Gene"].str.extract(r"\(([^()]*)\)\s*$")[0]
    mk["cell"] = mk["Territory_eq"].str.rstrip("*")
    mk["clone"] = mk["cell"].map(
        lambda t: sorted(to_generation(t, gen) & set(NP_GRID)),
    )
    mk = mk.explode("clone").dropna(subset=["gene", "clone"])
    markers = (
        mk.groupby(["stage", "gene"])
        .agg(
            aniseed=("aniseed", "first"),
            truth=("clone", set),
            cells=("cell", set),
        )
        .reset_index()
    )
    print(
        "ANISEED genes per stage:",
        markers["stage"].value_counts().to_dict(),
    )
    return markers


def load_known_genes() -> set[str]:
    """Collect every gene in MARKERS, at any stage.

    :returns: KY21 IDs of the genes with ANISEED in situ data.
    :rtype: set[str]
    """
    kh2ky = pd.read_csv(GENE_MAP, sep="\t").set_index("KH2012")["KY2021"]
    mk = pd.read_csv(MARKERS, sep="\t")
    kh = mk["Gene"].str.extract(r"KH2012:(\S+)")[0]
    known = set(kh.map(kh2ky).dropna())
    print(f"{len(known)} genes with ANISEED in situ data")
    return known


def load_homologs() -> pd.DataFrame:
    """Read the SwissProt homolog of each KY21 gene from HOMOLOGS, with
    {ECO:...} evidence tags removed.

    :returns: Indexed by KY21 ID, columns "KH2012", "uniprot" and "homolog".
    :rtype: pandas.DataFrame
    """
    homologs = pd.read_csv(HOMOLOGS).set_index("KY2021")
    homologs["homolog"] = homologs["fullname"].str.replace(
        r"\s*\{ECO:[^}]*\}",
        "",
        regex=True,
    )
    return homologs[["KH2012", "uniprot", "homolog"]]


def load_stage_maps() -> dict[str, gpd.GeoDataFrame]:
    """Read the neural plate polygons of each stage in STAGE_MAPS, with the
    "*" of right-side names dropped.

    :returns: Per stage, polygons with "name" and "clone", the generation-9
        ancestor of each cell.
    :rtype: dict[str, geopandas.GeoDataFrame]
    """
    gen = IDENTITY_GEN[ANCHOR_STAGE]
    maps = {}
    for st, path in STAGE_MAPS.items():
        m = gpd.read_file(path)
        m["name"] = m["name"].str.rstrip("*")
        clones = [join_names(to_generation(b, gen)) for b in m["name"]]
        m["clone"] = clones
        maps[st] = m
    return maps


# %% Clusters per stage
def present_stages(adata: ad.AnnData) -> list[str]:
    """List the CHAIN stages that have cells.

    :param adata: Cells.
    :type adata: anndata.AnnData
    :returns: Stages in CHAIN order.
    :rtype: list[str]
    """
    return [
        s
        for s in adata.obs[STAGE_KEY].cat.categories
        if s in CHAIN and (adata.obs[STAGE_KEY] == s).any()
    ]


def cluster_stages(adata: ad.AnnData, stages: list[str]) -> np.ndarray:
    """Run neighbours on LATENT_KEY, Leiden at RESOLUTION and UMAP on each
    stage separately. Sets obs "stage_leiden", obs "stage_cluster"
    ("{stage}_{leiden}", missing for clusters under MIN_CLUSTER cells) and
    obsm "X_umap_stage".

    :param adata: Cells; modified in place.
    :type adata: anndata.AnnData
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :returns: stage_cluster of each cell, "" where missing.
    :rtype: numpy.ndarray
    """
    leiden = pd.Series(pd.NA, index=adata.obs_names, dtype=object)
    umap = np.full((adata.n_obs, 2), np.nan)
    obs_stage = labels_of(adata, STAGE_KEY)
    for st in stages:
        idx = np.flatnonzero(obs_stage == st)
        adata_sub = adata[idx].copy()
        sc.pp.neighbors(adata_sub, use_rep=LATENT_KEY)
        sc.tl.leiden(
            adata_sub,
            resolution=RESOLUTION,
            flavor="igraph",
            n_iterations=-1,
            directed=False,
        )
        sc.tl.umap(adata_sub)
        leiden.iloc[idx] = adata_sub.obs["leiden"].astype(str).to_numpy()
        umap[idx] = adata_sub.obsm["X_umap"]
    stage = pd.Series(obs_stage, index=adata.obs_names)
    name = stage + "_" + leiden.astype(str)
    adata.obs["stage_leiden"] = leiden
    adata.obs["stage_cluster"] = name.where(
        name.map(name.value_counts()) >= MIN_CLUSTER,
    )
    adata.obsm["X_umap_stage"] = umap
    print(
        adata.obs.groupby(STAGE_KEY, observed=True)["stage_cluster"]
        .nunique()
        .reindex(stages)
        .to_string(),
    )
    return labels_of(adata, "stage_cluster")


# %% Differential expression
def differential_expression(
    adata: ad.AnnData,
    stages: list[str],
    homologs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Test each cluster against the other clusters of its stage with the
    saved scVI model (change mode, DE_DELTA, batch corrected), then repeat
    with Wilcoxon within each source that has two or more clusters of
    MIN_DE_CELLS cells. A gene is replicated when scVI calls it and every
    tested source (at least two where available) agrees in sign at FDR DE_FDR.

    :param adata: Cells with stage_cluster.
    :type adata: anndata.AnnData
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param homologs: Homologs from load_homologs.
    :type homologs: pandas.DataFrame
    :returns: Per stage, one row per cluster and gene with the scVI
        statistics, per-source lfc and padj, sources_tested, sources_agree,
        replicated and homologs.
    :rtype: dict[str, pandas.DataFrame]
    """
    cluster = labels_of(adata, "stage_cluster")
    obs_stage = labels_of(adata, STAGE_KEY)
    sources = sorted(set(labels_of(adata, BATCH_KEY)))
    results = {}
    for st in stages:
        cells = adata[(obs_stage == st) & (cluster != "")].copy()
        cells.obs["stage_cluster"] = pd.Categorical(
            cells.obs["stage_cluster"].astype(str),
        )
        if cells.obs["stage_cluster"].nunique() < 2:
            print(f"{st}: fewer than two clusters, skipped")
            continue
        model = scvi.model.SCVI.load(SCVI_MODEL, adata=cells)
        de = model.differential_expression(
            groupby="stage_cluster",
            batch_correction=True,
            mode="change",
            delta=DE_DELTA,
            fdr_target=DE_FDR,
        )
        de = (
            de.rename_axis("gene")
            .reset_index()
            .rename(columns={"group1": "stage_cluster"})
        )
        de = de[
            [
                "stage_cluster",
                "gene",
                "lfc_mean",
                "lfc_std",
                "proba_de",
                "bayes_factor",
                IS_DE,
                "non_zeros_proportion1",
                "non_zeros_proportion2",
                "scale1",
                "scale2",
            ]
        ]

        present = []
        for src in sources:
            s_cells = cells[cells.obs[BATCH_KEY].astype(str) == src].copy()
            n = s_cells.obs["stage_cluster"].value_counts()
            groups = sorted(n.index[n >= MIN_DE_CELLS])
            if len(groups) < 2:
                continue
            present.append(src)
            sc.tl.rank_genes_groups(
                s_cells,
                "stage_cluster",
                groups=groups,
                reference="rest",
                method="wilcoxon",
                use_raw=False,
            )
            r = sc.get.rank_genes_groups_df(s_cells, group=groups).rename(
                columns={
                    "group": "stage_cluster",
                    "names": "gene",
                },
            )
            r = r[
                [
                    "stage_cluster",
                    "gene",
                    "logfoldchanges",
                    "pvals_adj",
                ]
            ]
            r = r.rename(
                columns={
                    "logfoldchanges": f"lfc_{src}",
                    "pvals_adj": f"padj_{src}",
                },
            )
            de = de.merge(r, on=["stage_cluster", "gene"], how="left")
        tested = de[[f"padj_{src}" for src in present]].notna()
        agree = pd.concat(
            [
                (np.sign(de[f"lfc_{src}"]) == np.sign(de["lfc_mean"]))
                & (de[f"padj_{src}"] < DE_FDR)
                for src in present
            ],
            axis=1,
        )
        de["sources_tested"] = tested.sum(axis=1)
        de["sources_agree"] = agree.sum(axis=1)
        de["replicated"] = (
            de[IS_DE]
            & (de["sources_tested"] >= min(2, len(present)))
            & (de["sources_agree"] == de["sources_tested"])
        )
        de = de.join(homologs, on="gene")
        front = ["stage_cluster", "gene", "KH2012", "uniprot", "homolog"]
        de = de[front + [c for c in de if c not in front]]
        if "gene_name" in adata.var:
            de.insert(2, "gene_name", de["gene"].map(adata.var["gene_name"]))
        de["order"] = de["stage_cluster"].map(num)
        de = de.sort_values(
            ["order", IS_DE, "lfc_mean"],
            ascending=[True, False, False],
        ).drop(columns="order")
        print(f"{st}: per-source check in {present}")
        print(
            de.assign(up=de[IS_DE] & (de["lfc_mean"] > 0))
            .groupby("stage_cluster")[[IS_DE, "up", "replicated"]]
            .sum()
            .to_string(),
        )
        results[st] = de
    return results


# %% Optimal transport
def solve_transport(
    adata: ad.AnnData,
    stages: list[str],
) -> tuple[TemporalProblem, ad.AnnData]:
    """Solve one moscot TemporalProblem across the stages on LATENT_KEY with
    EPSILON and TAU_A.

    :param adata: Cells with stage_cluster.
    :type adata: anndata.AnnData
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :returns: The solved problem and its copy of adata, whose obs has "time"
        (stage index) and "node" (stage cluster or "none").
    :rtype: tuple[moscot.problems.time.TemporalProblem, anndata.AnnData]
    """
    cluster = labels_of(adata, "stage_cluster")
    sub = adata.copy()
    sub.obs["time"] = pd.Categorical(
        sub.obs[STAGE_KEY]
        .map({s: float(i) for i, s in enumerate(stages)})
        .astype(float),
    )
    sub.obs["node"] = pd.Categorical(np.where(cluster != "", cluster, "none"))
    tp = (
        TemporalProblem(sub)
        .prepare(time_key="time", joint_attr=LATENT_KEY)
        .solve(epsilon=EPSILON, tau_a=TAU_A, batch_size=OT_BATCH)
    )
    return tp, sub


def clusters_by_stage(
    adata: ad.AnnData,
    stages: list[str],
) -> dict[str, list[str]]:
    """List the stage clusters of each stage.

    :param adata: Cells with stage_cluster.
    :type adata: anndata.AnnData
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :returns: Clusters of each stage in Leiden order.
    :rtype: dict[str, list[str]]
    """
    cluster = labels_of(adata, "stage_cluster")
    obs_stage = labels_of(adata, STAGE_KEY)
    out = {}
    for st in stages:
        out[st] = sorted({c for c in cluster[obs_stage == st] if c}, key=num)
    return out


def transport_edges(
    tp: TemporalProblem,
    stages: list[str],
    clusters_at: dict[str, list[str]],
) -> pd.DataFrame:
    """Tabulate the transported mass between every pair of clusters of
    adjacent stages.

    :param tp: Solved problem from solve_transport.
    :type tp: moscot.problems.time.TemporalProblem
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param clusters_at: Stage clusters of each stage, from clusters_by_stage.
    :type clusters_at: dict[str, list[str]]
    :returns: "from", "to", "mass", "fwd" (share of the source's outgoing
        mass) and "bwd" (share of the target's incoming mass).
    :rtype: pandas.DataFrame
    """
    edges = []
    for i, (a, b) in enumerate(zip(stages[:-1], stages[1:])):
        m = tp.cell_transition(
            source=float(i),
            target=float(i + 1),
            source_groups={"node": clusters_at[a]},
            target_groups={"node": clusters_at[b]},
            forward=True,
            normalize=False,
            batch_size=OT_BATCH,
            key_added=None,
        )
        edges.append(
            pd.DataFrame(
                {
                    "mass": m.stack(),
                    "fwd": m.div(m.sum(axis=1), axis=0).stack(),
                    "bwd": m.div(m.sum(axis=0), axis=1).stack(),
                },
            )
            .rename_axis(["from", "to"])
            .reset_index(),
        )
    return pd.concat(edges, ignore_index=True)


def cluster_table(adata: ad.AnnData) -> pd.DataFrame:
    """Summarise each stage cluster.

    :param adata: Cells with stage_cluster.
    :type adata: anndata.AnnData
    :returns: Indexed by cluster: "stage", cells per source ("n_*"),
        "n_cells", the most common NODE_TISSUE_KEY label ("tissue") and its
        share of the labelled cells ("tissue_frac").
    :rtype: pandas.DataFrame
    """
    cluster = labels_of(adata, "stage_cluster")
    has = cluster != ""
    cells = pd.DataFrame(
        {
            "stage_cluster": cluster[has],
            "stage": labels_of(adata, STAGE_KEY)[has],
            "source": labels_of(adata, BATCH_KEY)[has],
        },
    )
    nodes = (
        pd.crosstab(cells["stage_cluster"], cells["source"])
        .add_prefix("n_")
        .join(cells.groupby("stage_cluster")["stage"].first())
    )
    nodes["n_cells"] = nodes.filter(like="n_").sum(axis=1)
    tissue = pd.Series(
        labels_of(adata, NODE_TISSUE_KEY)[has],
        index=cluster[has],
    )
    tissue = tissue[tissue != ""].groupby(level=0)
    nodes["tissue"] = tissue.agg(lambda t: t.value_counts().index[0])
    nodes["tissue_frac"] = tissue.agg(
        lambda t: t.value_counts(normalize=True).iloc[0],
    )
    nodes["tissue"] = nodes["tissue"].fillna("")
    return nodes


def transport_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build the transport network, keeping edges whose fwd or bwd share is
    MIN_EDGE or more.

    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :returns: Clusters with the nodes columns as attributes and edges with
        mass, fwd and bwd.
    :rtype: networkx.DiGraph
    """
    strong = edges[(edges["fwd"] >= MIN_EDGE) | (edges["bwd"] >= MIN_EDGE)]
    G = nx.DiGraph()
    for n, row in nodes.iterrows():
        G.add_node(n, **{k: plain(v) for k, v in row.items()})
    for rec in strong.to_dict("records"):
        G.add_edge(rec.pop("from"), rec.pop("to"), **rec)
    print(
        f"transport graph: {G.number_of_nodes()} clusters, "
        f"{G.number_of_edges()} edges with a share of {MIN_EDGE}+",
    )
    return G


# %% Anchor mid-gastrula clusters
def anchor_clusters(
    adata: ad.AnnData,
    net: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Score each midG cluster against the ANISEED territories with decoupler
    ULM. Expression of the integration and marker genes is z-scored within
    each source, averaged per cluster and z-scored across clusters. A cluster
    is neural plate when its top territory scores above 0 at padj below
    MAX_PADJ; its blastomeres are every territory within MIN_MARGIN of the top
    score.

    :param adata: Cells with stage_cluster and raw counts.
    :type adata: anndata.AnnData
    :param net: Territory network from load_territory_net.
    :type net: pandas.DataFrame
    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :returns: ULM scores and p-values (clusters x territories), the call table
        (top territory, score, padj, margin, "neural_plate", "blastomeres")
        and the blastomere set of each cluster.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame,
        pandas.Series]
    """
    cluster = labels_of(adata, "stage_cluster")
    obs_stage = labels_of(adata, STAGE_KEY)
    obs_src = labels_of(adata, BATCH_KEY)
    expr = adata.raw.to_adata()
    panel = sorted(set(net["target"]) & set(expr.var_names))
    print(
        f"{len(panel)} of {net['target'].nunique()} marker genes in the data",
    )

    keep = np.flatnonzero((obs_stage == ANCHOR_STAGE) & (cluster != ""))
    genes = sorted(set(adata.var_names) | set(panel))
    x = expr[adata.obs_names[keep], genes].X
    x = pd.DataFrame(
        x.toarray() if hasattr(x, "toarray") else np.asarray(x),
        columns=genes,
    )
    for src in set(obs_src[keep]):
        m = obs_src[keep] == src
        x.loc[m] = (x.loc[m] - x.loc[m].mean()) / x.loc[m].std().replace(
            0,
            np.nan,
        )
    x = x.fillna(0)
    pb = x.groupby(cluster[keep]).mean()
    pb = pb.loc[:, pb.std() > 0]
    es, pv = dc.mt.ulm(data=(pb - pb.mean()) / pb.std(), net=net, tmin=TMIN)
    es.index.name = "stage_cluster"

    top2 = np.argsort(-es.to_numpy(), axis=1)[:, :2]
    rows = np.arange(len(es))
    first = es.to_numpy()[rows, top2[:, 0]]
    second = es.to_numpy()[rows, top2[:, 1]]
    calls = pd.DataFrame(
        {
            "territory": es.columns[top2[:, 0]],
            "score": first,
            "padj": pv.to_numpy()[rows, top2[:, 0]],
            "second": es.columns[top2[:, 1]],
            "margin": first - second,
        },
        index=es.index,
    ).join(nodes)
    calls["neural_plate"] = (calls["padj"] < MAX_PADJ) & (calls["score"] > 0)
    near_top = es.ge(es.max(axis=1) - MIN_MARGIN, axis=0)
    cluster_pick = pd.Series(
        [
            set("/".join(es.columns[m]).split("/")) if ok else set()
            for m, ok in zip(near_top.to_numpy(), calls["neural_plate"])
        ],
        index=es.index,
    )
    calls["blastomeres"] = cluster_pick.map(lambda b: "/".join(sorted(b)))
    print(
        f"{calls['neural_plate'].sum()} of {len(calls)} {ANCHOR_STAGE} "
        "clusters match a neural plate territory",
    )
    return es, pv, calls, cluster_pick


def best_cluster_per_blastomere(
    es: pd.DataFrame,
    pv: pd.DataFrame,
    nodes: pd.DataFrame,
    blastomeres: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the midG clusters for each blastomere by the ULM score of its
    territory.

    :param es: ULM scores from anchor_clusters.
    :type es: pandas.DataFrame
    :param pv: ULM p-values from anchor_clusters.
    :type pv: pandas.DataFrame
    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :param blastomeres: Blastomeres to rank for.
    :type blastomeres: list[str]
    :returns: Every cluster-blastomere score with its rank; and per blastomere
        its grid row and column, territory, best cluster with its cells, score
        and padj, second-best cluster and score, and the margin.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    group_of = {b: t for t in es.columns for b in t.split("/")}
    full = (
        pd.concat(
            {
                "score": es,
                "padj": pv.reindex(index=es.index, columns=es.columns),
            },
            axis=1,
        )
        .stack(level=1, future_stack=True)
        .rename_axis(["stage_cluster", "territory"])
        .reset_index()
    )
    full = pd.DataFrame(
        {"blastomere": list(group_of), "territory": list(group_of.values())},
    ).merge(full, on="territory")
    full["rank"] = (
        full.groupby("blastomere")["score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    full = full.sort_values(["blastomere", "rank"])

    first = full[full["rank"] == 1].set_index("blastomere")
    second = full[full["rank"] == 2].set_index("blastomere")
    best = pd.DataFrame(index=pd.Index(blastomeres, name="blastomere"))
    best["row"] = [NP_GRID.get(b, (None, None))[0] for b in best.index]
    best["column"] = pd.array(
        [NP_GRID.get(b, (None, None))[1] for b in best.index],
        dtype="Int64",
    )
    best["territory"] = first["territory"]
    best["stage_cluster"] = first["stage_cluster"]
    n_cells = best["stage_cluster"].map(nodes["n_cells"])
    best["n_cells"] = n_cells.astype("Int64")
    best["score"] = first["score"]
    best["padj"] = first["padj"]
    best["second_cluster"] = second["stage_cluster"]
    best["second_score"] = second["score"]
    best["margin"] = best["score"] - best["second_score"]
    return full, best


# %% Benchmarks
def winkley_midg_benchmark(
    adata: ad.AnnData,
    cluster_pick: pd.Series,
    best: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the blastomere set of each Winkley midG cell's cluster with the
    cells of its grid label using set_scores.

    :param adata: Cells with stage_cluster and Winkley labels.
    :type adata: anndata.AnnData
    :param cluster_pick: Blastomere set of each midG cluster, from
        anchor_clusters; empty for clusters that are not neural plate.
    :type cluster_pick: pandas.Series
    :param best: Best cluster per blastomere.
    :type best: pandas.DataFrame
    :returns: Per-cell scores, and their means per label, per blastomere and
        overall.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame,
        pandas.DataFrame]
    """
    wl = pd.DataFrame(
        {
            "label": labels_of(adata, WINKLEY_LABEL_KEY),
            "stage_cluster": labels_of(adata, "stage_cluster"),
        },
        index=adata.obs_names,
    )
    wl = wl[
        (labels_of(adata, BATCH_KEY) == WINKLEY)
        & (labels_of(adata, STAGE_KEY) == ANCHOR_STAGE)
    ]
    wl["truth"] = wl["label"].map(winkley_blastomeres)
    wl = wl[wl["truth"].map(len) > 0]
    wl["pick"] = (
        wl["stage_cluster"]
        .map(cluster_pick)
        .map(lambda b: b if isinstance(b, set) else set())
    )
    wl = wl.join(
        pd.DataFrame(
            [set_scores(p, t) for p, t in zip(wl["pick"], wl["truth"])],
            index=wl.index,
            columns=METRICS,
        ),
    )
    by_label = wl.groupby("label").agg(
        n_cells=("score", "size"),
        truth=("truth", lambda t: "/".join(sorted(t.iloc[0]))),
        top_pick=(
            "pick",
            lambda p: p.map(lambda b: "/".join(sorted(b))).mode()[0],
        ),
        **{m: (m, "mean") for m in METRICS},
    )
    by_blastomere = pd.DataFrame(
        [
            {
                "blastomere": b,
                "winkley_labelled": int(
                    wl["truth"].map(lambda t: b in t).sum(),
                ),
                "picked": int(wl["pick"].map(lambda p: b in p).sum()),
                **{
                    m: wl.loc[wl["truth"].map(lambda t: b in t), m].mean()
                    for m in METRICS
                },
                "best_cluster": best.loc[b, "stage_cluster"],
            }
            for b in best.index
        ],
    ).set_index("blastomere")
    summary = wl[METRICS].mean().to_frame().T.assign(n_cells=len(wl))
    print(
        f"Winkley grid-labelled cells: {len(wl)}; mean "
        + ", ".join(f"{m} {wl[m].mean():.3f}" for m in METRICS),
    )
    return wl, by_label, by_blastomere, summary


def predict_identities(
    edges: pd.DataFrame,
    cluster_pick: pd.Series,
    clusters_at: dict[str, list[str]],
    nodes: pd.DataFrame,
    stages: list[str],
    stage_maps: dict[str, gpd.GeoDataFrame],
) -> tuple[pd.DataFrame, pd.Series]:
    """Carry the midG blastomere sets through the transport to the later
    IDENTITY_GEN stages. A cluster's weight on a cell sums bwd x the parent's
    weight over its parent edges, split evenly over the cell's map_cells.

    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :param cluster_pick: Blastomere set of each midG cluster, from
        anchor_clusters; empty for clusters that are not neural plate.
    :type cluster_pick: pandas.Series
    :param clusters_at: Stage clusters of each stage, from clusters_by_stage.
    :type clusters_at: dict[str, list[str]]
    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param stage_maps: Neural plate maps from load_stage_maps.
    :type stage_maps: dict[str, geopandas.GeoDataFrame]
    :returns: One row per later cluster (top midG ancestor, its cells and all
        weights), and the identity text of every cluster.
    :rtype: tuple[pandas.DataFrame, pandas.Series]
    """
    anchor_gen = IDENTITY_GEN[ANCHOR_STAGE]
    identity: dict[str, dict[str, float]] = {}
    for c, bs in cluster_pick.items():
        if bs:
            identity[c] = {b: 1 / len(bs) for b in bs}
    node_identity = {c: "/".join(sorted(w)) for c, w in identity.items()}
    rows = []
    i0 = stages.index(ANCHOR_STAGE)
    later = [s for s in stages[i0 + 1 :] if s in IDENTITY_GEN]
    for st in later:
        nxt: dict[str, dict[str, float]] = {}
        step = edges_between(edges, list(identity), clusters_at[st])
        for rec in step.to_dict("records"):
            weights = nxt.setdefault(rec["to"], {})
            for b, frac in identity[rec["from"]].items():
                cells = map_cells(b, st, stage_maps)
                for d in sorted(cells, key=blastomere_key):
                    add = rec["bwd"] * frac / len(cells)
                    weights[d] = weights.get(d, 0.0) + add
        identity = nxt
        for n in clusters_at[st]:
            w = identity.get(n, {})
            anc: dict[str, float] = {}
            for d, v in w.items():
                (a,) = to_generation(d, anchor_gen)
                anc[a] = anc.get(a, 0.0) + v
            kept = {a for a, v in anc.items() if v >= MIN_SHOW}
            top = max(sorted(anc), key=lambda a: anc[a]) if anc else ""
            kids = map_cells(top, st, stage_maps) if top else set()
            keep = {d for d in w if to_generation(d, anchor_gen) & kept}
            rows.append(
                {
                    "stage_cluster": n,
                    "stage": st,
                    "generation": max(map(generation_of, kids), default=None),
                    "tissue": nodes.loc[n, "tissue"],
                    "assigned_share": sum(w.values()),
                    "top_ancestor": top,
                    "top_ancestor_share": anc.get(top, 0.0),
                    "predicted": join_names(kids),
                    "ancestors": fmt(anc),
                    "identities": fmt(w, keep=keep),
                },
            )
            node_identity[n] = join_names(kids)
    identities = pd.DataFrame(rows).set_index("stage_cluster")
    identities["generation"] = identities["generation"].astype("Int64")
    node_identity = pd.Series(node_identity).reindex(nodes.index).fillna("")
    return identities, node_identity


def adjacency_table(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_identity: pd.Series,
    stages: list[str],
) -> pd.DataFrame:
    """Summarise each cluster with its neighbours in the transport.

    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :param node_identity: Identity text of each cluster.
    :type node_identity: pandas.Series
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :returns: Per cluster: stage, cells, tissue, identity, and children and
        parents with a share of MIN_EDGE or more.
    :rtype: pandas.DataFrame
    """
    strong = edges[(edges["fwd"] >= MIN_EDGE) | (edges["bwd"] >= MIN_EDGE)]
    adjacency = nodes[["stage", "n_cells", "tissue"]].assign(
        identity=node_identity,
    )
    adjacency["children"] = (
        share_list(strong, "from", "fwd").reindex(adjacency.index).fillna("")
    )
    adjacency["parents"] = (
        share_list(strong, "to", "bwd").reindex(adjacency.index).fillna("")
    )
    adjacency["stage"] = pd.Categorical(
        adjacency["stage"],
        categories=stages,
        ordered=True,
    )
    return (
        adjacency.assign(order=adjacency.index.map(num))
        .sort_values(["stage", "order"])
        .drop(columns="order")
    )


def backward_benchmark(
    adata: ad.AnnData,
    tp: TemporalProblem,
    sub: ad.AnnData,
    cluster_pick: pd.Series,
    neural_clusters: list[str],
    stages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transport each midG neural plate cluster back to the Winkley-labelled
    cells of each BENCH_GEN stage and compare each label's share of the
    ancestry with its share of labelled cells. A label is expected when it
    covers an ancestor of the cluster's blastomeres.

    :param adata: Cells with Winkley labels.
    :type adata: anndata.AnnData
    :param tp: Solved problem from solve_transport.
    :type tp: moscot.problems.time.TemporalProblem
    :param sub: The problem's AnnData from solve_transport; gains a label
        column per stage.
    :type sub: anndata.AnnData
    :param cluster_pick: Blastomere set of each midG cluster, from
        anchor_clusters; empty for clusters that are not neural plate.
    :type cluster_pick: pandas.Series
    :param neural_clusters: midG clusters called neural plate.
    :type neural_clusters: list[str]
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :returns: Per-cluster rows (expected and top label with shares and folds),
        per-label shares, and per-stage means.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    lab = labels_of(adata, WINKLEY_LABEL_KEY)
    obs_src = labels_of(adata, BATCH_KEY)
    obs_stage = labels_of(adata, STAGE_KEY)
    backward, shares = [], []
    for st, gen in BENCH_GEN.items():
        at = (obs_src == WINKLEY) & (obs_stage == st) & (lab != "")
        if st not in stages or not at.any():
            print(f"{st}: no labelled Winkley cells, skipped")
            continue
        groups = sorted(set(lab[at]))
        key = f"winkley_{st}"
        sub.obs[key] = pd.Categorical(np.where(at, lab, "none"))
        back = tp.cell_transition(
            source=float(stages.index(st)),
            target=float(stages.index(ANCHOR_STAGE)),
            source_groups={key: groups},
            target_groups={"node": neural_clusters},
            forward=False,
            batch_size=OT_BATCH,
            key_added=None,
        )
        counts = pd.Series(lab[at]).value_counts(normalize=True)
        background = counts.reindex(groups)
        named = {l: early_blastomeres(l, gen) for l in groups}
        for c in neural_clusters:
            share = back[c].reindex(groups).fillna(0)
            fold = share / background
            truth = set().union(
                *(to_generation(b, gen) for b in cluster_pick[c]),
            )
            expected = [l for l in groups if named[l] & truth]
            top = share.idxmax()
            shares += [
                {
                    "stage": st,
                    "stage_cluster": c,
                    "label": l,
                    "share": share[l],
                    "background": background[l],
                    "fold": fold[l],
                    "expected": l in expected,
                }
                for l in groups
            ]
            backward.append(
                {
                    "stage": st,
                    "stage_cluster": c,
                    "blastomeres": "/".join(sorted(cluster_pick[c])),
                    "expected_ancestors": "/".join(sorted(truth)),
                    "expected_labels": "/".join(expected),
                    "expected_share": share[expected].sum(),
                    "expected_background": background[expected].sum(),
                    "expected_fold": (
                        share[expected].sum() / background[expected].sum()
                        if expected
                        else np.nan
                    ),
                    "top_label": top,
                    "top_share": share[top],
                    "top_fold": fold[top],
                    "top_is_expected": top in expected,
                    "labels": ",".join(
                        f"{l}:{share[l]:.2f}({fold[l]:.1f}x)"
                        for l in share.sort_values(ascending=False).index
                        if share[l] >= MIN_SHOW
                    ),
                },
            )
    backward = pd.DataFrame(backward)
    summary = (
        backward.groupby("stage")[
            [
                "expected_share",
                "expected_background",
                "expected_fold",
                "top_is_expected",
            ]
        ].mean()
        if len(backward)
        else pd.DataFrame()
    )
    print(summary.round(3).to_string())
    return backward, pd.DataFrame(shares), summary


def forward_benchmark(
    adata: ad.AnnData,
    tp: TemporalProblem,
    sub: ad.AnnData,
    cluster_pick: pd.Series,
    neural_clusters: list[str],
    clusters_at: dict[str, list[str]],
    stages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transport every midG cluster forward to the Cao tissue-labelled cells
    of each later stage and compare each tissue's share of the descendants
    with its share of labelled cells.

    :param adata: Cells with Cao tissue labels.
    :type adata: anndata.AnnData
    :param tp: Solved problem from solve_transport.
    :type tp: moscot.problems.time.TemporalProblem
    :param sub: The problem's AnnData from solve_transport; gains a label
        column per stage.
    :type sub: anndata.AnnData
    :param cluster_pick: Blastomere set of each midG cluster, from
        anchor_clusters; empty for clusters that are not neural plate.
    :type cluster_pick: pandas.Series
    :param neural_clusters: midG clusters called neural plate.
    :type neural_clusters: list[str]
    :param clusters_at: Stage clusters of each stage, from clusters_by_stage.
    :type clusters_at: dict[str, list[str]]
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :returns: Per-cluster rows (nervous system share and fold, top tissue),
        per-tissue shares, and means by stage and neural plate status.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    tissue = labels_of(adata, CAO_TISSUE_KEY)
    obs_src = labels_of(adata, BATCH_KEY)
    obs_stage = labels_of(adata, STAGE_KEY)
    midg_clusters = clusters_at[ANCHOR_STAGE]
    forward, shares = [], []
    for st in stages[stages.index(ANCHOR_STAGE) + 1 :]:
        at = (obs_src == CAO) & (obs_stage == st) & (tissue != "")
        if not at.any():
            continue
        groups = sorted(set(tissue[at]))
        key = f"cao_tissue_{st}"
        sub.obs[key] = pd.Categorical(np.where(at, tissue, "none"))
        fwd = tp.cell_transition(
            source=float(stages.index(ANCHOR_STAGE)),
            target=float(stages.index(st)),
            source_groups={"node": midg_clusters},
            target_groups={key: groups},
            forward=True,
            batch_size=OT_BATCH,
            key_added=None,
        )
        counts = pd.Series(tissue[at]).value_counts(normalize=True)
        background = counts.reindex(groups)
        for c in midg_clusters:
            share = fwd.loc[c].reindex(groups).fillna(0)
            top = share.idxmax()
            shares += [
                {
                    "stage": st,
                    "stage_cluster": c,
                    "neural_plate": c in neural_clusters,
                    "tissue": t,
                    "share": share[t],
                    "background": background[t],
                }
                for t in groups
            ]
            forward.append(
                {
                    "stage": st,
                    "stage_cluster": c,
                    "neural_plate": c in neural_clusters,
                    "blastomeres": "/".join(
                        sorted(cluster_pick.get(c, set())),
                    ),
                    "nervous_share": share.get(NEURAL_LABEL, 0.0),
                    "nervous_background": background.get(NEURAL_LABEL, 0.0),
                    "nervous_fold": share.get(NEURAL_LABEL, 0.0)
                    / background.get(NEURAL_LABEL, np.nan),
                    "top_tissue": top,
                    "top_share": share[top],
                    "tissues": ",".join(
                        f"{t}:{share[t]:.2f}"
                        for t in share.sort_values(ascending=False).index
                        if share[t] >= MIN_SHOW
                    ),
                },
            )
    forward = pd.DataFrame(forward)
    summary = pd.DataFrame()
    if len(forward):
        forward["stage"] = pd.Categorical(
            forward["stage"],
            categories=stages,
            ordered=True,
        )
        summary = forward.groupby(["stage", "neural_plate"], observed=True)[
            ["nervous_share", "nervous_background", "nervous_fold"]
        ].mean()
        print(summary.round(3).to_string())
    return forward, pd.DataFrame(shares), summary


def neural_plate_lineage(
    edges: pd.DataFrame,
    stages: list[str],
    clusters_at: dict[str, list[str]],
    neural_clusters: list[str],
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Score each cluster's share of lineage through the midG neural plate
    clusters: 1 or 0 at midG, the bwd-weighted score of the parents later on,
    the fwd-weighted score of the children earlier on.

    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param clusters_at: Stage clusters of each stage, from clusters_by_stage.
    :type clusters_at: dict[str, list[str]]
    :param neural_clusters: midG clusters called neural plate.
    :type neural_clusters: list[str]
    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :returns: Per cluster: "np_share", stage, cells, tissue, "role" (ancestor,
        anchor or descendant) and "selected" (np_share of LINEAGE_MIN or
        more).
    :rtype: pandas.DataFrame
    """
    anchors = clusters_at[ANCHOR_STAGE]
    share = {c: float(c in neural_clusters) for c in anchors}
    i0 = stages.index(ANCHOR_STAGE)
    for a, b in zip(stages[i0:-1], stages[i0 + 1 :]):
        e = edges_between(edges, clusters_at[a], clusters_at[b])
        s = (e["bwd"] * e["from"].map(share)).groupby(e["to"]).sum()
        share.update(s.reindex(clusters_at[b]).fillna(0).to_dict())
    for a, b in zip(stages[:i0][::-1], stages[1 : i0 + 1][::-1]):
        e = edges_between(edges, clusters_at[a], clusters_at[b])
        s = (e["fwd"] * e["to"].map(share)).groupby(e["from"]).sum()
        share.update(s.reindex(clusters_at[a]).fillna(0).to_dict())
    table = pd.DataFrame({"np_share": pd.Series(share)}).join(
        nodes[["stage", "n_cells", "tissue"]],
    )
    table["role"] = np.select(
        [
            table["stage"].map(stages.index) < i0,
            table["stage"] == ANCHOR_STAGE,
        ],
        ["ancestor", "anchor"],
        default="descendant",
    )
    table["selected"] = table["np_share"] >= LINEAGE_MIN
    table["stage"] = pd.Categorical(
        table["stage"],
        categories=stages,
        ordered=True,
    )
    table = (
        table.assign(order=table.index.map(num))
        .sort_values(["stage", "order"])
        .drop(columns="order")
        .rename_axis("stage_cluster")
    )
    print(
        table[table["selected"]]
        .groupby("stage", observed=True)
        .agg(clusters=("np_share", "size"), cells=("n_cells", "sum"))
        .to_string(),
    )
    return table


# %% Expression by blastomere
def clone_shares(
    edges: pd.DataFrame,
    cluster_pick: pd.Series,
    clusters_at: dict[str, list[str]],
    stages: list[str],
) -> dict[str, pd.DataFrame]:
    """Trace each midG neural plate cell (clone) forward through the
    transport. midG clusters split evenly over their blastomere set; later
    clusters sum bwd x their parents' shares, so sisters always get equal
    shares.

    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :param cluster_pick: Blastomere set of each midG cluster, from
        anchor_clusters; empty for clusters that are not neural plate.
    :type cluster_pick: pandas.Series
    :param clusters_at: Stage clusters of each stage, from clusters_by_stage.
    :type clusters_at: dict[str, list[str]]
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :returns: For midG and the later IDENTITY_GEN stages, the share of each
        cluster's cells descending from each clone (clusters x clones).
    :rtype: dict[str, pandas.DataFrame]
    """
    share: dict[str, dict[str, float]] = {}
    for c, bs in cluster_pick.items():
        if bs:
            share[c] = {b: 1 / len(bs) for b in bs}
    out = {ANCHOR_STAGE: share}
    i0 = stages.index(ANCHOR_STAGE)
    later = [s for s in stages[i0 + 1 :] if s in IDENTITY_GEN]
    for st in later:
        nxt: dict[str, dict[str, float]] = {}
        step = edges_between(edges, list(share), clusters_at[st])
        for rec in step.to_dict("records"):
            weights = nxt.setdefault(rec["to"], {})
            for b, w in share[rec["from"]].items():
                weights[b] = weights.get(b, 0.0) + rec["bwd"] * w
        share = out[st] = nxt
    tables = {}
    for st, s in out.items():
        t = pd.DataFrame.from_dict(s, orient="index").fillna(0.0)
        rows = sorted(t.index, key=num)
        extra = sorted(set(t) - set(NP_ORDER))
        cols = [b for b in NP_ORDER if b in t] + extra
        tables[st] = t.loc[rows, cols].rename_axis("stage_cluster")
    return tables


def clone_expression(
    de: pd.DataFrame,
    share: pd.DataFrame,
    nodes: pd.DataFrame,
    stage: str,
    cells_of: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Average each gene's per-cluster DE statistics over each clone's cells
    at one stage, spreading a clone over clusters by share x cluster size. A
    gene is expressed in a clone at support EXPRESSED_MIN and specific when
    also SPECIFIC_MIN log2 above its mean over the clones.

    :param de: DE table of the stage.
    :type de: pandas.DataFrame
    :param share: Clone shares of the stage's clusters.
    :type share: pandas.DataFrame
    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :param stage: Stage.
    :type stage: str
    :param cells_of: Cells of each clone at this stage; None uses
        IDENTITY_GEN.
    :type cells_of: dict[str, set[str]] or None
    :returns: One row per gene and clone: descendants, clone_cells, support
        (share in clusters where scVI calls the gene up), replicated_support,
        expression (scVI scale1), log_cp10k, detection (share of cells with a
        count), lfc, specificity (log2 over the clone mean), expressed,
        specific, n_clones and n_specific.
    :rtype: pandas.DataFrame
    """
    tested = sorted(set(de["stage_cluster"]) & set(share.index), key=num)
    size = nodes.loc[tested, "n_cells"].astype(float)
    cells = share.loc[tested].mul(size, axis=0)
    cells = cells.loc[:, cells.sum() > 0]
    comp = cells / cells.sum()
    d = de[de["stage_cluster"].isin(tested)]
    up = d[IS_DE].astype(bool) & (d["lfc_mean"] > 0)
    d = d.assign(
        up=up.astype(float),
        up_replicated=(up & d["replicated"].astype(bool)).astype(float),
    )

    def per_clone(col: str) -> pd.DataFrame:
        m = d.pivot(index="gene", columns="stage_cluster", values=col)
        return m.reindex(columns=comp.index).fillna(0.0) @ comp

    expression = per_clone("scale1")
    mean = expression.mean(axis=1)
    ratio = expression.div(mean.where(mean > 0), axis=0)
    matrices = {
        "support": per_clone("up"),
        "replicated_support": per_clone("up_replicated"),
        "expression": expression,
        "log_cp10k": np.log1p(expression * 1e4),
        "detection": per_clone("non_zeros_proportion1"),
        "lfc": per_clone("lfc_mean"),
        "specificity": np.log2(ratio),
    }
    table = pd.DataFrame(
        {k: m.stack(future_stack=True) for k, m in matrices.items()},
    )
    table = table.rename_axis(["gene", "clone"]).reset_index()
    table["expressed"] = table["support"] >= EXPRESSED_MIN
    specific = table["specificity"] >= SPECIFIC_MIN
    table["specific"] = table["expressed"] & specific
    by_gene = table.groupby("gene")
    table["n_clones"] = by_gene["expressed"].transform("sum")
    table["n_specific"] = by_gene["specific"].transform("sum")

    if cells_of is None:
        gen = IDENTITY_GEN[stage]
        cells_of = {b: to_generation(b, gen) for b in comp}
    descendants = {b: join_names(cells_of.get(b, ())) for b in comp}
    table["stage"] = stage
    table["descendants"] = table["clone"].map(descendants)
    table["clone_cells"] = table["clone"].map(cells.sum())
    columns = ["gene_name", "KH2012", "uniprot", "homolog"]
    info = [c for c in columns if c in de]
    ann = de.drop_duplicates("gene").set_index("gene")[info]
    table = table.join(ann, on="gene")
    front = ["stage", "clone", "descendants", "clone_cells"]
    front += ["gene"] + info
    table = table[front + [c for c in table if c not in front]]
    order = {b: i for i, b in enumerate(comp.columns)}
    table = (
        table.assign(order=table["clone"].map(order))
        .sort_values(
            ["order", "specific", "specificity", "gene"],
            ascending=[True, False, False, True],
        )
        .drop(columns="order")
        .reset_index(drop=True)
    )
    return table


def predict_expression(
    de: dict[str, pd.DataFrame],
    shares: dict[str, pd.DataFrame],
    nodes: pd.DataFrame,
    stage_maps: dict[str, gpd.GeoDataFrame],
    known_genes: set[str],
) -> dict[str, pd.DataFrame]:
    """Run clone_expression at every stage that has DE results and clone
    shares.

    :param de: DE tables per stage.
    :type de: dict[str, pandas.DataFrame]
    :param shares: Clone shares per stage.
    :type shares: dict[str, pandas.DataFrame]
    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :param stage_maps: Neural plate maps from load_stage_maps.
    :type stage_maps: dict[str, geopandas.GeoDataFrame]
    :param known_genes: Genes with ANISEED in situ data.
    :type known_genes: set[str]
    :returns: Clone expression per stage, with descendants named from the
        stage maps and "known" flagging known_genes.
    :rtype: dict[str, pandas.DataFrame]
    """
    results = {}
    for st, share in shares.items():
        if st not in de or not share.shape[1]:
            continue
        cells_of = {b: map_cells(b, st, stage_maps) for b in share}
        t = results[st] = clone_expression(
            de[st],
            share,
            nodes,
            st,
            cells_of,
        )
        known = t["gene"].isin(known_genes)
        t.insert(t.columns.get_loc("support"), "known", known)
        per_gene = t.drop_duplicates("gene")
        print(
            f"{st}: {t['clone'].nunique()} clones, "
            f"{(per_gene['n_clones'] > 0).sum()} genes expressed in at "
            f"least one, {(per_gene['n_specific'] > 0).sum()} specific",
        )
    return results


def top_genes(expression: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pick the TOP_GENES most specific genes of each clone at each stage
    among the specific genes without ANISEED data.

    :param expression: Clone expression per stage, from predict_expression.
    :type expression: dict[str, pandas.DataFrame]
    :returns: Rows of the clone expression tables with a "rank" column,
        ordered by stage, clone and rank.
    :rtype: pandas.DataFrame
    """
    tops = []
    for t in expression.values():
        t = t[t["specific"] & ~t["known"]].sort_values(
            ["specificity", "gene"],
            ascending=[False, True],
        )
        top = t.groupby("clone", sort=False).head(TOP_GENES).copy()
        rank = top.groupby("clone", sort=False).cumcount() + 1
        top.insert(4, "rank", rank)
        tops.append(top)
    if not tops:
        return pd.DataFrame()
    table = pd.concat(tops, ignore_index=True)
    order = {b: i for i, b in enumerate(NP_ORDER)}
    stage_order = {s: i for i, s in enumerate(CHAIN)}
    table = table.assign(
        s=table["stage"].map(stage_order),
        c=table["clone"].map(order),
    )
    table = table.sort_values(["s", "c", "rank"]).drop(columns=["s", "c"])
    return table.reset_index(drop=True)


def expression_benchmark(
    expression: dict[str, pd.DataFrame],
    stage_markers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score, for each ANISEED gene and stage, the clones where the gene is
    expressed against its in situ clones (set_scores), and rank all clones by
    expression (auroc).

    :param expression: Clone expression per stage, from predict_expression.
    :type expression: dict[str, pandas.DataFrame]
    :param stage_markers: ANISEED genes from load_stage_markers.
    :type stage_markers: pandas.DataFrame
    :returns: Per-gene scores, and per-stage means with midG flagged circular
        because these genes anchored the clusters.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    rows = []
    for rec in stage_markers.to_dict("records"):
        st, gene, truth = rec["stage"], rec["gene"], rec["truth"]
        row = {
            "stage": st,
            "gene": gene,
            "aniseed": rec["aniseed"],
            "homolog": None,
            "cells": join_names(rec["cells"]),
            "truth": join_names(truth),
            "n_truth": len(truth),
        }
        t = expression.get(st)
        g = t[t["gene"] == gene] if t is not None else pd.DataFrame()
        row["tested"] = len(g) > 0
        if row["tested"]:
            pick = set(g.loc[g["expressed"], "clone"])
            row["predicted"] = join_names(pick)
            row["n_predicted"] = len(pick)
            row.update(set_scores(pick, truth))
            row["auroc"] = auroc(g["expression"], g["clone"].isin(truth))
            row["homolog"] = g["homolog"].iloc[0] if "homolog" in g else None
        rows.append(row)
    bench = pd.DataFrame(rows)
    if not len(bench):
        return bench, pd.DataFrame()
    if "n_predicted" in bench:
        bench["n_predicted"] = bench["n_predicted"].astype("Int64")
    stage_order = {s: i for i, s in enumerate(CHAIN)}
    bench = (
        bench.assign(s=bench["stage"].map(stage_order))
        .sort_values(["s", "gene"])
        .drop(columns="s")
        .reset_index(drop=True)
    )
    tested = bench[bench["tested"]]
    means = tested.groupby("stage")[METRICS + ["auroc"]].mean()
    summary = pd.DataFrame(
        {
            "n_genes": bench.groupby("stage").size(),
            "n_tested": tested.groupby("stage").size(),
            "auroc_above_half": tested.groupby("stage")["auroc"].apply(
                lambda a: (a.dropna() > 0.5).mean(),
            ),
        },
    ).join(means)
    summary = summary.reindex(
        [s for s in CHAIN if s in summary.index],
    ).fillna({"n_tested": 0})
    summary["n_tested"] = summary["n_tested"].astype(int)
    summary["circular"] = summary.index == ANCHOR_STAGE
    print(summary.to_string())
    return bench, summary


# %% Plots
def plot_stage_umaps(adata: ad.AnnData, stages: list[str], out: str) -> None:
    """Draw each stage's UMAP coloured by its Leiden clusters
    ({out}_{stage}_umap).

    :param adata: Cells from cluster_stages.
    :type adata: anndata.AnnData
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param out: Output file prefix.
    :type out: str
    """
    obs_stage = labels_of(adata, STAGE_KEY)
    for st in stages:
        m = obs_stage == st
        leiden = adata.obs.loc[m, "stage_leiden"].astype(str)
        view = ad.AnnData(
            obs=pd.DataFrame(
                {
                    "leiden": pd.Categorical(
                        leiden,
                        categories=sorted(set(leiden), key=int),
                    ),
                },
                index=adata.obs_names[m],
            ),
            obsm={"X_umap": adata.obsm["X_umap_stage"][m]},
        )
        fig = sc.pl.umap(
            view,
            color="leiden",
            legend_loc="on data",
            legend_fontsize=6,
            legend_fontoutline=2,
            frameon=False,
            title=f"{st} Leiden clusters",
            return_fig=True,
        )
        save(fig, out, f"{st}_umap")


def plot_transport_heatmaps(
    edges: pd.DataFrame,
    stages: list[str],
    clusters_at: dict[str, list[str]],
    out: str,
) -> None:
    """Draw the fwd share between each pair of adjacent stages as a heatmap,
    target clusters ordered by their main source ({out}_transport_{a}_{b}).

    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param clusters_at: Stage clusters of each stage, from clusters_by_stage.
    :type clusters_at: dict[str, list[str]]
    :param out: Output file prefix.
    :type out: str
    """
    for a, b in zip(stages[:-1], stages[1:]):
        mat = (
            edges_between(edges, clusters_at[a], clusters_at[b])
            .pivot(index="from", columns="to", values="fwd")
            .reindex(index=clusters_at[a], columns=clusters_at[b])
            .fillna(0)
        )
        mat = mat[
            sorted(
                mat.columns,
                key=lambda t: (
                    mat.index.get_loc(mat[t].idxmax()),
                    -mat[t].max(),
                ),
            )
        ]
        fig, ax = plt.subplots(
            figsize=(
                0.18 * len(mat.columns) + 2.5,
                0.18 * len(mat.index) + 1.5,
            ),
        )
        im = ax.imshow(
            mat.to_numpy(),
            cmap="Blues",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_yticks(
            range(len(mat.index)),
            [num(n) for n in mat.index],
            fontsize=6,
        )
        ax.set_xticks(
            range(len(mat.columns)),
            [num(n) for n in mat.columns],
            fontsize=6,
            rotation=90,
        )
        ax.set_ylabel(f"{a} cluster")
        ax.set_xlabel(f"{b} cluster")
        ax.set_title(f"Transport {a} to {b}", loc="left", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.6, label="share of the row cluster")
        save(fig, out, f"transport_{a}_{b}")


def plot_transport_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    stages: list[str],
    clusters_at: dict[str, list[str]],
    out: str,
) -> None:
    """Draw the transport network: clusters in stage columns, grouped and
    coloured by tissue and ordered to reduce crossings, with edges of fwd
    PLOT_MIN_EDGE or more ({out}_transport_network).

    :param nodes: Per-cluster table from cluster_table.
    :type nodes: pandas.DataFrame
    :param edges: Cluster transport edges from transport_edges.
    :type edges: pandas.DataFrame
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param clusters_at: Stage clusters of each stage, from clusters_by_stage.
    :type clusters_at: dict[str, list[str]]
    :param out: Output file prefix.
    :type out: str
    """
    shown = edges[edges["fwd"] >= PLOT_MIN_EDGE]
    tissue_rank = {t: i for i, t in enumerate(TISSUE_COLORS)}

    def rank(n: str) -> int:
        return tissue_rank.get(nodes.loc[n, "tissue"], len(tissue_rank))

    order = {}
    for st in stages:
        order[st] = sorted(clusters_at[st], key=lambda n: (rank(n), num(n)))
    y = {}
    for ns in order.values():
        for i, n in enumerate(ns):
            y[n] = (i + 0.5) / len(ns)
    for sweep in range(8):
        side, other = ("to", "from") if sweep % 2 == 0 else ("from", "to")
        for st in stages if sweep % 2 == 0 else stages[::-1]:
            bary = {}
            for n in order[st]:
                e = shown[shown[side] == n]
                bary[n] = (
                    np.average(
                        e[other].map(y).to_numpy(float),
                        weights=e["mass"].to_numpy(float),
                    )
                    if len(e)
                    else y[n]
                )
            order[st] = sorted(order[st], key=lambda n: (rank(n), bary[n]))
            for i, n in enumerate(order[st]):
                y[n] = (i + 0.5) / len(order[st])
    height = max(len(v) for v in order.values())
    pos = {}
    for n in nodes.index:
        pos[n] = (stages.index(nodes.loc[n, "stage"]), y[n] * height)

    fig, ax = plt.subplots(
        figsize=(1.5 * len(stages) + 2.5, 0.32 * height + 1.2),
    )
    for r in shown.to_dict("records"):
        (x0, y0), (x1, y1) = pos[r["from"]], pos[r["to"]]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=MUTED,
            lw=0.3 + 2.5 * r["fwd"],
            alpha=0.15 + 0.5 * r["fwd"],
            solid_capstyle="round",
            zorder=1,
        )
    fill = nodes["tissue"].map(TISSUE_COLORS).fillna(OTHER_COLOR)
    xs, ys = zip(*(pos[n] for n in nodes.index))
    ax.scatter(
        xs,
        ys,
        s=160,
        c=fill.to_list(),
        edgecolors="white",
        linewidths=1.0,
        zorder=2,
    )
    for n in nodes.index:
        ax.text(
            *pos[n],
            str(num(n)),
            ha="center",
            va="center",
            fontsize=5.5,
            zorder=3,
            color=(
                "white"
                if fill[n] in ("#2a78d6", "#eb6834", "#008300", "#4a3aa7")
                else "black"
            ),
        )
    ax.set_xticks(range(len(stages)), stages)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_yticks([])
    ax.set_ylim(height + 0.5, -0.5)
    ax.set_xlim(-0.5, len(stages) - 0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=TISSUE_COLORS[t],
            markersize=8,
            label=t,
        )
        for t in TISSUE_COLORS
        if t in set(nodes["tissue"])
    ]
    if (~nodes["tissue"].isin(TISSUE_COLORS)).any():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=OTHER_COLOR,
                markersize=8,
                label="other or unlabelled",
            ),
        )
    handles += [
        plt.Line2D(
            [],
            [],
            color=MUTED,
            lw=0.3 + 2.5 * v,
            alpha=0.15 + 0.5 * v,
            label=f"share {v:g}",
        )
        for v in (PLOT_MIN_EDGE, 0.5, 1.0)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=8,
    )
    ax.set_title(
        "Optimal transport between clusters",
        loc="left",
        fontsize=10,
        pad=20,
    )
    save(fig, out, "transport_network")


def plot_neural_plate(
    neural_plate: gpd.GeoDataFrame,
    values: pd.Series,
    label: str,
    title: str,
    out: str,
    name: str,
    clusters: pd.Series | None = None,
    vmax: float | None = None,
    notes: pd.Series | None = None,
    outline: set[str] | None = None,
    outline_label: str | None = None,
    key: str = "name",
) -> None:
    """Draw a neural plate map shaded by one value per cell, with every cell
    labelled ({out}_{name}).

    :param neural_plate: Map polygons with name and clone.
    :type neural_plate: geopandas.GeoDataFrame
    :param values: Shading, indexed by the map column key.
    :type values: pandas.Series
    :param label: Colour bar label.
    :type label: str
    :param title: Title.
    :type title: str
    :param out: Output file prefix.
    :type out: str
    :param name: File name suffix.
    :type name: str
    :param clusters: Cluster to number under each cell name, indexed by name.
    :type clusters: pandas.Series or None
    :param vmax: Top of the colour scale.
    :type vmax: float or None
    :param notes: Text under each cell name, indexed by key; ignored with
        clusters.
    :type notes: pandas.Series or None
    :param outline: Cells to outline, by name.
    :type outline: set[str] or None
    :param outline_label: Legend text for the outline.
    :type outline_label: str or None
    :param key: Map column that values and notes are indexed by, "name" or
        "clone".
    :type key: str
    """
    gdf = neural_plate.join(values.rename("value"), on=key)
    second = pd.Series(None, index=gdf.index, dtype=object)
    if clusters is not None:
        cl = gdf["name"].map(clusters)
        second = cl.map(lambda c: str(num(c)) if isinstance(c, str) else None)
    elif notes is not None:
        second = gdf[key].map(notes)
    x0, y0, x1, y1 = gdf.total_bounds
    height = min(max(5.6 * (y1 - y0) / (x1 - x0) + 0.8, 3.0), 10.0)
    fig, ax = plt.subplots(figsize=(7, height))
    gdf.plot(
        column="value",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        ax=ax,
        edgecolor="white",
        linewidth=1.0,
        legend=True,
        legend_kwds={"label": label, "shrink": 0.6},
        missing_kwds={"color": "#f0efec", "edgecolor": "white"},
    )
    if outline:
        gdf[gdf["name"].isin(outline)].boundary.plot(
            ax=ax,
            color=INK,
            linewidth=1.6,
        )
        ax.legend(
            handles=[
                Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=1.6),
            ],
            labels=[outline_label or "outlined cells"],
            loc="upper left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=False,
            fontsize=7,
        )
    top = vmax or gdf["value"].max()
    points = gdf.representative_point()
    for pt, cell_name, v, text in zip(
        points,
        gdf["name"],
        gdf["value"],
        second,
    ):
        ink = "white" if v > 0.55 * top else INK
        has_text = isinstance(text, str) and text != ""
        ax.annotate(
            cell_name,
            (pt.x, pt.y),
            xytext=(0, 7 if has_text else 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=5.5,
            color=ink,
        )
        if not has_text:
            continue
        big = clusters is not None
        ax.annotate(
            text,
            (pt.x, pt.y),
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8 if big else 4.5,
            fontweight="bold" if big else "normal",
            color=ink,
        )
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    save(fig, out, name)


def plot_winkley_labels(by_label: pd.DataFrame, out: str) -> None:
    """Draw the mean precision, recall and score of the midG picks per Winkley
    grid label ({out}_midG_winkley_labels).

    :param by_label: Means per label from winkley_midg_benchmark.
    :type by_label: pandas.DataFrame
    :param out: Output file prefix.
    :type out: str
    """
    if not len(by_label):
        return
    shown_metrics = ["precision", "recall", "score"]
    bl = by_label.sort_values("score")
    ypos = np.arange(len(bl))
    height = 0.8 / len(shown_metrics)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(bl) + 1.2))
    for i, (m, color) in enumerate(zip(shown_metrics, SERIES)):
        ax.barh(
            ypos + (i - 1) * height,
            bl[m],
            height=height,
            color=color,
            edgecolor=SURFACE,
            linewidth=1,
            label=m,
        )
    ax.set_yticks(
        ypos,
        [f"{l}  (n={n})" for l, n in zip(bl.index, bl["n_cells"])],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean over Winkley cells with that label")
    style(ax, grid_axis="x")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.set_title(
        f"{ANCHOR_STAGE} picks against Winkley's neural plate labels",
        loc="left",
        fontsize=9,
        color=INK,
    )
    save(fig, out, f"{ANCHOR_STAGE}_winkley_labels")


def plot_backward(
    backward: pd.DataFrame,
    shares: pd.DataFrame,
    calls: pd.DataFrame,
    out: str,
) -> None:
    """Draw, per BENCH_GEN stage, each neural plate cluster's log2 label
    enrichment with expected labels boxed ({out}_backward_{st}_enrichment),
    and its expected share against the background
    ({out}_backward_{st}_expected).

    :param backward: Per-cluster rows from backward_benchmark.
    :type backward: pandas.DataFrame
    :param shares: Per-label shares from backward_benchmark.
    :type shares: pandas.DataFrame
    :param calls: Call table from anchor_clusters.
    :type calls: pandas.DataFrame
    :param out: Output file prefix.
    :type out: str
    """
    for st, g in shares.groupby("stage") if len(shares) else []:
        fold = g.pivot(index="stage_cluster", columns="label", values="fold")
        expected = g.pivot(
            index="stage_cluster",
            columns="label",
            values="expected",
        )
        rows = sorted(fold.index, key=num)
        fold, expected = fold.loc[rows], expected.loc[rows].astype(bool)
        value = np.log2(fold.clip(lower=1 / 8, upper=8))
        fig, ax = plt.subplots(
            figsize=(0.45 * fold.shape[1] + 3, 0.28 * fold.shape[0] + 1.5),
        )
        im = ax.imshow(
            value.to_numpy(),
            cmap=DIVERGING,
            vmin=-3,
            vmax=3,
            aspect="auto",
            interpolation="nearest",
        )
        for i, j in zip(*np.nonzero(expected.to_numpy())):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=INK,
                    lw=1.2,
                ),
            )
        ax.set_yticks(
            range(len(rows)),
            [f"{num(c)}  {calls.loc[c, 'blastomeres']}" for c in rows],
        )
        ax.set_xticks(
            range(fold.shape[1]),
            fold.columns,
            rotation=45,
            ha="right",
        )
        ax.set_ylabel(f"{ANCHOR_STAGE} neural plate cluster")
        ax.set_xlabel(f"Winkley label at {st}")
        style(ax, grid_axis=None)
        cb = fig.colorbar(im, ax=ax, shrink=0.6)
        cb.set_label("log2 enrichment over background", color=MUTED)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_visible(False)
        ax.legend(
            handles=[
                Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=1.2),
            ],
            labels=["expected from the blastomere lineage"],
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            frameon=False,
            fontsize=7,
        )
        ax.set_title(
            f"Ancestry at {st} of {ANCHOR_STAGE} neural plate clusters",
            loc="left",
            fontsize=9,
            color=INK,
        )
        save(fig, out, f"backward_{st}_enrichment")

    for st, g in backward.groupby("stage") if len(backward) else []:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        for flag, color, text in (
            (True, SERIES[0], "top label is expected"),
            (False, SERIES[1], "top label is not expected"),
        ):
            d = g[g["top_is_expected"] == flag]
            ax.scatter(
                d["expected_background"],
                d["expected_share"],
                s=36,
                color=color,
                edgecolor=SURFACE,
                linewidth=1,
                label=f"{text} ({len(d)})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(
            f"share of Winkley cells at {st} with the expected labels",
        )
        ax.set_ylabel("share of the cluster's ancestry on those labels")
        style(ax, grid_axis="both")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        title = f"Expected ancestry at {st}, per {ANCHOR_STAGE} "
        title += "neural plate cluster"
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        save(fig, out, f"backward_{st}_expected")


def plot_forward(
    forward: pd.DataFrame,
    shares: pd.DataFrame,
    stages: list[str],
    out: str,
) -> None:
    """Draw the nervous system share of each midG cluster's descendants per
    stage ({out}_forward_nervous_share) and the mean tissue mix of the neural
    plate clusters' descendants ({out}_forward_tissue_composition).

    :param forward: Per-cluster rows from forward_benchmark.
    :type forward: pandas.DataFrame
    :param shares: Per-tissue shares from forward_benchmark.
    :type shares: pandas.DataFrame
    :param stages: Stages present, in CHAIN order.
    :type stages: list[str]
    :param out: Output file prefix.
    :type out: str
    """
    if not len(forward):
        return
    fw_stages = [s for s in stages if s in set(forward["stage"].astype(str))]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(0.9 * len(fw_stages) + 2, 3.8))
    for k, (flag, color, text) in enumerate(
        (
            (True, SERIES[0], f"{ANCHOR_STAGE} neural plate clusters"),
            (False, SERIES[1], f"other {ANCHOR_STAGE} clusters"),
        ),
    ):
        for i, st in enumerate(fw_stages):
            at_stage = forward["stage"].astype(str) == st
            d = forward[at_stage & (forward["neural_plate"] == flag)]
            center = i + (k - 0.5) * 0.36
            ax.scatter(
                center + rng.uniform(-0.08, 0.08, len(d)),
                d["nervous_share"],
                s=12,
                color=color,
                alpha=0.8,
                edgecolor="none",
                label=text if i == 0 else None,
            )
            if len(d):
                ax.hlines(
                    d["nervous_share"].median(),
                    center - 0.14,
                    center + 0.14,
                    color=INK,
                    linewidth=1.5,
                )
    background = (
        forward.assign(stage=forward["stage"].astype(str))
        .groupby("stage")["nervous_background"]
        .first()
        .reindex(fw_stages)
    )
    ax.plot(
        range(len(fw_stages)),
        background,
        color=MUTED,
        linewidth=1,
        linestyle="--",
        label="all Cao cells at the stage",
    )
    ax.set_xticks(range(len(fw_stages)), fw_stages)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of descendants in Cao nervous system")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.5))
    labels.append("median")
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.set_title(
        f"Nervous system fate of {ANCHOR_STAGE} clusters",
        loc="left",
        fontsize=9,
        color=INK,
    )
    save(fig, out, "forward_nervous_share")

    comp = (
        shares[shares["neural_plate"]]
        .groupby(["stage", "tissue"])["share"]
        .mean()
        .unstack(fill_value=0)
        .reindex(fw_stages)
        .fillna(0)
    )
    named_tissues = [t for t in TISSUE_COLORS if t in comp]
    other = comp.drop(columns=named_tissues).sum(axis=1)
    fig, ax = plt.subplots(figsize=(0.7 * len(fw_stages) + 3, 3.8))
    bottom = np.zeros(len(comp))
    for t in named_tissues:
        ax.bar(
            range(len(comp)),
            comp[t],
            bottom=bottom,
            width=0.7,
            color=TISSUE_COLORS[t],
            edgecolor=SURFACE,
            linewidth=1,
            label=t,
        )
        bottom += comp[t].to_numpy()
    if other.any():
        ax.bar(
            range(len(comp)),
            other,
            bottom=bottom,
            width=0.7,
            color=OTHER_COLOR,
            edgecolor=SURFACE,
            linewidth=1,
            label="other",
        )
    ax.set_xticks(range(len(comp)), comp.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean share of descendants")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    title = "Cao tissue of the descendants of "
    title += f"{ANCHOR_STAGE} neural plate clusters"
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    save(fig, out, "forward_tissue_composition")


def plot_expression_heatmaps(
    expression: dict[str, pd.DataFrame],
    out: str,
) -> None:
    """Draw, per stage, the specificity of each clone's TOP_GENES predicted
    genes across all clones, with a dot where the gene is expressed
    ({out}_{stage}_clone_expression).

    :param expression: Clone expression per stage, from predict_expression.
    :type expression: dict[str, pandas.DataFrame]
    :param out: Output file prefix.
    :type out: str
    """
    cmap = DIVERGING.with_extremes(bad="#f0efec")
    for st, t in expression.items():
        spec = t[t["specific"] & ~t["known"]].sort_values(
            ["specificity", "gene"],
            ascending=[False, True],
        )
        chosen = spec.groupby("clone", sort=False).head(TOP_GENES)
        if not len(chosen):
            continue
        peak = chosen.drop_duplicates("gene").set_index("gene")["clone"]
        cols = [b for b in NP_ORDER if b in set(t["clone"])]
        genes = sorted(peak.index, key=lambda x: (cols.index(peak[x]), x))
        g = t[t["gene"].isin(genes)]
        value = g.pivot(index="gene", columns="clone", values="specificity")
        value = value.reindex(index=genes, columns=cols)
        dots = g.pivot(index="gene", columns="clone", values="expressed")
        dots = dots.reindex(index=genes, columns=cols).fillna(False)
        info = g.drop_duplicates("gene").set_index("gene", drop=False)
        labels = [gene_label(info.loc[x]) for x in genes]
        fig, ax = plt.subplots(
            figsize=(0.28 * len(cols) + 3.5, 0.16 * len(genes) + 1.6),
        )
        im = ax.imshow(
            value.clip(-3, 3).to_numpy(dtype=float),
            cmap=cmap,
            vmin=-3,
            vmax=3,
            aspect="auto",
            interpolation="nearest",
        )
        i, j = np.nonzero(dots.to_numpy(dtype=bool))
        ax.scatter(j, i, s=3, color=INK, linewidths=0)
        names = t.drop_duplicates("clone").set_index("clone")["descendants"]
        ticks = [names.get(b) or b for b in cols]
        ax.set_xticks(range(len(cols)), ticks, rotation=90)
        ax.set_yticks(range(len(genes)), labels, fontsize=5)
        xlabel = f"{st} cells"
        if st != ANCHOR_STAGE:
            xlabel += f", grouped by the {ANCHOR_STAGE} cell they come from"
        ax.set_xlabel(xlabel)
        style(ax, grid_axis=None)
        cb = fig.colorbar(im, ax=ax, shrink=0.5)
        cb.set_label("log2 over the plate mean", color=MUTED, fontsize=7)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_visible(False)
        ax.legend(
            handles=[plt.Line2D([], [], marker="o", ls="", ms=2, color=INK)],
            labels=[f"expressed (support ≥ {EXPRESSED_MIN:g})"],
            loc="upper left",
            bbox_to_anchor=(1.0, 0.1),
            frameon=False,
            fontsize=7,
        )
        ax.set_title(
            f"Predicted expression at {st} across the neural plate",
            loc="left",
            fontsize=9,
            color=INK,
        )
        save(fig, out, f"{st}_clone_expression")


def plot_expression_maps(
    expression: dict[str, pd.DataFrame],
    tops: pd.DataFrame,
    bench: pd.DataFrame,
    stage_maps: dict[str, gpd.GeoDataFrame],
    out: str,
) -> None:
    """Draw per stage, on that stage's map (the midG map when it has none):
    the number of specific predicted genes per clone, labelled with the top
    one ({out}_{stage}_top_gene_map); and log1p CP10k, on one scale per gene
    across stages, of each predicted gene (MAP_PER_CLONE per clone plus
    MAP_GENES) and of each ANISEED gene at its benchmark stage with the in
    situ cells outlined ({out}_{stage}_gene_{gene}).

    :param expression: Clone expression per stage, from predict_expression.
    :type expression: dict[str, pandas.DataFrame]
    :param tops: Top genes from top_genes.
    :type tops: pandas.DataFrame
    :param bench: Per-gene rows from expression_benchmark.
    :type bench: pandas.DataFrame
    :param stage_maps: Neural plate maps from load_stage_maps.
    :type stage_maps: dict[str, geopandas.GeoDataFrame]
    :param out: Output file prefix.
    :type out: str
    """
    peak = pd.concat(
        [t.groupby("gene")["log_cp10k"].max() for t in expression.values()],
        axis=1,
    ).max(axis=1)
    truth = {}
    if len(bench):
        ok = bench[bench["tested"]]
        for st, gene, cells in zip(ok["stage"], ok["gene"], ok["cells"]):
            truth[st, gene] = cells.split("/")
    predicted = list(MAP_GENES)
    if len(tops):
        picked = tops.loc[tops["rank"] <= MAP_PER_CLONE, "gene"]
        predicted = list(dict.fromkeys(list(picked) + predicted))
    for st, t in expression.items():
        m = stage_maps.get(st, stage_maps[ANCHOR_STAGE])
        first = tops[(tops["stage"] == st) & (tops["rank"] == 1)]
        first = first.set_index("clone")
        notes = first["gene"].str.replace(r"^KY21\.", "", regex=True)
        if "gene_name" in first:
            names = first["gene_name"].map(
                lambda n: short_name(n) if isinstance(n, str) else "",
            )
            notes = names.where(names != "", notes)
        novel = t["specific"] & ~t["known"]
        n_specific = novel.groupby(t["clone"]).sum()
        title = f"Most specific predicted gene at {st}"
        if st != ANCHOR_STAGE:
            title += f", per {ANCHOR_STAGE} clone"
        plot_neural_plate(
            m,
            n_specific,
            "specific predicted genes",
            title,
            out,
            f"{st}_top_gene_map",
            notes=notes,
            key="clone",
        )

        jobs = {gene: None for gene in predicted}
        for (s, gene), cells in truth.items():
            if s == st:
                jobs[gene] = cells
        for gene, cells in jobs.items():
            g = t[t["gene"] == gene]
            if not len(g):
                continue
            level = g.set_index("clone")["log_cp10k"]
            title = f"{gene_label(g.iloc[0])} at {st}"
            outline = None
            if cells is not None:
                outline = {n for n in m["name"] if covered(n, cells)}
            plot_neural_plate(
                m,
                level,
                "expected expression, log1p(CP10k)",
                title,
                out,
                f"{st}_gene_{file_safe(gene)}",
                vmax=peak[gene],
                outline=outline,
                outline_label=f"ANISEED in situ, {st}",
                key="clone",
            )


def plot_expression_benchmark(summary: pd.DataFrame, out: str) -> None:
    """Draw the mean precision, recall and AUROC of the expression benchmark
    per stage ({out}_expression_benchmark).

    :param summary: Per-stage means from expression_benchmark.
    :type summary: pandas.DataFrame
    :param out: Output file prefix.
    :type out: str
    """
    if not len(summary):
        return
    shown = ["precision", "recall", "auroc"]
    x = np.arange(len(summary))
    width = 0.8 / len(shown)
    fig, ax = plt.subplots(figsize=(1.2 * len(summary) + 2.5, 3.6))
    for i, (m, color) in enumerate(zip(shown, SERIES)):
        ax.bar(
            x + (i - 1) * width,
            summary[m],
            width=width,
            color=color,
            edgecolor=SURFACE,
            linewidth=1,
            label=m.upper() if m == "auroc" else m,
        )
    ax.axhline(
        0.5,
        color=MUTED,
        linewidth=1,
        linestyle="--",
        label="AUROC by chance",
    )
    ticks = []
    for st, row in summary.iterrows():
        tick = f"{st}\n{int(row['n_tested'])}/{int(row['n_genes'])} genes"
        if row["circular"]:
            tick += "\n(anchoring markers)"
        ticks.append(tick)
    ax.set_xticks(x, ticks)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean over ANISEED genes")
    style(ax)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.set_title(
        "Predicted expression against ANISEED in situ patterns",
        loc="left",
        fontsize=9,
        color=INK,
    )
    save(fig, out, "expression_benchmark")


# %% Pipeline
def analyze(
    adata: ad.AnnData,
    net: pd.DataFrame,
    homologs: pd.DataFrame,
    stage_maps: dict[str, gpd.GeoDataFrame],
    stage_markers: pd.DataFrame,
    known_genes: set[str],
    out: str,
) -> dict:
    """Run every step on one set of cells, writing each step's tables and
    figures as soon as it finishes, and the cells with their stage clusters,
    per-stage UMAP and np_blastomeres to {out}_lineage.h5ad.

    :param adata: Cells; modified in place.
    :type adata: anndata.AnnData
    :param net: Territory network from load_territory_net.
    :type net: pandas.DataFrame
    :param homologs: Homologs from load_homologs.
    :type homologs: pandas.DataFrame
    :param stage_maps: Neural plate maps from load_stage_maps.
    :type stage_maps: dict[str, geopandas.GeoDataFrame]
    :param stage_markers: ANISEED genes from load_stage_markers.
    :type stage_markers: pandas.DataFrame
    :param known_genes: Genes with ANISEED in situ data.
    :type known_genes: set[str]
    :param out: Output file prefix.
    :type out: str
    :returns: Every step's result, keyed by name.
    :rtype: dict
    """
    res = {"adata": adata}
    stages = res["stages"] = present_stages(adata)
    print(pd.crosstab(adata.obs[STAGE_KEY], adata.obs[BATCH_KEY]).to_string())
    neural_plate = stage_maps[ANCHOR_STAGE]

    cluster_stages(adata, stages)
    plot_stage_umaps(adata, stages, out)

    res["de"] = differential_expression(adata, stages, homologs)
    for st, de in res["de"].items():
        write_csv(de, out, f"{st}_de", index=False)

    res["tp"], res["sub"] = solve_transport(adata, stages)
    clusters_at = res["clusters_at"] = clusters_by_stage(adata, stages)
    edges = res["edges"] = transport_edges(res["tp"], stages, clusters_at)
    nodes = res["nodes"] = cluster_table(adata)
    res["graph"] = transport_graph(nodes, edges)
    write_csv(edges, out, "transport_edges", index=False)
    nx.write_graphml(res["graph"], f"{out}_transport_graph.graphml")
    plot_transport_heatmaps(edges, stages, clusters_at, out)
    plot_transport_network(nodes, edges, stages, clusters_at, out)

    es, pv, calls, cluster_pick = anchor_clusters(adata, net, nodes)
    neural_clusters = list(calls.index[calls["neural_plate"]])
    res.update(
        es=es,
        pv=pv,
        calls=calls,
        cluster_pick=cluster_pick,
        neural_clusters=neural_clusters,
    )
    write_csv(es, out, f"{ANCHOR_STAGE}_territory_scores")
    write_csv(calls, out, f"{ANCHOR_STAGE}_territory_calls")

    blastomeres = sorted(
        set(NP_GRID) | set(neural_plate["name"]),
        key=blastomere_key,
    )
    full, best = best_cluster_per_blastomere(es, pv, nodes, blastomeres)
    res.update(full=full, best=best)
    write_csv(full, out, "blastomere_cluster_scores", index=False)
    write_csv(best, out, "blastomere_best_cluster")
    plot_neural_plate(
        neural_plate,
        best["score"],
        "ULM score (t) of the best cluster",
        f"Best {ANCHOR_STAGE} cluster for each blastomere",
        out,
        f"{ANCHOR_STAGE}_blastomere_map",
        clusters=best["stage_cluster"],
    )

    res["winkley"] = winkley_midg_benchmark(adata, cluster_pick, best)
    _, by_label, by_blastomere, summary = res["winkley"]
    write_csv(by_label, out, f"{ANCHOR_STAGE}_winkley_labels")
    write_csv(by_blastomere, out, f"{ANCHOR_STAGE}_winkley_blastomeres")
    write_csv(summary, out, f"{ANCHOR_STAGE}_winkley_summary", index=False)
    plot_winkley_labels(by_label, out)
    if len(by_label):
        plot_neural_plate(
            neural_plate,
            by_blastomere["score"],
            "mean score against Winkley's labels",
            f"{ANCHOR_STAGE} benchmark by blastomere",
            out,
            f"{ANCHOR_STAGE}_winkley_map",
            vmax=1,
        )

    res["identities"], node_identity = predict_identities(
        edges,
        cluster_pick,
        clusters_at,
        nodes,
        stages,
        stage_maps,
    )
    res["adjacency"] = adjacency_table(nodes, edges, node_identity, stages)
    write_csv(res["identities"], out, "predicted_identities")
    write_csv(res["adjacency"], out, "adjacency", sep="\t")

    res["clone_shares"] = clone_shares(
        edges,
        cluster_pick,
        clusters_at,
        stages,
    )
    for st, share in res["clone_shares"].items():
        write_csv(share, out, f"{st}_clone_shares")

    res["expression"] = predict_expression(
        res["de"],
        res["clone_shares"],
        nodes,
        stage_maps,
        known_genes,
    )
    for st, t in res["expression"].items():
        expressed = t[t["n_clones"] > 0]
        write_csv(expressed, out, f"{st}_clone_expression", index=False)
    plot_expression_heatmaps(res["expression"], out)

    res["top_genes"] = top_genes(res["expression"])
    write_csv(res["top_genes"], out, "clone_top_genes", index=False)

    res["expression_benchmark"] = expression_benchmark(
        res["expression"],
        stage_markers,
    )
    bench, bench_summary = res["expression_benchmark"]
    write_csv(bench, out, "expression_benchmark", index=False)
    write_csv(bench_summary, out, "expression_benchmark_summary")
    plot_expression_benchmark(bench_summary, out)
    plot_expression_maps(
        res["expression"],
        res["top_genes"],
        bench,
        stage_maps,
        out,
    )

    res["backward"] = backward_benchmark(
        adata,
        res["tp"],
        res["sub"],
        cluster_pick,
        neural_clusters,
        stages,
    )
    backward, backward_shares, backward_summary = res["backward"]
    write_csv(backward, out, "backward_benchmark", index=False)
    write_csv(backward_shares, out, "backward_label_shares", index=False)
    write_csv(backward_summary, out, "backward_summary")
    plot_backward(backward, backward_shares, calls, out)

    res["forward"] = forward_benchmark(
        adata,
        res["tp"],
        res["sub"],
        cluster_pick,
        neural_clusters,
        clusters_at,
        stages,
    )
    forward, forward_shares, forward_summary = res["forward"]
    write_csv(forward, out, "forward_benchmark", index=False)
    write_csv(forward_shares, out, "forward_tissue_shares", index=False)
    write_csv(forward_summary, out, "forward_summary")
    plot_forward(forward, forward_shares, stages, out)

    picks = adata.obs["stage_cluster"].map(calls["blastomeres"])
    adata.obs["np_blastomeres"] = picks.astype(object)
    for col in ("stage_leiden", "stage_cluster", "np_blastomeres"):
        adata.obs[col] = pd.Categorical(adata.obs[col].astype(object))
    adata.write_h5ad(f"{out}_lineage.h5ad")
    return res


# %% All cells
net = load_territory_net()
homologs = load_homologs()
stage_maps = load_stage_maps()
stage_markers = load_stage_markers()
known_genes = load_known_genes()
adata = sc.read_h5ad(f"{PREFIX}_{GENOME}.h5ad")
adata = adata[adata.obs[STAGE_KEY].isin(CHAIN).to_numpy()].copy()
res = analyze(
    adata,
    net,
    homologs,
    stage_maps,
    stage_markers,
    known_genes,
    OUT,
)

# %% Ancestors and descendants of the midG neural plate clusters
np_lineage = neural_plate_lineage(
    res["edges"],
    res["stages"],
    res["clusters_at"],
    res["neural_clusters"],
    res["nodes"],
)
write_csv(np_lineage, OUT, "np_lineage_clusters")

# %% Repeat on the neural plate lineage
selected = np_lineage.index[np_lineage["selected"]]
adata_np = adata[adata.obs["stage_cluster"].isin(selected).to_numpy()].copy()
adata_np.obs["stage_cluster_all"] = adata_np.obs["stage_cluster"].astype(str)
adata_np.obs = adata_np.obs.drop(
    columns=[
        "stage_leiden",
        "stage_cluster",
        "np_blastomeres",
    ],
)
del adata_np.obsm["X_umap_stage"]
res_np = analyze(
    adata_np,
    net,
    homologs,
    stage_maps,
    stage_markers,
    known_genes,
    f"{OUT}_np",
)
