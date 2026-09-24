# %% [markdown]
# Stage-by-stage clusters of the pooled Cao, Sharma and Winkley cells, linked
# by optimal transport (moscot) from the 64-cell stage to larva. Mid-gastrula
# clusters are anchored to the ANISEED stage-12 neural plate map (pass_02.tsv)
# and checked against Winkley's labels at midG, backward to c64 and iniG, and
# forward against Cao's tissue labels. The whole process runs twice: on all
# cells, then on the clusters that the transport network places upstream or
# downstream of the midG neural plate clusters. Analysis functions return
# objects; the write_ and plot_ helpers put them on disk.

# %% Setup
import re

import anndata as ad
import decoupler as dc
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from matplotlib.colors import LinearSegmentedColormap
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
ANCHOR_MAP = "npisc/mid_gastrula.geojson"
MARKER_STAGES = ["Stage 12 (mid gastrula)"]

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
BENCH_GEN = {"c64": 7, "iniG": 8}

RESOLUTION = 1.0
RESOLUTION_BY_STAGE = {"midG": 2.0}
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
IDENTITY_GEN = {"midG": 9, "earN": 10, "latN": 11}
LINEAGE_MIN = 0.5
DPI = 300
METRICS = ["precision", "recall", "jaccard", "bits", "score", "gain"]
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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging",
    ["#184f95", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"],
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


# %% Helpers
def to_generation(label, gen):
    """Blastomeres named in a label, moved to one generation: ancestors of
    later cells, descendants of earlier ones."""
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


def winkley_blastomeres(label):
    """Blastomeres covered by a Winkley neural plate grid label."""
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


def set_scores(pick, truth, n=len(NP_GRID)):
    """Two blastomere sets: precision-weighted bits of our pick, and bits
    gained over the truth."""
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


def num(node):
    """Leiden number of a "{stage}_{leiden}" cluster name."""
    return int(node.rsplit("_", 1)[-1])


def labels_of(adata, key):
    """One obs column as strings, with missing values as empty strings."""
    return adata.obs[key].astype(object).fillna("").astype(str).to_numpy()


def share_list(e, key, share):
    """Children or parents of each cluster as "cluster:share" text, strongest
    first."""
    e = e[e[share] >= MIN_EDGE].sort_values(share, ascending=False)
    other = "to" if key == "from" else "from"

    def pairs(g):
        return ",".join(f"{n}:{v:.2f}" for n, v in zip(g[other], g[share]))

    return e.groupby(key)[[other, share]].apply(pairs)


def plain(v):
    """A numpy scalar as a plain Python value, other values unchanged."""
    return v.item() if hasattr(v, "item") else v


def edges_between(edges, sources, targets):
    """Edges from any of the source clusters to any of the target clusters."""
    keep = edges["from"].isin(sources) & edges["to"].isin(targets)
    return edges[keep]


def early_blastomeres(label, gen):
    """Blastomeres of a Winkley label at one generation, translating the
    early neural labels through WINKLEY_EARLY."""
    return to_generation(WINKLEY_EARLY.get(label, label), gen)


def fmt(weights, keep=None):
    """A {name: weight} dict as "name:weight" text, strongest first."""
    items = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    if keep is None:
        items = [(k, v) for k, v in items if v >= MIN_SHOW]
    else:
        items = [(k, v) for k, v in items if k in keep]
    return ",".join(f"{k}:{v:.2f}" for k, v in items)


def style(ax, grid_axis="y"):
    """Recessive axes, muted ticks and a hairline grid."""
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


def save(fig, out, name):
    """Write a figure as {out}_{name}.png and close it."""
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(f"{out}_{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_csv(table, out, name, **kwargs):
    """Write a table as {out}_{name}.csv, or .tsv when sep is a tab."""
    ext = "tsv" if kwargs.get("sep") == "\t" else "csv"
    table.to_csv(f"{out}_{name}.{ext}", **kwargs)


# %% Inputs
def load_territory_net():
    """ANISEED stage-12 marker genes as a decoupler network in KY21 gene IDs.

    Territories with identical marker sets are merged into one source named
    like "a9.35/a9.36/a9.39/a9.40".
    """
    kh2ky = pd.read_csv(GENE_MAP, sep="\t").set_index("KH2012")["KY2021"]
    mk = pd.read_csv(MARKERS, sep="\t")
    mk = mk[mk["Stage"].isin(MARKER_STAGES)]
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


def load_homologs():
    """SwissProt homolog of each KY21 gene, with evidence tags removed."""
    homologs = pd.read_csv(HOMOLOGS).set_index("KY2021")
    homologs["homolog"] = homologs["fullname"].str.replace(
        r"\s*\{ECO:[^}]*\}",
        "",
        regex=True,
    )
    return homologs[["KH2012", "uniprot", "homolog"]]


def load_neural_plate():
    """Mid-gastrula neural plate cells as polygons, left and right sides named
    alike."""
    neural_plate = gpd.read_file(ANCHOR_MAP)
    neural_plate["name"] = neural_plate["name"].str.rstrip("*")
    return neural_plate


# %% Clusters per stage
def present_stages(adata):
    """Stages of CHAIN that have cells, in developmental order."""
    return [
        s
        for s in adata.obs[STAGE_KEY].cat.categories
        if s in CHAIN and (adata.obs[STAGE_KEY] == s).any()
    ]


def cluster_stages(adata, stages):
    """Leiden clusters and a UMAP computed separately for each stage.

    Neighbours use the scVI latent space of that stage's cells only. Adds
    obs["stage_leiden"] (every Leiden cluster), obs["stage_cluster"]
    ("{stage}_{leiden}", missing for clusters under MIN_CLUSTER cells) and
    obsm["X_umap_stage"]. Returns the per-cell cluster names.
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
            resolution=RESOLUTION_BY_STAGE.get(st, RESOLUTION),
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
def differential_expression(adata, stages, homologs):
    """scVI DE between the clusters of each stage, checked within each source.

    Each cluster is compared with the other clusters of its stage using the
    saved scVI model with batch correction. Sources with two or more clusters
    of MIN_DE_CELLS cells repeat the test with Wilcoxon on log-normalized
    counts. A gene is replicated when scVI calls it and every tested source
    agrees in direction at FDR DE_FDR. Returns {stage: DataFrame}."""
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
                columns={"group": "stage_cluster", "names": "gene"},
            )
            r = r[["stage_cluster", "gene", "logfoldchanges", "pvals_adj"]]
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
def solve_transport(adata, stages):
    """One moscot TemporalProblem over all stages on the scVI latent space.

    Returns the solved problem and the AnnData it holds, whose obs carries
    "time" and "node" (the stage cluster, "none" when unclustered).
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


def clusters_by_stage(adata, stages):
    """Stage clusters of each stage, ordered by Leiden number."""
    cluster = labels_of(adata, "stage_cluster")
    obs_stage = labels_of(adata, STAGE_KEY)
    out = {}
    for st in stages:
        out[st] = sorted({c for c in cluster[obs_stage == st] if c}, key=num)
    return out


def transport_edges(tp, stages, clusters_at):
    """Transported mass between clusters of adjacent stages.

    "fwd" is the share of the source cluster's mass sent to the target and
    "bwd" the share of the target cluster's mass received from the source.
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


def cluster_table(adata):
    """Per cluster: stage, cells per source, and majority NODE_TISSUE_KEY
    tissue with its share."""
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


def transport_graph(nodes, edges):
    """Directed graph of clusters, keeping edges with a forward or backward
    share of MIN_EDGE or more."""
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
def anchor_clusters(adata, net, nodes):
    """Score every midG cluster against the ANISEED stage-12 territories.

    Genes are standardized within each source, averaged per cluster and scored
    with decoupler ULM over the integration genes plus the marker panel. A
    cluster is neural plate when its best territory is positive at FDR
    MAX_PADJ; its blastomere set is every territory within MIN_MARGIN of the
    best. Returns the score and p-value matrices, the call table and the
    blastomere set of each cluster.
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


def best_cluster_per_blastomere(es, pv, nodes, blastomeres):
    """Rank the midG clusters for each blastomere by its territory score.

    Returns every cluster-blastomere score with its rank, and one row per
    blastomere with the best and second-best cluster.
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
def winkley_midg_benchmark(adata, cluster_pick, best):
    """Compare Winkley's midG grid labels with the blastomere set of each
    cell's cluster.

    Uses set_scores per cell. Returns the per-cell table, means per Winkley
    label, means per blastomere and the overall means."""
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


def predict_identities(edges, cluster_pick, clusters_at, nodes, stages):
    """Carry midG blastomere sets to the stages in IDENTITY_GEN.

    Each cluster takes its parents' blastomeres in proportion to the backward
    share of each edge, and every blastomere splits into its daughters.
    Returns one row per cluster at those stages and each cluster's identity.
    """
    anchor_gen = IDENTITY_GEN[ANCHOR_STAGE]
    identity = {}
    for c, bs in cluster_pick.items():
        if bs:
            identity[c] = {b: 1 / len(bs) for b in bs}
    node_identity = {c: "/".join(sorted(w)) for c, w in identity.items()}
    rows = []
    i0 = stages.index(ANCHOR_STAGE)
    later = [s for s in stages[i0 + 1 :] if s in IDENTITY_GEN]
    for st in later:
        gen = IDENTITY_GEN[st]
        nxt = {}
        step = edges_between(edges, list(identity), clusters_at[st])
        for rec in step.to_dict("records"):
            weights = nxt.setdefault(rec["to"], {})
            for b, w in identity[rec["from"]].items():
                kids = sorted(to_generation(b, gen))
                for d in kids:
                    add = rec["bwd"] * w / len(kids)
                    weights[d] = weights.get(d, 0.0) + add
        identity = nxt
        for n in clusters_at[st]:
            w = identity.get(n, {})
            anc = {}
            for d, v in w.items():
                (a,) = to_generation(d, anchor_gen)
                anc[a] = anc.get(a, 0.0) + v
            kept = {a for a, v in anc.items() if v >= MIN_SHOW}
            top = max(sorted(anc), key=anc.get) if anc else ""
            kids = sorted(to_generation(top, gen)) if top else []
            predicted = "/".join(kids)
            keep = {d for d in w if to_generation(d, anchor_gen) & kept}
            rows.append(
                {
                    "stage_cluster": n,
                    "stage": st,
                    "generation": gen,
                    "tissue": nodes.loc[n, "tissue"],
                    "assigned_share": sum(w.values()),
                    "top_ancestor": top,
                    "top_ancestor_share": anc.get(top, 0.0),
                    "predicted": predicted,
                    "ancestors": fmt(anc),
                    "identities": fmt(w, keep=keep),
                },
            )
            node_identity[n] = predicted
    identities = pd.DataFrame(rows).set_index("stage_cluster")
    node_identity = pd.Series(node_identity).reindex(nodes.index).fillna("")
    return identities, node_identity


def adjacency_table(nodes, edges, node_identity, stages):
    """One row per cluster with its tissue, identity, children and parents."""
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


def backward_benchmark(adata, tp, sub, cluster_pick, neural_clusters, stages):
    """Winkley labels among the transported ancestors of each midG neural
    plate cluster.

    For each stage in BENCH_GEN, the ancestry of every neural plate cluster is
    spread over Winkley's labelled cells and compared with each label's
    background share. Labels are expected when they cover the ancestor of the
    cluster's blastomeres. Returns per-cluster rows, per-label shares and per-
    stage means."""
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
    adata,
    tp,
    sub,
    cluster_pick,
    neural_clusters,
    clusters_at,
    stages,
):
    """Cao tissue labels among the transported descendants of each midG
    cluster.

    For each later stage, every midG cluster's descendants are spread over
    Cao's published tissue labels and compared with each tissue's background
    share. Returns per-cluster rows, per-tissue shares and means per stage and
    neural plate status."""
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


def neural_plate_lineage(edges, stages, clusters_at, neural_clusters, nodes):
    """Share of each cluster that lies upstream or downstream of the midG
    neural plate clusters.

    midG clusters score 1 when neural plate and 0 otherwise. Later clusters
    take the backward-share-weighted score of their parents, earlier clusters
    the forward-share-weighted score of their children. Clusters scoring
    LINEAGE_MIN or more are selected."""
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


def analyze(adata, net, homologs, neural_plate):
    """Run every analysis step on one set of cells and return the results.

    Adds stage clusters and the per-stage UMAP to adata. Writes nothing.
    """
    res = {"adata": adata}
    stages = res["stages"] = present_stages(adata)
    print(pd.crosstab(adata.obs[STAGE_KEY], adata.obs[BATCH_KEY]).to_string())
    cluster_stages(adata, stages)
    res["de"] = differential_expression(adata, stages, homologs)
    res["tp"], res["sub"] = solve_transport(adata, stages)
    clusters_at = res["clusters_at"] = clusters_by_stage(adata, stages)
    edges = res["edges"] = transport_edges(res["tp"], stages, clusters_at)
    nodes = res["nodes"] = cluster_table(adata)
    res["graph"] = transport_graph(nodes, edges)
    res["es"], res["pv"], res["calls"], cluster_pick = anchor_clusters(
        adata,
        net,
        nodes,
    )
    res["cluster_pick"] = cluster_pick
    neural_clusters = res["neural_clusters"] = list(
        res["calls"].index[res["calls"]["neural_plate"]],
    )
    blastomeres = sorted(
        set(NP_GRID) | set(neural_plate["name"]),
        key=lambda b: (b[0], int(b.split(".")[1])),
    )
    res["full"], res["best"] = best_cluster_per_blastomere(
        res["es"],
        res["pv"],
        nodes,
        blastomeres,
    )
    res["winkley"] = winkley_midg_benchmark(adata, cluster_pick, res["best"])
    res["identities"], node_identity = predict_identities(
        edges,
        cluster_pick,
        clusters_at,
        nodes,
        stages,
    )
    res["adjacency"] = adjacency_table(nodes, edges, node_identity, stages)
    res["backward"] = backward_benchmark(
        adata,
        res["tp"],
        res["sub"],
        cluster_pick,
        neural_clusters,
        stages,
    )
    res["forward"] = forward_benchmark(
        adata,
        res["tp"],
        res["sub"],
        cluster_pick,
        neural_clusters,
        clusters_at,
        stages,
    )
    picks = adata.obs["stage_cluster"].map(res["calls"]["blastomeres"])
    adata.obs["np_blastomeres"] = picks.astype(object)
    return res


# %% Plots
def plot_stage_umaps(adata, stages, out):
    """One UMAP per stage coloured by that stage's Leiden clusters."""
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


def plot_transport_heatmaps(edges, stages, clusters_at, out):
    """One heatmap per pair of adjacent stages of the forward transport
    share."""
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


def plot_transport_network(nodes, edges, stages, clusters_at, out):
    """Clusters in stage columns, coloured by tissue, linked by forward share
    of PLOT_MIN_EDGE or more."""
    shown = edges[edges["fwd"] >= PLOT_MIN_EDGE]
    tissue_rank = {t: i for i, t in enumerate(TISSUE_COLORS)}

    def rank(n):
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
    neural_plate,
    values,
    label,
    title,
    out,
    name,
    clusters=None,
    vmax=None,
):
    """Neural plate map shaded by one value per blastomere, optionally with a
    cluster number in each cell."""
    gdf = neural_plate.join(values.rename("value"), on="name")
    if clusters is not None:
        gdf = gdf.join(clusters.rename("stage_cluster"), on="name")
    fig, ax = plt.subplots(figsize=(7, 4.8))
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
    top = vmax or gdf["value"].max()
    for i, (pt, cell_name, v) in enumerate(
        zip(gdf.representative_point(), gdf["name"], gdf["value"]),
    ):
        ink = "white" if v > 0.55 * top else INK
        cl = gdf["stage_cluster"].iloc[i] if clusters is not None else None
        ax.annotate(
            cell_name,
            (pt.x, pt.y),
            xytext=(0, 7 if isinstance(cl, str) else 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=5.5,
            color=ink,
        )
        if isinstance(cl, str):
            ax.annotate(
                str(num(cl)),
                (pt.x, pt.y),
                xytext=(0, -3),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=ink,
            )
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    save(fig, out, name)


def plot_winkley_labels(by_label, out):
    """Precision, recall and score of the midG picks for each Winkley grid
    label."""
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


def plot_backward(backward, shares, calls, out):
    """Per early stage: label enrichment heatmap and expected-ancestry
    scatter."""
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


def plot_forward(forward, shares, stages, out):
    """Nervous system share of midG descendants per stage, and their tissue
    mix."""
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


# %% Writers
def write_results(res, out):
    """Write the tables, graph and AnnData of one analyze() result."""
    for st, de in res["de"].items():
        write_csv(de, out, f"{st}_de", index=False)
    write_csv(res["edges"], out, "transport_edges", index=False)
    nx.write_graphml(res["graph"], f"{out}_transport_graph.graphml")
    write_csv(res["es"], out, f"{ANCHOR_STAGE}_territory_scores")
    write_csv(res["calls"], out, f"{ANCHOR_STAGE}_territory_calls")
    write_csv(res["full"], out, "blastomere_cluster_scores", index=False)
    write_csv(res["best"], out, "blastomere_best_cluster")
    _, by_label, by_blastomere, summary = res["winkley"]
    write_csv(by_label, out, f"{ANCHOR_STAGE}_winkley_labels")
    write_csv(by_blastomere, out, f"{ANCHOR_STAGE}_winkley_blastomeres")
    write_csv(summary, out, f"{ANCHOR_STAGE}_winkley_summary", index=False)
    write_csv(res["identities"], out, "predicted_identities")
    write_csv(res["adjacency"], out, "adjacency", sep="\t")
    backward, backward_shares, backward_summary = res["backward"]
    write_csv(backward, out, "backward_benchmark", index=False)
    write_csv(backward_shares, out, "backward_label_shares", index=False)
    write_csv(backward_summary, out, "backward_summary")
    forward, forward_shares, forward_summary = res["forward"]
    write_csv(forward, out, "forward_benchmark", index=False)
    write_csv(forward_shares, out, "forward_tissue_shares", index=False)
    write_csv(forward_summary, out, "forward_summary")
    adata = res["adata"]
    for col in ("stage_leiden", "stage_cluster", "np_blastomeres"):
        adata.obs[col] = pd.Categorical(adata.obs[col].astype(object))
    adata.write_h5ad(f"{out}_lineage.h5ad")


def plot_results(res, neural_plate, out):
    """Draw every figure of one analyze() result."""
    stages, clusters_at = res["stages"], res["clusters_at"]
    plot_stage_umaps(res["adata"], stages, out)
    plot_transport_heatmaps(res["edges"], stages, clusters_at, out)
    plot_transport_network(
        res["nodes"],
        res["edges"],
        stages,
        clusters_at,
        out,
    )
    plot_neural_plate(
        neural_plate,
        res["best"]["score"],
        "ULM score (t) of the best cluster",
        f"Best {ANCHOR_STAGE} cluster for each blastomere",
        out,
        f"{ANCHOR_STAGE}_blastomere_map",
        clusters=res["best"]["stage_cluster"],
    )
    _, by_label, by_blastomere, _ = res["winkley"]
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
    backward, backward_shares, _ = res["backward"]
    plot_backward(backward, backward_shares, res["calls"], out)
    forward, forward_shares, _ = res["forward"]
    plot_forward(forward, forward_shares, stages, out)


# %% All cells
net = load_territory_net()
homologs = load_homologs()
neural_plate = load_neural_plate()
adata = sc.read_h5ad(f"{PREFIX}_{GENOME}.h5ad")
adata = adata[adata.obs[STAGE_KEY].isin(CHAIN).to_numpy()].copy()
res = analyze(adata, net, homologs, neural_plate)
write_results(res, OUT)
plot_results(res, neural_plate, OUT)

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
    columns=["stage_leiden", "stage_cluster", "np_blastomeres"],
)
del adata_np.obsm["X_umap_stage"]
res_np = analyze(adata_np, net, homologs, neural_plate)
write_results(res_np, f"{OUT}_np")
plot_results(res_np, neural_plate, f"{OUT}_np")
