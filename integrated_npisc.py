# %% [markdown]
# Stage-by-stage clusters of the pooled Cao, Sharma and Winkley cells, linked
# by optimal transport (moscot) from the 64-cell stage to larva. Mid-gastrula
# clusters are anchored to the ANISEED stage-12 neural plate map (pass_02.tsv)
# and checked against Winkley's labels at midG, backward to c64 and iniG, and
# forward against Cao's tissue labels.

# %% Setup
import re

import decoupler as dc
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
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
DPI = 300
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

adata = sc.read_h5ad(f"{PREFIX}_{GENOME}.h5ad")
adata = adata[adata.obs[STAGE_KEY].isin(CHAIN).to_numpy()].copy()
stages = [s for s in adata.obs[STAGE_KEY].cat.categories if s in CHAIN]
t_of = {s: float(i) for i, s in enumerate(stages)}
obs_src = adata.obs[BATCH_KEY].astype(str).to_numpy()
obs_stage = adata.obs[STAGE_KEY].astype(str).to_numpy()
sources = sorted(set(obs_src))
print(pd.crosstab(adata.obs[STAGE_KEY], adata.obs[BATCH_KEY]).to_string())


# %% Helpers
def to_generation(label, gen):
    """Blastomeres named in a label, moved to one generation: ancestors of later cells, descendants of earlier ones."""
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
    """Two blastomere sets: precision-weighted bits of our pick, and bits gained
    over the truth."""
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
    return int(node.rsplit("_", 1)[-1])


def labels_of(key):
    return adata.obs[key].astype(object).fillna("").astype(str).to_numpy()


# %% Clusters and UMAP per stage
adata.obs["stage_cluster"] = pd.Series(pd.NA, index=adata.obs_names, dtype=object)
adata.obsm["X_umap_stage"] = np.full((adata.n_obs, 2), np.nan)
col = adata.obs.columns.get_loc("stage_cluster")
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
    cl = adata_sub.obs["leiden"].astype(str)
    small = cl.map(cl.value_counts()).astype(int) < MIN_CLUSTER
    adata.obs.iloc[idx, col] = (st + "_" + cl).mask(small).to_numpy()
    adata.obsm["X_umap_stage"][idx] = adata_sub.obsm["X_umap"]

    adata_sub.uns.pop("leiden_colors", None)
    fig = sc.pl.umap(
        adata_sub,
        color="leiden",
        legend_loc="on data",
        legend_fontsize=6,
        legend_fontoutline=2,
        outline_color="white",
        frameon=False,
        title=f"{st} Leiden clusters",
        return_fig=True,
    )
    fig.savefig(f"{OUT}_{st}_umap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
cluster = labels_of("stage_cluster")
print(
    adata.obs.groupby(STAGE_KEY, observed=True)["stage_cluster"]
    .nunique()
    .reindex(stages)
    .to_string()
)

# %% Differential expression between clusters at each stage
homologs = pd.read_csv(HOMOLOGS).set_index("KY2021")
homologs["homolog"] = homologs["fullname"].str.replace(
    r"\s*\{ECO:[^}]*\}", "", regex=True
)
IS_DE = f"is_de_fdr_{DE_FDR}"
for st in stages:
    cells = adata[(obs_stage == st) & (cluster != "")].copy()
    cells.obs["stage_cluster"] = pd.Categorical(cells.obs["stage_cluster"].astype(str))
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
        de.rename_axis("gene").reset_index().rename(columns={"group1": "stage_cluster"})
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
            columns={"group": "stage_cluster", "names": "gene"}
        )
        de = de.merge(
            r[["stage_cluster", "gene", "logfoldchanges", "pvals_adj"]].rename(
                columns={"logfoldchanges": f"lfc_{src}", "pvals_adj": f"padj_{src}"},
            ),
            on=["stage_cluster", "gene"],
            how="left",
        )
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
    de = de.join(homologs[["KH2012", "uniprot", "homolog"]], on="gene")
    front = ["stage_cluster", "gene", "KH2012", "uniprot", "homolog"]
    de = de[front + [c for c in de if c not in front]]
    if "gene_name" in adata.var:
        de.insert(2, "gene_name", de["gene"].map(adata.var["gene_name"]))
    de["order"] = de["stage_cluster"].map(num)
    de = de.sort_values(
        ["order", IS_DE, "lfc_mean"], ascending=[True, False, False]
    ).drop(columns="order")
    de.to_csv(f"{OUT}_{st}_de.csv", index=False)
    print(f"{st}: per-source check in {present}")
    print(
        de.assign(up=de[IS_DE] & (de["lfc_mean"] > 0))
        .groupby("stage_cluster")[[IS_DE, "up", "replicated"]]
        .sum()
        .to_string(),
    )

# %% Optimal transport from c64 to larva
sub = adata.copy()
sub.obs["time"] = pd.Categorical(sub.obs[STAGE_KEY].map(t_of).astype(float))
sub.obs["node"] = pd.Categorical(np.where(cluster != "", cluster, "none"))
tp = (
    TemporalProblem(sub)
    .prepare(time_key="time", joint_attr=LATENT_KEY)
    .solve(epsilon=EPSILON, tau_a=TAU_A, batch_size=OT_BATCH)
)
clusters_at = {
    st: sorted({c for c in cluster[obs_stage == st] if c}, key=num) for st in stages
}

edges = []
for a, b in zip(stages[:-1], stages[1:]):
    m = tp.cell_transition(
        source=t_of[a],
        target=t_of[b],
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
            }
        )
        .rename_axis(["from", "to"])
        .reset_index(),
    )
edges = pd.concat(edges, ignore_index=True)
edges.to_csv(f"{OUT}_transport_edges.csv", index=False)
strong = edges[(edges["fwd"] >= MIN_EDGE) | (edges["bwd"] >= MIN_EDGE)]

nodes = pd.DataFrame(
    {
        "stage_cluster": cluster[cluster != ""],
        "stage": obs_stage[cluster != ""],
        "source": obs_src[cluster != ""],
    }
)
nodes = (
    pd.crosstab(nodes["stage_cluster"], nodes["source"])
    .add_prefix("n_")
    .join(
        nodes.groupby("stage_cluster")["stage"].first(),
    )
)
nodes["n_cells"] = nodes.filter(like="n_").sum(axis=1)
node_tissue = pd.Series(
    labels_of(NODE_TISSUE_KEY)[cluster != ""], index=cluster[cluster != ""]
)
nodes["tissue"] = (
    node_tissue[node_tissue != ""]
    .groupby(level=0)
    .agg(lambda t: t.value_counts().index[0])
)
nodes["tissue_frac"] = (
    node_tissue[node_tissue != ""]
    .groupby(level=0)
    .agg(lambda t: t.value_counts(normalize=True).iloc[0])
)
nodes["tissue"] = nodes["tissue"].fillna("")
G = nx.DiGraph()
for n, row in nodes.iterrows():
    G.add_node(
        n,
        **{
            k: (
                v.item()
                if hasattr(
                    v,
                    "item",
                )
                else v
            )
            for k, v in row.items()
        },
    )
for rec in strong.to_dict("records"):
    G.add_edge(rec.pop("from"), rec.pop("to"), **rec)
nx.write_graphml(G, f"{OUT}_transport_graph.graphml")


def share_list(e, key, share):
    e = e[e[share] >= MIN_EDGE].sort_values(share, ascending=False)
    other = "to" if key == "from" else "from"
    return e.groupby(key)[[other, share]].apply(
        lambda g: ",".join(f"{n}:{v:.2f}" for n, v in zip(g[other], g[share]))
    )


print(
    f"transport graph: {G.number_of_nodes()} clusters, {G.number_of_edges()} edges with a share of {MIN_EDGE}+"
)

# %% Transport strength between clusters of adjacent stages
for a, b in zip(stages[:-1], stages[1:]):
    mat = (
        edges[edges["from"].isin(clusters_at[a]) & edges["to"].isin(clusters_at[b])]
        .pivot(
            index="from",
            columns="to",
            values="fwd",
        )
        .reindex(index=clusters_at[a], columns=clusters_at[b])
        .fillna(0)
    )
    mat = mat[
        sorted(
            mat.columns,
            key=lambda t: (mat.index.get_loc(mat[t].idxmax()), -mat[t].max()),
        )
    ]
    fig, ax = plt.subplots(
        figsize=(0.18 * len(mat.columns) + 2.5, 0.18 * len(mat.index) + 1.5)
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
    ax.set_title(
        f"Transport {a} to {b}",
        loc="left",
        fontsize=9,
    )
    fig.colorbar(
        im,
        ax=ax,
        shrink=0.6,
        label="share of the row cluster",
    )
    fig.savefig(
        f"{OUT}_transport_{a}_{b}.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

# %% Transport network with clusters grouped by stage
shown = strong[strong["fwd"] >= PLOT_MIN_EDGE]
tissue_rank = {t: i for i, t in enumerate(TISSUE_COLORS)}
order = {
    st: sorted(
        clusters_at[st],
        key=lambda n: (
            tissue_rank.get(nodes.loc[n, "tissue"], len(tissue_rank)),
            num(n),
        ),
    )
    for st in stages
}
y = {n: (i + 0.5) / len(ns) for ns in order.values() for i, n in enumerate(ns)}
for sweep in range(8):
    side, other = ("to", "from") if sweep % 2 == 0 else ("from", "to")
    for st in stages if sweep % 2 == 0 else stages[::-1]:
        bary = {}
        for n in order[st]:
            e = shown[shown[side] == n]
            if len(e):
                bary[n] = np.average(
                    e[other].map(y).to_numpy(float),
                    weights=e["mass"].to_numpy(float),
                )
            else:
                bary[n] = y[n]
        order[st] = sorted(
            order[st],
            key=lambda n: (
                tissue_rank.get(nodes.loc[n, "tissue"], len(tissue_rank)),
                bary[n],
            ),
        )
        for i, n in enumerate(order[st]):
            y[n] = (i + 0.5) / len(order[st])
height = max(len(v) for v in order.values())
pos = {
    n: (
        stages.index(nodes.loc[n, "stage"]),
        y[n] * height,
    )
    for n in nodes.index
}

fig, ax = plt.subplots(figsize=(1.5 * len(stages) + 2.5, 0.32 * height + 1.2))
for r in shown.to_dict("records"):
    (x0, y0), (x1, y1) = pos[r["from"]], pos[r["to"]]
    ax.plot(
        [x0, x1],
        [y0, y1],
        color="#52514e",
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
present = [t for t in TISSUE_COLORS if t in set(nodes["tissue"])]
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
    for t in present
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
        )
    )
handles += [
    plt.Line2D(
        [],
        [],
        color="#52514e",
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
fig.savefig(f"{OUT}_transport_network.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# %% Anchor mid-gastrula clusters to the ANISEED stage-12 territories
kh2ky = pd.read_csv(GENE_MAP, sep="\t").set_index("KH2012")["KY2021"]
mk = pd.read_csv(MARKERS, sep="\t")
mk = mk[mk["Stage"].isin(MARKER_STAGES)]
mk["gene"] = mk["Gene"].str.extract(r"KH2012:(\S+)")[0].map(kh2ky)
mk["territory"] = mk["Territory_eq"].str.rstrip("*")
sets = mk.groupby("territory")["gene"].apply(frozenset)
merged = sets.groupby(sets).transform(lambda t: "/".join(sorted(t.index)))
net = (
    pd.DataFrame({"source": merged.reindex(sets.index), "target": sets})
    .explode("target")
    .drop_duplicates()
    .assign(weight=1.0)
)
expr = adata.raw.to_adata()
panel = sorted(set(net["target"]) & set(expr.var_names))
print(
    "territories with identical markers:",
    sorted(set(merged) - set(sets.index)),
)
print(f"{len(panel)} of {net['target'].nunique()} marker genes in the data")

keep = np.flatnonzero((obs_stage == ANCHOR_STAGE) & (cluster != ""))
genes = sorted(set(adata.var_names) | set(panel))
x = expr[adata.obs_names[keep], genes].X
x = pd.DataFrame(
    x.toarray() if hasattr(x, "toarray") else np.asarray(x),
    columns=genes,
)
for src in set(obs_src[keep]):
    m = obs_src[keep] == src
    x.loc[m] = (x.loc[m] - x.loc[m].mean()) / x.loc[m].std().replace(0, np.nan)
x = x.fillna(0)
pb = x.groupby(cluster[keep]).mean()
pb = pb.loc[:, pb.std() > 0]
es, pv = dc.mt.ulm(data=(pb - pb.mean()) / pb.std(), net=net, tmin=TMIN)
es.index.name = "stage_cluster"
es.to_csv(f"{OUT}_{ANCHOR_STAGE}_territory_scores.csv")

top2 = np.argsort(-es.to_numpy(), axis=1)[:, :2]
rows = np.arange(len(es))
calls = pd.DataFrame(
    {
        "territory": es.columns[top2[:, 0]],
        "score": es.to_numpy()[rows, top2[:, 0]],
        "padj": pv.to_numpy()[rows, top2[:, 0]],
        "second": es.columns[top2[:, 1]],
        "margin": es.to_numpy()[rows, top2[:, 0]] - es.to_numpy()[rows, top2[:, 1]],
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
neural_clusters = list(calls.index[calls["neural_plate"]])
calls.to_csv(f"{OUT}_{ANCHOR_STAGE}_territory_calls.csv")
print(
    f"{len(neural_clusters)} of {len(calls)} {ANCHOR_STAGE} clusters match a neural plate territory"
)

# %% Best cluster for each blastomere on the neural plate map
neural_plate = gpd.read_file(ANCHOR_MAP)
neural_plate["name"] = neural_plate["name"].str.rstrip("*")
group_of = {b: t for t in es.columns for b in t.split("/")}
blastomeres = sorted(
    set(NP_GRID) | set(neural_plate["name"]),
    key=lambda b: (b[0], int(b.split(".")[1])),
)

full = (
    pd.concat(
        {
            "score": es,
            "padj": pv.reindex(
                index=es.index,
                columns=es.columns,
            ),
        },
        axis=1,
    )
    .stack(level=1, future_stack=True)
    .rename_axis(["stage_cluster", "territory"])
    .reset_index()
)
full = pd.DataFrame(
    {"blastomere": list(group_of), "territory": list(group_of.values())}
).merge(full, on="territory")
full["rank"] = (
    full.groupby("blastomere")["score"]
    .rank(ascending=False, method="first")
    .astype(int)
)
full = full.sort_values(["blastomere", "rank"])
full.to_csv(f"{OUT}_blastomere_cluster_scores.csv", index=False)

first = full[full["rank"] == 1].set_index("blastomere")
second = full[full["rank"] == 2].set_index("blastomere")
best = pd.DataFrame(index=pd.Index(blastomeres, name="blastomere"))
best["row"] = [NP_GRID.get(b, (None, None))[0] for b in best.index]
best["column"] = pd.array(
    [NP_GRID.get(b, (None, None))[1] for b in best.index], dtype="Int64"
)
best["territory"] = first["territory"]
best["stage_cluster"] = first["stage_cluster"]
best["n_cells"] = best["stage_cluster"].map(nodes["n_cells"]).astype("Int64")
best["score"] = first["score"]
best["padj"] = first["padj"]
best["second_cluster"] = second["stage_cluster"]
best["second_score"] = second["score"]
best["margin"] = best["score"] - best["second_score"]
best.to_csv(f"{OUT}_blastomere_best_cluster.csv")
print(best.round(3).to_string())

neural_plate = neural_plate.join(best[["stage_cluster", "score"]], on="name")
fig, ax = plt.subplots(figsize=(7, 4.8))
neural_plate.plot(
    column="score",
    cmap="Blues",
    vmin=0,
    ax=ax,
    edgecolor="white",
    linewidth=1.0,
    legend=True,
    legend_kwds={"label": "ULM score (t) of the best cluster", "shrink": 0.6},
    missing_kwds={"color": "#f0efec", "edgecolor": "white"},
)
for pt, name, cl, v in zip(
    neural_plate.representative_point(),
    neural_plate["name"],
    neural_plate["stage_cluster"],
    neural_plate["score"],
):
    ink = "white" if v > 0.55 * neural_plate["score"].max() else "black"
    ax.annotate(
        name,
        (pt.x, pt.y),
        xytext=(0, 7),
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
ax.set_title(
    f"Best {ANCHOR_STAGE} cluster for each blastomere",
    loc="left",
    fontsize=9,
)
fig.savefig(
    f"{OUT}_{ANCHOR_STAGE}_blastomere_map.png",
    dpi=DPI,
    bbox_inches="tight",
)
plt.close(fig)

# %% Benchmark at midG against Winkley's neural plate grid labels
wl = pd.DataFrame(
    {"label": labels_of(WINKLEY_LABEL_KEY), "stage_cluster": cluster},
    index=adata.obs_names,
)
wl = wl[(obs_src == WINKLEY) & (obs_stage == ANCHOR_STAGE)]
wl["truth"] = wl["label"].map(winkley_blastomeres)
wl = wl[wl["truth"].map(len) > 0]
wl["pick"] = (
    wl["stage_cluster"]
    .map(cluster_pick)
    .map(lambda b: b if isinstance(b, set) else set())
)
wl = wl.join(
    pd.DataFrame(
        [
            set_scores(p, t)
            for p, t in zip(
                wl["pick"],
                wl["truth"],
            )
        ],
        index=wl.index,
    )
)
metrics = [
    "precision",
    "recall",
    "jaccard",
    "bits",
    "score",
    "gain",
]

by_label = wl.groupby("label").agg(
    n_cells=("score", "size"),
    truth=("truth", lambda t: "/".join(sorted(t.iloc[0]))),
    top_pick=(
        "pick",
        lambda p: p.map(
            lambda b: "/".join(sorted(b)),
        ).mode()[0],
    ),
    **{m: (m, "mean") for m in metrics},
)
by_label.to_csv(f"{OUT}_{ANCHOR_STAGE}_winkley_labels.csv")
print(by_label.round(3).to_string())

by_blastomere = pd.DataFrame(
    [
        {
            "blastomere": b,
            "winkley_labelled": int(wl["truth"].map(lambda t: b in t).sum()),
            "picked": int(wl["pick"].map(lambda p: b in p).sum()),
            **{
                m: wl.loc[
                    wl["truth"].map(lambda t: b in t),
                    m,
                ].mean()
                for m in metrics
            },
            "best_cluster": best.loc[b, "stage_cluster"],
        }
        for b in best.index
    ]
).set_index("blastomere")
by_blastomere.to_csv(f"{OUT}_{ANCHOR_STAGE}_winkley_blastomeres.csv")
print(by_blastomere.round(3).to_string())
print(
    f"Winkley grid-labelled cells: {len(wl)}; mean "
    + ", ".join(f"{m} {wl[m].mean():.3f}" for m in metrics)
)
midg_summary = wl[metrics].mean().to_frame().T.assign(n_cells=len(wl))
midg_summary.to_csv(f"{OUT}_{ANCHOR_STAGE}_winkley_summary.csv", index=False)


# %% Predicted blastomere identities of midG descendants at earN and latN
def fmt(weights, keep=None):
    return ",".join(
        f"{k}:{v:.2f}"
        for k, v in sorted(weights.items(), key=lambda kv: -kv[1])
        if (keep is None and v >= MIN_SHOW) or (keep is not None and k in keep)
    )


anchor_gen = IDENTITY_GEN[ANCHOR_STAGE]
identity = {
    c: {
        b: 1
        / len(
            bs,
        )
        for b in bs
    }
    for c, bs in cluster_pick.items()
    if bs
}
nodes["identity"] = pd.Series(
    {
        c: "/".join(
            sorted(w),
        )
        for c, w in identity.items()
    }
)
id_rows = []
for st in [s for s in stages[stages.index(ANCHOR_STAGE) + 1 :] if s in IDENTITY_GEN]:
    gen = IDENTITY_GEN[st]
    nxt = {}
    for rec in edges[
        edges["from"].isin(list(identity)) & edges["to"].isin(clusters_at[st])
    ].to_dict("records"):
        for b, w in identity[rec["from"]].items():
            kids = to_generation(b, gen)
            for d in kids:
                nxt.setdefault(rec["to"], {})
                nxt[rec["to"]][d] = nxt[rec["to"]].get(
                    d,
                    0.0,
                ) + rec[
                    "bwd"
                ] * w / len(kids)
    identity = nxt
    for n in clusters_at[st]:
        w = identity.get(n, {})
        anc = {}
        for d, v in w.items():
            (a,) = to_generation(d, anchor_gen)
            anc[a] = anc.get(a, 0.0) + v
        kept = {a for a, v in anc.items() if v >= MIN_SHOW}
        top = max(anc, key=anc.get) if anc else ""
        predicted = "/".join(sorted(to_generation(top, gen))) if top else ""
        id_rows.append(
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
                "identities": fmt(
                    w,
                    keep={
                        d
                        for d in w
                        if to_generation(
                            d,
                            anchor_gen,
                        )
                        & kept
                    },
                ),
            }
        )
        nodes.loc[n, "identity"] = predicted
identities = pd.DataFrame(id_rows).set_index("stage_cluster")
identities.to_csv(f"{OUT}_predicted_identities.csv")
print(identities.drop(columns="identities").round(3).to_string())

nodes["identity"] = nodes["identity"].fillna("")
adjacency = nodes[["stage", "n_cells", "tissue", "identity"]].copy()
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
adjacency = (
    adjacency.assign(order=adjacency.index.map(num))
    .sort_values(["stage", "order"])
    .drop(columns="order")
)
adjacency.to_csv(f"{OUT}_adjacency.tsv", sep="\t")

# %% Backward benchmark: Winkley labels among the ancestors of midG clusters
lab = labels_of(WINKLEY_LABEL_KEY)
backward = []
backward_shares = []
for st, gen in BENCH_GEN.items():
    at = (obs_src == WINKLEY) & (obs_stage == st) & (lab != "")
    if not at.any():
        print(f"{st}: no labelled Winkley cells, skipped")
        continue
    groups = sorted(set(lab[at]))
    key = f"winkley_{st}"
    sub.obs[key] = pd.Categorical(np.where(at, lab, "none"))
    back = tp.cell_transition(
        source=t_of[st],
        target=t_of[ANCHOR_STAGE],
        source_groups={key: groups},
        target_groups={"node": neural_clusters},
        forward=False,
        batch_size=OT_BATCH,
        key_added=None,
    )
    background = (
        pd.Series(lab[at])
        .value_counts(
            normalize=True,
        )
        .reindex(groups)
    )
    named = {
        l: to_generation(
            WINKLEY_EARLY.get(l, l),
            gen,
        )
        for l in groups
    }
    for c in neural_clusters:
        share = back[c].reindex(groups).fillna(0)
        fold = share / background
        truth = set().union(*(to_generation(b, gen) for b in cluster_pick[c]))
        expected = [l for l in groups if named[l] & truth]
        top = share.idxmax()
        backward_shares += [
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
            }
        )
backward = pd.DataFrame(backward)
backward.to_csv(f"{OUT}_backward_benchmark.csv", index=False)
backward_shares = pd.DataFrame(backward_shares)
backward_shares.to_csv(f"{OUT}_backward_label_shares.csv", index=False)
if len(backward):
    print(backward.drop(columns="labels").round(3).to_string(index=False))
    backward_summary = backward.groupby("stage")[
        [
            "expected_share",
            "expected_background",
            "expected_fold",
            "top_is_expected",
        ]
    ].mean()
    backward_summary.to_csv(f"{OUT}_backward_summary.csv")
    print(backward_summary.round(3).to_string())

# %% Forward benchmark: Cao tissue labels among the descendants of midG clusters
tissue = labels_of(CAO_TISSUE_KEY)
midg_clusters = clusters_at[ANCHOR_STAGE]
forward = []
forward_shares = []
for st in stages[stages.index(ANCHOR_STAGE) + 1 :]:
    at = (obs_src == CAO) & (obs_stage == st) & (tissue != "")
    if not at.any():
        continue
    groups = sorted(set(tissue[at]))
    key = f"cao_tissue_{st}"
    sub.obs[key] = pd.Categorical(np.where(at, tissue, "none"))
    fwd = tp.cell_transition(
        source=t_of[ANCHOR_STAGE],
        target=t_of[st],
        source_groups={"node": midg_clusters},
        target_groups={key: groups},
        forward=True,
        batch_size=OT_BATCH,
        key_added=None,
    )
    background = (
        pd.Series(tissue[at])
        .value_counts(
            normalize=True,
        )
        .reindex(groups)
    )
    for c in midg_clusters:
        share = fwd.loc[c].reindex(groups).fillna(0)
        top = share.idxmax()
        forward_shares += [
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
                "blastomeres": "/".join(sorted(cluster_pick.get(c, set()))),
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
            }
        )
forward = pd.DataFrame(forward)
forward.to_csv(f"{OUT}_forward_benchmark.csv", index=False)
forward_shares = pd.DataFrame(forward_shares)
forward_shares.to_csv(f"{OUT}_forward_tissue_shares.csv", index=False)
if len(forward):
    forward["stage"] = pd.Categorical(
        forward["stage"],
        categories=stages,
        ordered=True,
    )
    forward_summary = forward.groupby(
        [
            "stage",
            "neural_plate",
        ],
        observed=True,
    )[
        [
            "nervous_share",
            "nervous_background",
            "nervous_fold",
        ]
    ].mean()
    forward_summary.to_csv(f"{OUT}_forward_summary.csv")
    print(forward_summary.round(3).to_string())

# %% Benchmark graphics
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


def style(ax, grid_axis="y"):
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


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(f"{OUT}_{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if len(by_label):
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
    ax.set_yticks(ypos, [f"{l}  (n={n})" for l, n in zip(bl.index, bl["n_cells"])])
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
    save(fig, f"{ANCHOR_STAGE}_winkley_labels")

    bench_map = neural_plate.join(
        by_blastomere[["score"]].rename(columns={"score": "bench_score"}),
        on="name",
    )
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bench_map.plot(
        column="bench_score",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        edgecolor="white",
        linewidth=1.0,
        legend=True,
        legend_kwds={
            "label": "mean score against Winkley's labels",
            "shrink": 0.6,
        },
        missing_kwds={
            "color": "#f0efec",
            "edgecolor": "white",
        },
    )
    for pt, name, v in zip(
        bench_map.representative_point(),
        bench_map["name"],
        bench_map["bench_score"],
    ):
        ax.annotate(
            name,
            (pt.x, pt.y),
            ha="center",
            va="center",
            fontsize=5.5,
            color="white" if v > 0.55 else INK,
        )
    ax.set_axis_off()
    ax.set_title(
        f"{ANCHOR_STAGE} benchmark by blastomere",
        loc="left",
        fontsize=9,
        color=INK,
    )
    save(fig, f"{ANCHOR_STAGE}_winkley_map")

for st, g in backward_shares.groupby("stage") if len(backward_shares) else []:
    fold = g.pivot(
        index="stage_cluster",
        columns="label",
        values="fold",
    )
    expected = g.pivot(
        index="stage_cluster",
        columns="label",
        values="expected",
    )
    rows = sorted(fold.index, key=num)
    fold, expected = fold.loc[rows], expected.loc[rows].astype(bool)
    value = np.log2(fold.clip(lower=1 / 8, upper=8))
    fig, ax = plt.subplots(
        figsize=(0.45 * fold.shape[1] + 3, 0.28 * fold.shape[0] + 1.5)
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
            )
        )
    ax.set_yticks(
        range(len(rows)),
        [f"{num(c)}  {calls.loc[c, 'blastomeres']}" for c in rows],
    )
    ax.set_xticks(range(fold.shape[1]), fold.columns, rotation=45, ha="right")
    ax.set_ylabel(f"{ANCHOR_STAGE} neural plate cluster")
    ax.set_xlabel(f"Winkley label at {st}")
    style(ax, grid_axis=None)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("log2 enrichment over background", color=MUTED)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_visible(False)
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=1.2)],
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
    save(fig, f"backward_{st}_enrichment")

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
    ax.set_xlabel(f"share of Winkley cells at {st} with the expected labels")
    ax.set_ylabel("share of the cluster's ancestry on those labels")
    style(ax, grid_axis="both")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title(
        f"Expected ancestry at {st}, per {ANCHOR_STAGE} neural plate cluster",
        loc="left",
        fontsize=9,
        color=INK,
    )
    save(fig, f"backward_{st}_expected")

if len(forward):
    fw_stages = [s for s in stages if s in set(forward["stage"].astype(str))]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(0.9 * len(fw_stages) + 2, 3.8))
    for k, (flag, color, text) in enumerate(
        (
            (True, SERIES[0], f"{ANCHOR_STAGE} neural plate clusters"),
            (False, SERIES[1], f"other {ANCHOR_STAGE} clusters"),
        )
    ):
        for i, st in enumerate(fw_stages):
            d = forward[
                (forward["stage"].astype(str) == st) & (forward["neural_plate"] == flag)
            ]
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
    save(fig, "forward_nervous_share")

    comp = (
        forward_shares[forward_shares["neural_plate"]]
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
    ax.set_title(
        f"Cao tissue of the descendants of {ANCHOR_STAGE} neural plate clusters",
        loc="left",
        fontsize=9,
        color=INK,
    )
    save(fig, "forward_tissue_composition")

# %% Save
adata.obs["stage_cluster"] = pd.Categorical(adata.obs["stage_cluster"])
adata.obs["np_blastomeres"] = pd.Categorical(
    adata.obs["stage_cluster"].map(calls["blastomeres"]).astype(object)
)
adata.write_h5ad(f"{OUT}_lineage.h5ad")
