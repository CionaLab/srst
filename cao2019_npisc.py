# %%
import re

import pandas as pd
import scanpy as sc
import numpy as np
import networkx as nx
import seaborn as sns

import geopandas as gpd
import matplotlib.pyplot as plt

from npisc.build_matrix import (
    split_adata,
    pad_compatible,
    append_raw,
    adjacent,
    map_cells,
    plot_np,
    pca_raw,
    round_trip_distance,
)

from npisc.analyze_expression import diff_expression

# %%
PATTERN_STAGES = r"(early|mid|late) (gastrula|neurula)"
PATTERN_CELLS = r"[Aa]\d+\.\d+$"

STAGES_IN_SITU = [
    ("mid gastrula", "mid_gastrula.geojson", "midG"),
    ("late gastrula", "late_gastrula.geojson", "latG"),
    ("early neurula", "early_neurula.geojson", "earN"),
    ("mid neurula", "mid_neurula.geojson", "midN"),
    ("late neurula", "late_neurula.geojson", "latN"),
]

# %%
df = pd.read_csv("npisc/pass_02.tsv", sep="\t")
df_map = pd.read_csv("npisc/kh2012_ky2021_map.tsv", sep="\t")
df_map["query"] = "KH2012:" + df_map["query"]
df["Gene"] = df["Gene"].apply(lambda x: x.split(" ")[0])
df = pd.merge(df, df_map, how="left", left_on="Gene", right_on="query")
df = df.drop(columns=["Gene", "query"])
df = df.rename(columns={"subject": "Gene"})

df["Stage"] = df["Stage"].apply(lambda x: (" ".join(re.findall(PATTERN_STAGES, x)[0])))
df = df[df["Territory_eq"].str.contains(PATTERN_CELLS)]
df = df[["Stage", "Gene", "Territory_eq"]].drop_duplicates()

dfs = dict(tuple(df.groupby("Stage")))

df_counts = {
    stage: df.groupby("Territory_eq")["Gene"]
    .nunique()
    .reset_index()
    .rename(columns={"Gene": "n"})
    for stage, df in dfs.items()
}

d_patterns = {
    stage: pd.pivot_table(
        df,
        values="Gene",
        index="Territory_eq",
        columns="Gene",
        aggfunc="size",
        fill_value=0,
    ).astype(bool)
    for stage, df in dfs.items()
}

# %%
adata = sc.read_h5ad("cao2019_ky21.h5ad")
adata_raw = sc.read_h5ad("cao2019_ky21_raw.h5ad")

# %%
adatas = split_adata(adata, "stage")

# %%
for key in adatas.keys():
    adata = adatas[key]

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
    sc.tl.leiden(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga")

    adata.write_h5ad(f"cao2019_npisc_ky21_{key}.h5ad")

# %%
STAGES_SC = ["midG", "earN", "latN", "iniT", "earT", "midT", "latTI", "latTII", "larva"]

adatas = {s: sc.read_h5ad(f"cao2019_npisc_ky21_{s}.h5ad") for s in STAGES_SC}

adatas_raw = {
    k: append_raw(v, adata_raw[v.obs.index, v.var.index]) for k, v in adatas.items()
}

# %%
gdfs = {k: gpd.read_file(f"npisc/{file}") for _, file, k in STAGES_IN_SITU}

for stage in gdfs:
    gdfs[stage]["name"] = gdfs[stage]["name"].str.replace("*", "", regex=False)

# %%
STAGES_MAPPING = [
    ("mid gastrula", "midG"),
    ("early neurula", "earN"),
    ("late neurula", "latN"),
]

for k1, k2 in STAGES_MAPPING:
    t1, t2, t3 = map_cells(adatas_raw[k2], d_patterns[k1])

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    sc.pl.umap(adatas[k2], color=["leiden"], legend_loc="on data", ax=ax)
    fig.tight_layout()
    fig.savefig(f"cao2019_npisc_ky21_{k2}_umap.png")

    t3.to_csv(f"cao2019_npisc_ky21_{k2}_cos_theta.csv")

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax = plot_np(
        t3,
        gdfs[k2],
        ax,
        {
            "column": "cos_theta",
            "linewidth": 0.8,
            "categorical": False,
            "vmin": 0.1,
            "vmax": 0.7,
            "missing_kwds": {"color": "lightgrey"},
            "cmap": sns.color_palette("rocket", as_cmap=True),
        },
        {
            "color": "white",
            "fontsize": 6,
        },
    )
    ax.set_title(k1)
    fig.tight_layout()
    patch_col = ax.collections[0]
    fig.colorbar(patch_col, ax=ax, shrink=0.5)
    fig.savefig(f"cao2019_npisc_ky21_{k2}_np.png")

# %%
MARKERS = [
    ("Chr1.422", "midG", "ebf"),
    ("Chr4.720", "midG", "otx"),
    ("Chr7.1003", "midG", "osbp2"),
]

fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)
sc.pl.stacked_violin(
    adatas["midG"],
    [f"KY21:KY21.{g}" for g, _, _ in MARKERS],
    groupby="leiden",
    swap_axes=True,
    ax=ax,
)

fig.tight_layout()
fig.savefig("cao2019_npisc_midG_markers.png")

# %%

# Generate the BLAST map from the following:
# \time blastp -db swissprot -taxids 9606 -query ky2021p.fasta -parse_deflines \
# -outfmt "7 qacc sacc pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovs" \
# -out ky2021_swissprot.txt -num_threads 64 -mt_mode 1

df_ky_sp = pd.read_csv(
    "ky2021_swissprot.txt",
    sep="\t",
    names=[
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "coverage",
    ],
    comment="#",
)
df_ky_sp["qseqid"] = df_ky_sp["qseqid"].apply(lambda x: re.sub(r"\.v.+", "", x))

df_ky_sp = df_ky_sp.loc[df_ky_sp.groupby("qseqid")["evalue"].idxmin()]

(
    pd.merge(
        df_ky_sp, pd.read_csv("uniprot_data.csv"), left_on="sseqid", right_on="uniprot"
    )
    .groupby("qseqid")
    .apply(lambda x: x.nsmallest(1, "evalue"))
    .reset_index(drop=True)[["qseqid", "fullname"]]
).to_csv("ky2021_swissprot_map.csv", index=False)

# %%

df_ky_sp = pd.read_csv("ky2021_swissprot_map.csv")
NUM_TOP = 50

# %%

for k in STAGES_SC:
    a = sc.read_h5ad(f"cao2019_npisc_ky21_{k}.h5ad")
    diff_expression(a, top=NUM_TOP).merge(
        df_ky_sp, left_on="gene", right_on="qseqid", how="left"
    ).to_csv(f"cao2019_npisc_ky21_{k}_top{NUM_TOP}.csv", index=False)

# %%

for i, (k1, k2) in enumerate(adjacent(STAGES_SC)):
    print(k1, k2)

    a1, a2 = pad_compatible(adatas_raw[k1], adatas_raw[k2])

    pca_raw(a1).write_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k1}.h5ad")
    pca_raw(a2).write_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k2}.h5ad")

# %%


for i, (k1, k2) in enumerate(adjacent(STAGES_SC)):
    print(k1, k2)

    a1 = sc.read_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k1}.h5ad")
    a2 = sc.read_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k2}.h5ad")

    round_trip_distance(a1, a2, k1, k2).to_csv(
        f"cao2019_npisc_ky21_stage_stage_{k1}_{k2}.csv"
    )

# %%

G = nx.DiGraph()

for k1, k2 in adjacent(STAGES_SC):
    d5 = pd.read_csv(f"cao2019_npisc_ky21_stage_stage_{k1}_{k2}.csv", index_col=0)
    edges = d5.stack().reset_index()
    edges.columns = ["source", "target", "weight"]
    edges["source"] = edges["source"].apply(lambda x: f"{k1}_{x}")
    edges["target"] = edges["target"].apply(lambda x: f"{k2}_{x}")
    G.add_weighted_edges_from(edges.values)

# %%

d_leidens = {
    k: v.obs["leiden"].cat.categories.map(lambda x: f"{k}_{x}").to_list()
    for k, v in adatas.items()
}

# %%

df_shortest_paths = (
    pd.DataFrame(
        [
            {
                "source": s,
                "target": t,
                "distance": nx.shortest_path_length(
                    G, source=s, target=t, weight="weight"
                ),
            }
            for s in d_leidens["midG"]
            for t in d_leidens["latN"]
        ]
    )
    .groupby("source")
    .apply(lambda x: x.nsmallest(1, "distance"))
)

d_shortest_paths = {
    (s, t): nx.shortest_path(G, source=s, target=t, weight="weight")
    for s in d_leidens["midG"]
    for t in d_leidens["latN"]
}

# %%

STAGES_SUBCLUSTERS = [
    ("mid gastrula", "midG", ("5", "7", "10", "20")),
    ("early neurula", "earN", ("14", "16", "18", "21")),
    ("late neurula", "latN", ("8", "17", "18", "25", "29")),
]

# %%

for _, k2, sbc in STAGES_SUBCLUSTERS:
    adata = adatas[k2][adatas[k2].obs["leiden"].isin(sbc)]

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.leiden(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga")

    adata.write_h5ad(f"cao2019_npisc_ky21_np_{k2}.h5ad")

# %%
adatas_sbc = {
    s: sc.read_h5ad(f"cao2019_npisc_ky21_np_{s}.h5ad") for _, s, _ in STAGES_SUBCLUSTERS
}

adatas_sbc_raw = {
    k: append_raw(v, adata_raw[v.obs.index, v.var.index]) for k, v in adatas_sbc.items()
}

# %%

for k1, k2, _ in STAGES_SUBCLUSTERS:
    t1, t2, t3 = map_cells(adatas_sbc_raw[k2], d_patterns[k1])

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    sc.pl.umap(adatas_sbc[k2], color=["leiden"], legend_loc="on data", ax=ax)
    fig.tight_layout()
    fig.savefig(f"cao2019_npisc_ky21_np_{k2}_umap.png")

    t3.to_csv(f"cao2019_npisc_ky21_np_{k2}_cos_theta.csv")

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax = plot_np(
        t3,
        gdfs[k2],
        ax,
        {
            "column": "cos_theta",
            "linewidth": 0.8,
            "categorical": False,
            "vmin": 0.1,
            "vmax": 0.7,
            "missing_kwds": {"color": "lightgrey"},
            "cmap": sns.color_palette("rocket", as_cmap=True),
        },
        {
            "color": "white",
            "fontsize": 6,
        },
    )
    ax.set_title(k1)
    fig.tight_layout()
    patch_col = ax.collections[0]
    fig.colorbar(patch_col, ax=ax, shrink=0.5)
    fig.savefig(f"cao2019_npisc_ky21_np_{k2}_np.png")

# %%

for _, k, _ in STAGES_SUBCLUSTERS:
    a = sc.read_h5ad(f"cao2019_npisc_ky21_np_{k}.h5ad")
    diff_expression(a, top=NUM_TOP).merge(
        df_ky_sp, left_on="gene", right_on="qseqid", how="left"
    ).to_csv(f"cao2019_npisc_ky21_np_{k}_top{NUM_TOP}.csv", index=False)

# %%

for i, ((_, k1, _), (_, k2, _)) in enumerate(adjacent(STAGES_SUBCLUSTERS)):
    print(k1, k2)

    a1, a2 = pad_compatible(adatas_sbc_raw[k1], adatas_sbc_raw[k2])

    pca_raw(a1).write_h5ad(f"cao2019_npisc_ky21_np_stage_stage_{i}_{k1}.h5ad")
    pca_raw(a2).write_h5ad(f"cao2019_npisc_ky21_np_stage_stage_{i}_{k2}.h5ad")

# %%

for i, ((_, k1, _), (_, k2, _)) in enumerate(adjacent(STAGES_SUBCLUSTERS)):
    print(k1, k2)

    a1 = sc.read_h5ad(f"cao2019_npisc_ky21_np_stage_stage_{i}_{k1}.h5ad")
    a2 = sc.read_h5ad(f"cao2019_npisc_ky21_np_stage_stage_{i}_{k2}.h5ad")

    round_trip_distance(a1, a2, k1, k2).to_csv(
        f"cao2019_npisc_ky21_np_stage_stage_{k1}_{k2}.csv"
    )

# %%

G = nx.DiGraph()

for (_, k1, _), (_, k2, _) in adjacent(STAGES_SUBCLUSTERS):
    d5 = pd.read_csv(f"cao2019_npisc_ky21_np_stage_stage_{k1}_{k2}.csv", index_col=0)
    edges = d5.stack().reset_index()
    edges.columns = ["source", "target", "weight"]
    edges["source"] = edges["source"].apply(lambda x: f"{k1}_{x}")
    edges["target"] = edges["target"].apply(lambda x: f"{k2}_{x}")
    G.add_weighted_edges_from(edges.values)

# %%

d_leidens = {
    k: v.obs["leiden"].cat.categories.map(lambda x: f"{k}_{x}").to_list()
    for k, v in adatas_sbc.items()
}

# %%

df_shortest_paths = (
    pd.DataFrame(
        [
            {
                "source": s,
                "target": t,
                "distance": nx.shortest_path_length(
                    G, source=s, target=t, weight="weight"
                ),
            }
            for s in d_leidens["midG"]
            for t in d_leidens["latN"]
        ]
    )
    .groupby("source")
    .apply(lambda x: x.nsmallest(1, "distance"))
)

d_shortest_paths = {
    (s, t): nx.shortest_path(G, source=s, target=t, weight="weight")
    for s in d_leidens["midG"]
    for t in d_leidens["latN"]
}

# %%
