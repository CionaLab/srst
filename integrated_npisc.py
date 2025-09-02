# %%

import re

import pandas as pd
import scanpy as sc
import networkx as nx
import seaborn as sns
import scvi
import torch

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from npisc.build_matrix import (
    adjacent,
    split_adata,
    map_cells,
    plot_np,
    make_stage_name,
    make_digraph,
    CrossStage,
)

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63
scvi.settings.seed = 0

# %%

PREFIX = "integrated"
GENOME = "ky21"

PATTERN_STAGES = r"(early|mid|late) (gastrula|neurula)"
PATTERN_CELLS = r"[Aa]\d+\.\d+$"

STAGES_IN_SITU = [
    ("mid gastrula", "mid_gastrula.geojson", "midG"),
    ("late gastrula", "late_gastrula.geojson", "latG"),
    ("early neurula", "early_neurula.geojson", "earN"),
    ("mid neurula", "mid_neurula.geojson", "midN"),
    ("late neurula", "late_neurula.geojson", "latN"),
]

COUNTS_LAYER = "counts"
SIZE_FACTOR = "size_factor"
BATCH_KEY = "sample"
LINEAR_SCVI_BASIS = "LinearSCVI_basis"
LINEAR_SCVI_LATENT_KEY = "X_LinearSCVI"
SCVI_LATENT_KEY = "X_SCVI"
SCVI_EXPRESSION_KEY = "SCVI_normalized"
SCVI_LOG1P_KEY = "scVI_log1p"

# %%
df = pd.read_csv("npisc/pass_02.tsv", sep="\t")
df_map = pd.read_csv("npisc/kh2012_ky2021_map.tsv", sep="\t")
df_map["KH2012"] = "KH2012:" + df_map["KH2012"]
df["Gene"] = df["Gene"].apply(lambda x: x.split(" ")[0])
df = pd.merge(df, df_map, how="left", left_on="Gene", right_on="KH2012")
df = df.drop(columns=["Gene", "KH2012"])
df = df.rename(columns={"subject": "Gene"})

df["Stage"] = df["Stage"].apply(lambda x: (" ".join(re.findall(PATTERN_STAGES, x)[0])))
df = df[df["Territory_eq"].str.contains(PATTERN_CELLS)]
df = df[["Stage", "KY2021", "Territory_eq"]].drop_duplicates()

dfs = dict(tuple(df.groupby("Stage")))

df_counts = {
    k2: dfs[k1]
    .groupby("Territory_eq")["KY2021"]
    .nunique()
    .reset_index()
    .rename(columns={"KY2021": "n"})
    for k1, _, k2 in STAGES_IN_SITU
}

for k1, _, k2 in STAGES_IN_SITU:
    pd.pivot_table(
        dfs[k1],
        values="KY2021",
        index="Territory_eq",
        columns="KY2021",
        aggfunc="size",
        fill_value=0,
    ).astype(bool).to_csv(
        f"{PREFIX}_{GENOME}_npisc_{k2}_pattern.csv",
        index=True,
    )

# %%

d_patterns = {
    k: pd.read_csv(
        f"{PREFIX}_{GENOME}_npisc_{k}_pattern.csv",
        index_col=0,
    )
    for _, _, k in STAGES_IN_SITU
}

gdfs = {k: gpd.read_file(f"npisc/{file}") for _, file, k in STAGES_IN_SITU}

for k in gdfs:
    gdfs[k]["name"] = gdfs[k]["name"].str.replace(
        "*",
        "",
        regex=False,
    )

# %%

adata = sc.read_h5ad(f"{PREFIX}_{GENOME}.h5ad")
adatas = split_adata(adata, "stage")

for key in adatas.keys():
    a_tmp = adatas[key].copy()

    sc.pp.neighbors(
        a_tmp,
        n_neighbors=100,
        use_rep=SCVI_LATENT_KEY,
    )

    sc.tl.leiden(
        a_tmp,
        flavor="igraph",
        n_iterations=-1,
    )

    sc.pp.pca(
        a_tmp,
        layer=SCVI_EXPRESSION_KEY,
        svd_solver="arpack",
    )

    sc.tl.umap(
        a_tmp,
        min_dist=0.3,
    )

    a_tmp.write_h5ad(f"{PREFIX}_{GENOME}_npisc_{key}.h5ad")

# %%

STAGES_SC = [
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

adata = sc.read_h5ad(f"{PREFIX}_{GENOME}.h5ad")
adatas = {s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_{s}.h5ad") for s in STAGES_SC}

# %%

STAGES_MAPPING = [
    ("mid gastrula", "midG"),
    ("early neurula", "earN"),
    ("late neurula", "latN"),
]

# %%

for k1, k2 in STAGES_MAPPING:
    t1, t2, t3 = map_cells(
        adatas[k2],
        d_patterns[k2],
        basis=LINEAR_SCVI_BASIS,
        use_rep=LINEAR_SCVI_LATENT_KEY,
    )

    g = sns.clustermap(
        t2,
        xticklabels=1,
        yticklabels=1,
        figsize=(6.5, 6.5),
    )

    g.savefig(f"{PREFIX}_{GENOME}_npisc_{k2}_npmat.png", dpi=300)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    sc.pl.umap(
        adatas[k2],
        color=["leiden"],
        legend_loc="on data",
        legend_fontoutline=2,
        outline_color="white",
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_{GENOME}_npisc_{k2}_umap.png")

    t3.to_csv(f"{PREFIX}_{GENOME}_npisc_{k2}_cos_theta.csv")

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax = plot_np(
        t3,
        gdfs[k2],
        ax,
        {
            "column": "cos_theta",
            "linewidth": 0.8,
            "categorical": False,
            "missing_kwds": {"color": "lightgrey"},
            "cmap": sns.color_palette("rocket", as_cmap=True),
        },
        {
            "color": "black",
            "fontsize": 6,
            "path_effects": [
                pe.withStroke(linewidth=2, foreground="white"),
            ],
        },
    )
    ax.set_title(k1)
    fig.tight_layout()
    patch_col = ax.collections[0]
    fig.colorbar(patch_col, ax=ax, shrink=0.5)
    fig.savefig(f"{PREFIX}_{GENOME}_npisc_{k2}_npmap.png")
    t2.to_csv(f"{PREFIX}_{GENOME}_npisc_{k2}_npmap.csv")

# %%
a_tmp = adatas["midG"][
    :, adatas["midG"].var_names.intersection(d_patterns["midG"].T.index)
].copy()

# %%
scvi.external.CellAssign.setup_anndata(
    a_tmp,
    size_factor_key=SIZE_FACTOR,
    batch_key=BATCH_KEY,
)

# %%
a_model = scvi.external.CellAssign(a_tmp, d_patterns["midG"].T)

# %%
a_model.train(
    check_val_every_n_epoch=1,
    max_epochs=800,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="elbo_validation",
)

a_model.save(
    f"{PREFIX}_{GENOME}_cellassign_midG",
    overwrite=True,
    save_anndata=True,
)

# %%
a_model = scvi.external.CellAssign.load(
    f"{PREFIX}_{GENOME}_cellassign_midG",
)

a_tmp = a_model.adata

# %%
pred_cellassign = a_model.predict()

pred_npisc, _, _ = map_cells(
    a_tmp,
    d_patterns["midG"],
    basis=LINEAR_SCVI_BASIS,
    use_rep=LINEAR_SCVI_LATENT_KEY,
)

a_tmp.obs["pred_cellassign"] = pred_cellassign.idxmax(axis=1).values
a_tmp.obs["pred_npisc"] = pred_npisc.idxmax(axis=0).values

# %%

d_ground_truth = pd.read_csv(
    "npisc/ground_truth_map.tsv",
    sep="\t",
    index_col=0,
).to_dict()["cluster"]

a_tmp.obs["pred_cellassign"] = (
    a_tmp.obs["pred_cellassign"]
    .map(d_ground_truth)
    .fillna(a_tmp.obs["pred_cellassign"])
)
a_tmp.obs["pred_npisc"] = (
    a_tmp.obs["pred_npisc"].map(d_ground_truth).fillna(a_tmp.obs["pred_npisc"])
)

df_ground_truth = pd.read_csv(
    "npisc/winkley2021_meta.tsv",
    sep="\t",
)
df_ground_truth["Cell"] = df_ground_truth["Cell"].str.replace(
    "DMSO_MidG_",
    "",
    regex=False,
)
df_ground_truth = df_ground_truth[
    df_ground_truth["CellType"].str.startswith(
        "NP",
    )
]
df_ground_truth = df_ground_truth.set_index("Cell")

df_ground_truth = df_ground_truth[
    ~df_ground_truth["CellType"].str.startswith(
        "NP (b)",
    )
]

df_ground_truth = df_ground_truth.merge(
    a_tmp[a_tmp.obs["source"] == "winkley2021", :].obs,
    how="left",
    left_index=True,
    right_index=True,
)[["Cluster", "pred_cellassign", "pred_npisc"]]


df_ground_truth["agree_cellassign"] = (
    df_ground_truth["Cluster"] == df_ground_truth["pred_cellassign"]
)
df_ground_truth["agree_npisc"] = (
    df_ground_truth["Cluster"] == df_ground_truth["pred_npisc"]
)

df_ground_truth.dropna(inplace=True)

print(
    f"CellAssign: {df_ground_truth['agree_cellassign'].sum()}/{df_ground_truth['Cluster'].count()}"
)
print(
    f"npisc: {df_ground_truth['agree_npisc'].sum()}/{df_ground_truth['Cluster'].count()}"
)

# %%

dg = make_digraph(
    {k: adatas[k] for k in ["midG", "earN", "latN"]},
    ["midG", "earN", "latN"],
    use_rep=LINEAR_SCVI_LATENT_KEY,
)

nx.write_gml(dg, f"{PREFIX}_{GENOME}_npisc_cross_stage.gml")

# %%

d_leidens = {
    k: v.obs["leiden"].cat.categories.map(make_stage_name(k)).to_list()
    for k, v in adatas.items()
}

dg = nx.read_gml(f"{PREFIX}_{GENOME}_npisc_cross_stage.gml")

STATE_START = "midG"
STATE_END = "latN"

cs = CrossStage(
    dg,
    d_leidens[STATE_START],
    d_leidens[STATE_END],
)

# %%

df_edges = nx.to_pandas_edgelist(dg)

for (_, k1), (_, k2) in adjacent(STAGES_MAPPING):
    edges = df_edges[
        df_edges["source"].str.contains(k1) & df_edges["target"].str.contains(k2)
    ]

    pivot_edges = 1 - edges.pivot(
        index="source",
        columns="target",
        values="weight",
    ).fillna(0)

    g = sns.clustermap(
        pivot_edges,
        cmap="rocket",
        square=True,
        xticklabels=True,
        yticklabels=True,
        figsize=(6.5, 6.5),
    )

    g.savefig(f"{PREFIX}_{GENOME}_npisc_cross_stage_{k1}_{k2}.png", dpi=300)

# %%

l_cross_stage = (
    f"midG_{k}"
    for k in (
        "1",
        "7",
        "11",
    )
)

for (_, k1), (_, k2) in adjacent(STAGES_MAPPING):
    try:
        cs.update_internal(d_leidens[k1], d_leidens[k2])
        paths_cross_stage = [cs.find_paths(i) for i in l_cross_stage]
        l_cross_stage = (i for _, i in paths_cross_stage)
        for i in paths_cross_stage:
            print(i)
    except KeyError:
        pass
    except nx.NetworkXNoPath:
        pass

# %%

STAGES_SUBCLUSTERS = [
    (
        "mid gastrula",
        "midG",
        (
            "1",
            "7",
            "11",
        ),
    ),
    (
        "early neurula",
        "earN",
        (
            "3",
            "10",
            "14",
        ),
    ),
    (
        "late neurula",
        "latN",
        (
            "1",
            "28",
            "24",
        ),
    ),
]

# %%

adatas = {s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_{s}.h5ad") for s in STAGES_SC}

for _, k2, sbc in STAGES_SUBCLUSTERS:
    a_tmp = adatas[k2][adatas[k2].obs["leiden"].isin(sbc)].copy()

    sc.pp.neighbors(
        a_tmp,
        n_neighbors=50,
        use_rep=SCVI_LATENT_KEY,
    )

    sc.tl.leiden(
        a_tmp,
        resolution=2,
        flavor="igraph",
        n_iterations=-1,
    )

    sc.pp.pca(
        a_tmp,
        layer=SCVI_EXPRESSION_KEY,
        svd_solver="arpack",
    )

    sc.tl.umap(
        a_tmp,
        min_dist=0.3,
    )

    a_tmp.write_h5ad(f"{PREFIX}_{GENOME}_npisc_np_{k2}.h5ad")

# %%

adatas = {s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_{s}.h5ad") for s in STAGES_SC}
adatas_sbc = {
    s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_np_{s}.h5ad")
    for _, s, _ in STAGES_SUBCLUSTERS
}

# %%

for k1, k2, _ in STAGES_SUBCLUSTERS:
    t1, t2, t3 = map_cells(
        adatas_sbc[k2],
        d_patterns[k2],
        basis=LINEAR_SCVI_BASIS,
        use_rep=LINEAR_SCVI_LATENT_KEY,
    )

    g = sns.clustermap(
        t2,
        xticklabels=1,
        yticklabels=1,
        figsize=(6.5, 6.5),
    )

    g.savefig(f"{PREFIX}_{GENOME}_npisc_np_{k2}_npmat.png", dpi=300)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    sc.pl.umap(
        adatas_sbc[k2],
        color=["leiden"],
        legend_loc="on data",
        legend_fontoutline=2,
        outline_color="white",
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_{GENOME}_npisc_np_{k2}_umap.png")

    t3.to_csv(f"{PREFIX}_{GENOME}_npisc_np_{k2}_cos_theta.csv")

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax = plot_np(
        t3,
        gdfs[k2],
        ax,
        {
            "column": "cos_theta",
            "linewidth": 0.8,
            "categorical": False,
            "missing_kwds": {"color": "lightgrey"},
            "cmap": sns.color_palette("rocket", as_cmap=True),
        },
        {
            "color": "black",
            "fontsize": 6,
            "path_effects": [
                pe.withStroke(linewidth=2, foreground="white"),
            ],
        },
    )
    ax.set_title(k1)
    fig.tight_layout()
    patch_col = ax.collections[0]
    fig.colorbar(patch_col, ax=ax, shrink=0.5)
    fig.savefig(f"{PREFIX}_{GENOME}_npisc_np_{k2}_npmap.png")
    t2.to_csv(f"{PREFIX}_{GENOME}_npisc_np_{k2}_npmap.csv")
# %%

dg_sbc = make_digraph(
    adatas_sbc,
    [k for _, k, _ in STAGES_SUBCLUSTERS],
    use_rep=LINEAR_SCVI_LATENT_KEY,
)

nx.write_gml(dg_sbc, f"{PREFIX}_{GENOME}_npisc_np_cross_stage.gml")

# %%

d_leidens_sbc = {
    k: v.obs["leiden"].cat.categories.map(make_stage_name(k)).to_list()
    for k, v in adatas_sbc.items()
}

dg_sbc = nx.read_gml(f"{PREFIX}_{GENOME}_npisc_np_cross_stage.gml")

STATE_START = "midG"
STATE_END = "latN"

cs_sbc = CrossStage(
    dg_sbc,
    d_leidens_sbc[STATE_START],
    d_leidens_sbc[STATE_END],
)

# %%

for (_, k1, _), (_, k2, _) in adjacent(STAGES_SUBCLUSTERS):
    try:
        cs_sbc.update_internal(d_leidens_sbc[k1], d_leidens_sbc[k2])
        for i in d_leidens_sbc[k1]:
            print(cs_sbc.find_paths(i))
    except KeyError:
        pass
    except nx.NetworkXNoPath:
        pass

# %%

df_sbc_edges = nx.to_pandas_edgelist(dg_sbc)

for (_, k1, _), (_, k2, _) in adjacent(STAGES_SUBCLUSTERS):
    edges = df_sbc_edges[
        df_sbc_edges["source"].str.contains(k1)
        & df_sbc_edges["target"].str.contains(k2)
    ]

    pivot_edges = 1 - edges.pivot(
        index="source",
        columns="target",
        values="weight",
    ).fillna(0)

    g = sns.clustermap(
        pivot_edges,
        cmap="rocket",
        square=True,
        xticklabels=True,
        yticklabels=True,
        figsize=(6.5, 6.5),
    )

    g.savefig(f"{PREFIX}_{GENOME}_npisc_np_cross_stage_{k1}_{k2}.png", dpi=300)

# %%
