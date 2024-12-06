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
    split_adata,
    map_cells,
    plot_np,
    make_stage_name,
    make_digraph,
    CrossStage,
)

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63

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

SCVI_LATENT_KEY = "X_scVI"
SCVI_BASIS = "scVI_basis"
SCVI_MDE_KEY = "X_scVI_MDE"
SCVI_EXPRESSION_KEY = "scVI_normalized"
SCVI_LOG1P_KEY = "scVI_log1p"

# %%

adata = sc.read_h5ad("cao2019_ky21.h5ad")
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

    a_tmp.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(
        a_tmp.obsm[SCVI_LATENT_KEY],
        accelerator="cpu",
    )

    sc.pp.pca(
        a_tmp,
        layer=SCVI_EXPRESSION_KEY,
        svd_solver="arpack",
    )

    sc.tl.umap(a_tmp)

    a_tmp.write_h5ad(f"cao2019_npisc_ky21_{key}.h5ad")

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

adatas = {s: sc.read_h5ad(f"cao2019_npisc_ky21_{s}.h5ad") for s in STAGES_SC}

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

# %%

for k1, k2 in STAGES_MAPPING:
    t1, t2, t3 = map_cells(
        adatas[k2], d_patterns[k1], basis=SCVI_BASIS, use_rep=SCVI_LATENT_KEY
    )

    g = sns.clustermap(
        t2,
        xticklabels=1,
        yticklabels=1,
        figsize=(6.5, 9),
    )

    g.savefig(f"cao2019_npisc_ky21_{k2}_npmat.png", dpi=300)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    sc.pl.embedding(
        adatas[k2],
        basis=SCVI_MDE_KEY,
        color=["leiden"],
        legend_loc="on data",
        legend_fontoutline=2,
        outline_color="white",
        ax=ax,
    )
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
    fig.savefig(f"cao2019_npisc_ky21_{k2}_npmap.png")

# %%

for k1, k2 in STAGES_MAPPING:
    fig, ax = plt.subplots(figsize=(6.5, 9), dpi=300)
    vp = sc.pl.stacked_violin(
        adatas[k2],
        adatas[k2].var_names.intersection(d_patterns[k1].columns),
        groupby="leiden",
        layer=SCVI_LOG1P_KEY,
        ax=ax,
        return_fig=True,
    )
    d_ax = vp.get_axes()
    d_ax["mainplot_ax"].set_xticklabels(
        adatas[k2]
        .var.loc[adatas[k2].var_names.intersection(d_patterns[k1].columns)]["gene_name"]
        .to_list(),
    )

    fig.tight_layout()
    fig.savefig(f"cao2019_npisc_ky21_{k2}_markers.png")

# %%

dg = make_digraph(adatas, STAGES_SC, use_rep=SCVI_LATENT_KEY)

nx.write_gml(dg, "cao2019_npisc_ky21_cross_stage.gml")

# %%

d_leidens = {
    k: v.obs["leiden"].cat.categories.map(make_stage_name(k)).to_list()
    for k, v in adatas.items()
}

dg = nx.read_gml("cao2019_npisc_ky21_cross_stage.gml")

STATE_START = "midG"
STATE_END = "latN"

cs = CrossStage(
    dg,
    d_leidens[STATE_START],
    d_leidens[STATE_END],
)

# %%

NP_SOURCES = [
    (
        "midG",
        (
            "11",
            "14",
            "7",
            "9",
        ),
    ),
    (
        "earN",
        (
            "4",
            "12",
            "9",
            "13",
        ),
    ),
]

for s, l in NP_SOURCES:
    try:
        cs.update_internal(d_leidens[f"{s}"], d_leidens[STATE_END])
        for i in l:
            print(cs.find_paths(f"{s}_{i}"))
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
            "11",
            "14",
            "7",
            "9",
        ),
    ),
    (
        "early neurula",
        "earN",
        (
            "4",
            "12",
            "9",
            "13",
        ),
    ),
    (
        "late neurula",
        "latN",
        (
            "23",
            "2",
            "29",
            "3",
            "20",
            "9",
            "11",
            "27",
            "18",
            "6",
            "30",
        ),
    ),
]

adatas = {s: sc.read_h5ad(f"cao2019_npisc_ky21_{s}.h5ad") for s in STAGES_SC}

for _, k2, sbc in STAGES_SUBCLUSTERS:
    a_tmp = adatas[k2][adatas[k2].obs["leiden"].isin(sbc)].copy()

    sc.pp.neighbors(
        a_tmp,
        n_neighbors=50,
        use_rep=SCVI_LATENT_KEY,
    )

    sc.tl.leiden(
        a_tmp,
        flavor="igraph",
        n_iterations=-1,
    )

    a_tmp.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(
        a_tmp.obsm[SCVI_LATENT_KEY],
        accelerator="cpu",
    )

    sc.pp.pca(
        a_tmp,
        layer=SCVI_EXPRESSION_KEY,
        svd_solver="arpack",
    )

    sc.tl.umap(a_tmp)

    a_tmp.write_h5ad(f"cao2019_npisc_ky21_np_{k2}.h5ad")

# %%

adatas_sbc = {
    s: sc.read_h5ad(f"cao2019_npisc_ky21_np_{s}.h5ad") for _, s, _ in STAGES_SUBCLUSTERS
}

# %%

for k1, k2, _ in STAGES_SUBCLUSTERS:
    t1, t2, t3 = map_cells(
        adatas_sbc[k2], d_patterns[k1], basis=SCVI_BASIS, use_rep=SCVI_LATENT_KEY
    )

    g = sns.clustermap(
        t2,
        xticklabels=1,
        yticklabels=1,
        figsize=(6.5, 9),
    )

    g.savefig(f"cao2019_npisc_ky21_np_{k2}_npmat.png", dpi=300)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    sc.pl.embedding(
        adatas_sbc[k2],
        basis=SCVI_MDE_KEY,
        color=["leiden"],
        legend_loc="on data",
        legend_fontoutline=2,
        outline_color="white",
        ax=ax,
    )
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
    fig.savefig(f"cao2019_npisc_ky21_np_{k2}_npmap.png")

# %%

for k1, k2 in STAGES_MAPPING:
    fig, ax = plt.subplots(figsize=(6.5, 9), dpi=300)
    vp = sc.pl.stacked_violin(
        adatas_sbc[k2],
        adatas_sbc[k2].var_names.intersection(d_patterns[k1].columns),
        groupby="leiden",
        layer=SCVI_LOG1P_KEY,
        ax=ax,
        return_fig=True,
    )
    d_ax = vp.get_axes()
    d_ax["mainplot_ax"].set_xticklabels(
        adatas_sbc[k2]
        .var.loc[adatas_sbc[k2].var_names.intersection(d_patterns[k1].columns)][
            "gene_name"
        ]
        .to_list(),
    )

    fig.tight_layout()
    fig.savefig(f"cao2019_npisc_ky21_np_{k2}_markers.png")

# %%

dg_sbc = make_digraph(
    adatas_sbc,
    [k for _, k, _ in STAGES_SUBCLUSTERS],
    use_rep=SCVI_LATENT_KEY,
)

nx.write_gml(dg_sbc, "cao2019_npisc_ky21_np_cross_stage.gml")

# %%

d_leidens_sbc = {
    k: v.obs["leiden"].cat.categories.map(make_stage_name(k)).to_list()
    for k, v in adatas_sbc.items()
}

dg_sbc = nx.read_gml("cao2019_npisc_ky21_np_cross_stage.gml")

STATE_START = "midG"
STATE_END = "latN"

cs_sbc = CrossStage(
    dg_sbc,
    d_leidens_sbc[STATE_START],
    d_leidens_sbc[STATE_END],
)

# %%

for k, v in d_leidens_sbc.items():
    try:
        cs_sbc.update_internal(d_leidens_sbc[k], d_leidens_sbc[STATE_END])
        for i in v:
            print(cs_sbc.find_paths(i))
    except KeyError:
        pass
    except nx.NetworkXNoPath:
        pass

# %%
