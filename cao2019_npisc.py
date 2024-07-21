# %%
import re

import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import geopandas as gpd

from npisc.build_matrix import (
    split_adata,
    get_distance,
    pad_compatible,
    append_raw,
    adjacent,
)

# %%
PATTERN_STAGES = r"(early|mid|late) (gastrula|neurula)"
PATTERN_CELLS = r"[Aa]\d+\.\d+$"

STAGES_IN_SITU = [
    ("mid gastrula", "mid_gastrula.geojson", 6, "midG"),
    ("late gastrula", "late_gastrula.geojson", 7, "latG"),
    ("early neurula", "early_neurula.geojson", 10, "earN"),
    ("mid neurula", "mid_neurula.geojson", 11, "midN"),
    ("late neurula", "late_neurula.geojson", 12, "latN"),
]

# %%
df = pd.read_csv("npisc/pass_02.tsv", sep="\t")
df_map = pd.read_csv("npisc/kh2012_ky2021_map.tsv", sep="\t")
df_map["query"] = "KH2012:" + df_map["query"]
df_map["subject"] = "KY21:" + df_map["subject"]
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

df_patterns = pd.concat(
    [df.rename(index=lambda x: f"{key}_{x}") for key, df in d_patterns.items()],
    axis=0,
    sort=False,
).fillna(False)

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

plots = ["leiden", "KY21:KY21.Chr2.490"]

adatas = {s: sc.read_h5ad(f"cao2019_npisc_ky21_{s}.h5ad") for s in STAGES_SC}

adatas_raw = {
    k: append_raw(v, adata_raw[v.obs.index, v.var.index]) for k, v in adatas.items()
}

# %%

STAGES_SUBCLUSTER = [
    ("mid gastrula", "midG"),
    ("early neurula", "earN"),
    ("late neurula", "latN"),
]

# %%

STAGES_GEOJSON = [
    ("mid gastrula", "mid_gastrula.geojson", 6),
    ("late gastrula", "late_gastrula.geojson", 7),
    ("early neurula", "early_neurula.geojson", 10),
    ("mid neurula", "mid_neurula.geojson", 11),
    ("late neurula", "late_neurula.geojson", 12),
]

gdfs = {stage: (gpd.read_file(f"npisc/{file}"), l) for stage, file, l in STAGES_GEOJSON}


# %%
for k1, k2 in STAGES_SUBCLUSTER:
    print(k1)
    d1, a1 = pad_compatible(d_patterns[k1], adatas_raw[k2])
    sc.pp.normalize_total(a1, target_sum=1e4)
    sc.pp.log1p(a1)
    sc.tl.pca(a1, svd_solver="arpack")

    t = 1 - get_distance(
        d1 @ a1.varm["PCs"],
        pd.DataFrame(a1.obsm["X_pca"], index=a1.obs_names),
        "cosine",
    )

    sns.clustermap(t)

    t2 = t.T.join(a1.obs["leiden"], how="left").groupby("leiden").mean()
    sns.clustermap(t2.T)

    t3 = pd.DataFrame({"cluster": t2.idxmax(axis=0), "cos_theta": t2.max(axis=0)})

    t3.to_csv(f"cao2019_npisc_ky21_{k1}_cos_theta.csv")

    v, l = gdfs[k1]
    v = v.merge(t3, left_on="name", right_index=True)
    fig, ax = plt.subplots(1, 1)
    v.plot(
        column="cos_theta",
        cmap="rocket",
        ax=ax,
        linewidth=0.8,
        edgecolor="0.8",
        legend=True,
        legend_kwds={"shrink": 0.3},
        vmax=0.7,
        vmin=0.1,
    )
    v.apply(
        lambda x: ax.annotate(
            text=f"{x["name"]}\n{x["cluster"]}",
            xy=x.geometry.centroid.coords[0],
            ha="center",
            color="white",
            fontsize=12,
        ),
        axis=1,
    )
    fig.set_size_inches(6, l)
    plt.axis("off")
    plt.show()

# %%
for k1, _ in STAGES_SUBCLUSTER:
    sns.clustermap(d_patterns[k1], cbar_pos=None)

# %%
sc.pl.umap(adatas["midG"], color=["leiden"], legend_loc="on data")
sc.pl.umap(adatas["midG"], color=["KY21:KY21.Chr1.422"])

# %%

NUM_TOP = 50

for k in STAGES_SC:
    value = sc.read_h5ad(f"cao2019_npisc_ky21_{k}.h5ad")
    sc.tl.rank_genes_groups(value, "leiden", method="t-test")
    result = value.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    df_diff = pd.concat(
        [
            pd.DataFrame(
                {
                    "group": group,
                    "gene": result["names"][group],
                    "p_adj": result["pvals_adj"][group],
                    "log2fc": result["logfoldchanges"][group],
                }
            )
            for group in groups
        ]
    )

    df_diff.groupby("group").apply(
        lambda g: g[g["p_adj"] < 0.05]  # Filter step
        .sort_values(by="log2fc", ascending=False)  # Sort step
        .head(NUM_TOP)  # Select top 50 step
    ).reset_index(drop=True).to_csv(
        f"cao2019_npisc_ky21_{k}_top{NUM_TOP}.csv", index=False
    )

# %%

for i, (k1, k2) in enumerate(adjacent(STAGES_SC)):
    print(k1, k2)

    a1, a2 = pad_compatible(adatas_raw[k1], adatas_raw[k2])
    sc.pp.normalize_total(a1, target_sum=1e4)
    sc.pp.log1p(a1)
    sc.tl.pca(a1, svd_solver="arpack")
    sc.pp.normalize_total(a2, target_sum=1e4)
    sc.pp.log1p(a2)
    sc.tl.pca(a2, svd_solver="arpack")

    a1.write_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k1}.h5ad")
    a2.write_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k2}.h5ad")

# %%

for i, (k1, k2) in enumerate(adjacent(STAGES_SC)):
    print(k1, k2)

    a1 = sc.read_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k1}.h5ad")
    a2 = sc.read_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k2}.h5ad")

    t1 = 1 - get_distance(
        pd.DataFrame(a1.X @ a2.varm["PCs"], index=a1.obs_names),
        pd.DataFrame(a2.obsm["X_pca"], index=a2.obs_names),
        "cosine",
    )

    t2 = 1 - get_distance(
        pd.DataFrame(a2.X @ a1.varm["PCs"], index=a2.obs_names),
        pd.DataFrame(a1.obsm["X_pca"], index=a1.obs_names),
        "cosine",
    )

    t1.to_csv(f"cao2019_npisc_ky21_stage_stage_{i}_{k1}_cos_theta.csv")
    t2.to_csv(f"cao2019_npisc_ky21_stage_stage_{i}_{k2}_cos_theta.csv")

# %%

for i, (k1, k2) in enumerate(adjacent(STAGES_SC)):
    print(k1, k2)

    t1 = pd.read_csv(
        f"cao2019_npisc_ky21_stage_stage_{i}_{k1}_cos_theta.csv", index_col=0
    )
    t2 = pd.read_csv(
        f"cao2019_npisc_ky21_stage_stage_{i}_{k2}_cos_theta.csv", index_col=0
    )
    a1 = sc.read_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k1}.h5ad")
    a2 = sc.read_h5ad(f"cao2019_npisc_ky21_stage_stage_{i}_{k2}.h5ad")

    d1 = (
        t1.melt(ignore_index=False, var_name="target", value_name="cos_theta")
        .reset_index(names="source")
        .merge(a1.obs["leiden"], left_on=["source"], right_index=True)
        .merge(
            a2.obs["leiden"],
            left_on=["target"],
            right_index=True,
            suffixes=(f"_{k1}", f"_{k2}"),
        )
        .groupby([f"leiden_{k1}", f"leiden_{k2}"])["cos_theta"]
        .mean()
        .reset_index()
    )

    d2 = (
        t2.melt(ignore_index=False, var_name="target", value_name="cos_theta")
        .reset_index(names="source")
        .merge(a2.obs["leiden"], left_on=["source"], right_index=True)
        .merge(
            a1.obs["leiden"],
            left_on=["target"],
            right_index=True,
            suffixes=(f"_{k2}", f"_{k1}"),
        )
        .groupby([f"leiden_{k2}", f"leiden_{k1}"])["cos_theta"]
        .mean()
        .reset_index()
    )

    d3 = d1.pivot(index=f"leiden_{k1}", columns=f"leiden_{k2}", values="cos_theta")
    d4 = d2.pivot(index=f"leiden_{k2}", columns=f"leiden_{k1}", values="cos_theta")

    sns.clustermap(d3 + d4.T)

# %%
