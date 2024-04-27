# %%
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from pandas.core.frame import DataFrame
from anndata import AnnData

from npisc.build_matrix import (
    split_adata,
    get_distance,
    to_df,
    find_coi,
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

count_dfs = {
    stage: df.groupby("Territory_eq")["Gene"]
    .nunique()
    .reset_index()
    .rename(columns={"Gene": "n"})
    for stage, df in dfs.items()
}

pattern_dfs = {
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

pattern_df = pd.concat(
    [df.rename(index=lambda x: f"{key}_{x}") for key, df in pattern_dfs.items()],
    axis=0,
    sort=False,
).fillna(False)

# %%
adata = sc.read_h5ad("cao2019_ky21.h5ad")

# %%
adatas = split_adata(adata, "stage")

# %%
for key in adatas.keys():
    adata = adatas[key]

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga")

    adata.write_h5ad(f"cao2019_npisc_ky21_{key}.h5ad")

# %%
STAGES_SC = ["midG", "earN", "latN", "iniT", "earT", "midT", "latTI", "latTII", "larva"]

plots = ["leiden", "KY21:KY21.Chr2.490"]

adatas = {
    stage: sc.read_h5ad(f"cao2019_npisc_ky21_{stage}.h5ad") for stage in STAGES_SC
}

# %%

STAGES_SUBCLUSTER = [
    ("mid gastrula", "midG"),
    ("early neurula", "earN"),
    ("late neurula", "latN"),
]

# %%
df_coi = (
    pd.concat(
        [
            find_coi(
                1 - get_distance(pattern_dfs[k1], adatas[k2], metric="cosine").T,
                adatas[k2],
            ).assign(stage=k2)
            for k1, k2 in STAGES_SUBCLUSTER
        ]
    )
    .reset_index(drop=True)
    .groupby("stage")
    .apply(lambda x: x[x["value"] > 0.06])
    .reset_index(drop=True)
)

df_coi.to_csv("cao2019_npisc_ky21_cois.csv", index=False)

# %%
df_coi = pd.read_csv("cao2019_npisc_ky21_cois.csv")

STAGES_COI = [
    (stage, tuple(map(str, df["leiden"].values)))
    for stage, df in df_coi.groupby("stage")
]


# %%
for k, c in STAGES_COI:
    adata_filtered = adatas[k][adatas[k].obs["leiden"].isin(c)]
    sc.tl.pca(adata_filtered, svd_solver="arpack")
    sc.pp.neighbors(adata_filtered, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata_filtered)
    sc.tl.paga(adata_filtered)
    sc.pl.paga(adata_filtered, plot=False)
    sc.tl.umap(adata_filtered, init_pos="paga")
    adata_filtered.write(f"cao2019_npisc_ky21_coi_{k}.h5ad")

# %%
for k1, k2 in STAGES_SUBCLUSTER:
    print(k1, k2)

    adata_filtered = sc.read_h5ad(f"cao2019_npisc_ky21_coi_{k2}.h5ad")
    df1 = to_df(adata_filtered)
    df2 = to_df(pattern_dfs[k1])

    df = 1 - get_distance(df2, df1, metric="cosine").T
    sns.clustermap(df, cmap="viridis", xticklabels=True, yticklabels=False)
    adata_filtered.obs = adata_filtered.obs.join(df)
    adata_filtered.obs["top_cluster"] = df.apply(
        lambda s: ", ".join(s.nlargest(1).index.tolist()), axis=1
    )
    adata_filtered.obs["top_cluster"] = adata_filtered.obs["top_cluster"].astype(
        "category"
    )

    for c in df.columns:
        sc.pl.umap(adata_filtered, color=[c])

    sc.pl.umap(adata_filtered, color=["leiden"], legend_loc="on data")
    sc.pl.umap(adata_filtered, color=["top_cluster"])

# %%
num_top = 50
value = sc.read_h5ad("cao2019_npisc_ky21_midG_np.h5ad")
sc.tl.rank_genes_groups(value, f"leiden", method="t-test")
result = value.uns["rank_genes_groups"]
groups = result["names"].dtype.names

df = pd.concat(
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

df.groupby("group").apply(
    lambda g: g[g["p_adj"] < 0.05]  # Filter step
    .sort_values(by="log2fc", ascending=False)  # Sort step
    .head(num_top)  # Select top 50 step
).reset_index(drop=True).to_csv(
    f"cao2019_npisc_ky21_midG_np_top{num_top}.csv", index=False
)

# %%
