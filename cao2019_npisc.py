# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from npisc.build_matrix import (
    preprocess_tsv,
    build_from_df,
    split_adata,
    get_cos_similarity,
    pairwise,
    find_similar_clusters,
    plot_distance,
)

# %%
adata = sc.read_h5ad("cao2019_ky21.h5ad")
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
stages = ["midG", "earN", "latN", "iniT", "earT", "midT", "latTI", "latTII", "larva"]

plots = ["leiden", "KY21:KY21.Chr2.490"]

adatas = {stage: sc.read_h5ad(f"cao2019_npisc_ky21_{stage}.h5ad") for stage in stages}

# %%
for s1, s2 in pairwise(stages):
    df = find_similar_clusters(adatas[s1], adatas[s2])
    df.to_csv(f"map_{s1}_{s2}.csv")
    print(s1, s2)
    matrix = df.pivot(index="leiden", columns="leiden_2", values="similarity")
    plot_distance(matrix)

# %%

for s in stages:
    print(s)
    sc.pl.dotplot(adatas[s], ["KY21:KY21.Chr3.483"], groupby="leiden", vmin=0, vmax=4)
    sc.pl.umap(adatas[s], color=["KY21:KY21.Chr3.483"])
    # for p in plots:
    #     sc.pl.umap(adatas[s], color=[p])

# %%
df = preprocess_tsv(
    "npisc/all_territories.tsv",
    "kh2012_ky21_map.txt",
    "npisc/developmental_ontology.txt",
)

map_stage = {
    "early neurula": "earN",
    "late gastrula": "latG",
    "late neurula": "latN",
    "mid gastrula": "midG",
    "mid neurula": "midN",
}

stages = {
    map_stage.get(stage, None): build_from_df(stage_df)
    for stage, stage_df in df.groupby("Stage")
}

# %%
for key, value in stages.items():
    print(key)
    try:
        df = get_cos_similarity(value, adatas[key])
        group = adatas[key].obs["leiden"]
        mean_sim = df.groupby(group, axis=1).mean()
        plot_distance(mean_sim)
    except KeyError:
        pass

# %%
genes = [
    f"KY21:{gene}"
    for gene in [
        "KY21.Chr2.490",
        "KY21.Chr3.244",
        # "KY21.Chr3.483",
        # "KY21.Chr4.922",
        # "KY21.Chr1.783",
        # "KY21.Chr6.24",
        # "KY21.Chr11.1113",
        # "KY21.Chr10.362",
        # "KY21.Chr11.1087",
        # "KY21.Chr1.783",
        # "KY21.Chr2.1082",
        # "KY21.Chr11.539"
        # "KY21.Chr3.511",
        # "KY21.Chr1.838",
        # "KY21.Chr1.563",
        # "KY21.Chr4.768",
        # "KY21.Chr2.793",
        # "KY21.Chr3.483",
        # "KY21.Chr11.539",
    ]
]

plots = ["leiden"] + genes

# %%

for key, value in adatas.items():
    print(key)
    # sc.pl.umap(value, color=["KY21:KY21.Chr2.490", "KY21:KY21.Chr11.539"])
    # sc.pl.umap(value, color=["KY21:KY21.Chr2.490", "KY21:KY21.Chr3.483"])
    # sc.pl.umap(value, color=["KY21:KY21.Chr2.490", "KY21:KY21.Chr12.157"])
    for p in plots:
        sc.pl.umap(value, color=[p], vmin=0, vmax=4)


# %%

for p in plots:
    sc.pl.umap(
        adata,
        color=[p],
    )

# %%
for key, value in adatas.items():
    sc.tl.pca(value, svd_solver="arpack")
    sc.pp.neighbors(value, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(value)
    sc.tl.paga(value)
    sc.pl.paga(
        value, plot=False
    )  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(value, init_pos="paga")
    value.write(f"cao2019_npisc_ky21_{key}.h5ad")

# %%
for key in adatas.keys():
    value = sc.read_h5ad(f"cao2019_npisc_ky21_{key}.h5ad")
    print(key)
    # for p in plots:
    #     sc.pl.umap(value, color=[p], vmin=0, vmax=4)
    sc.pl.dotplot(value, genes, groupby="leiden", vmin=0, vmax=4)

# %%
num_top = 50
for key in adatas.keys():
    value = sc.read_h5ad(f"cao2019_npisc_ky21_{key}.h5ad")
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
        f"cao2019_npisc_diff_{key}_top{num_top}.csv", index=False
    )

# %%
