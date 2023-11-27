import scanpy as sc

samples = [
    ("SRR8111691", "larva"),
    ("SRR8111692", "larva"),
    ("SRR8111693", "larva"),
    ("SRR8111694", "larva"),
]

adatas = [
    sc.read_10x_mtx(
        f"/home/mys_721tx/WorkSpace/ciona/tcga/data/{s}/outs/filtered_feature_bc_matrix",
        var_names="gene_ids",
    )
    for s, _ in samples
]


for i, (a, (_, j)) in enumerate(zip(adatas, samples)):
    a.obs["sample"] = i
    a.obs["stage"] = j

adata = sc.concat(adatas, fill_value=0)
adata.obs_names_make_unique()
adata.var_names_make_unique()
adata.obs["sample"] = adata.obs["sample"].astype("category")

adata.write_h5ad("sharma2019_ky21_raw.h5ad")
