import scanpy as sc

samples = ["SRR8111691", "SRR8111692", "SRR8111693", "SRR8111694"]

adatas = [
    sc.read_10x_mtx(
        f"/home/mys_721tx/WorkSpace/ciona/tcga/data/{s}/outs/filtered_feature_bc_matrix"
    )
    for s in samples
]

for i, a in enumerate(adatas):
    a.obs["sample"] = i

adata = sc.concat(adatas)
adata.obs_names_make_unique()

adata.write_h5ad("sharma2019_ky21_raw.h5ad")
