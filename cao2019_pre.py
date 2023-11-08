import scanpy as sc

samples = ["SRR9051005", "SRR9051006", "SRR9051007"]

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

adata.write_h5ad("cao2019_ky21_raw.h5ad")
