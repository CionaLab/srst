import scanpy as sc

samples = [
    ("SRR9050987", "midG"),
    ("SRR9050988", "midG"),
    ("SRR9050989", "earN"),
    ("SRR9050990", "earN"),
    ("SRR9050991", "latN"),
    ("SRR9050992", "latN"),
    ("SRR9050993", "iniT"),
    ("SRR9050994", "iniT"),
    ("SRR9050995", "earT"),
    ("SRR9050996", "earT"),
    ("SRR9050997", "midT"),
    ("SRR9050998", "midT"),
    ("SRR9050999", "latTI"),
    ("SRR9051000", "latTI"),
    ("SRR9051001", "latTI"),
    ("SRR9051002", "latTII"),
    ("SRR9051003", "latTII"),
    ("SRR9051004", "latTII"),
    ("SRR9051005", "larva"),
    ("SRR9051006", "larva"),
    ("SRR9051007", "larva"),
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

adata.write_h5ad("cao2019_ky21_raw.h5ad")
