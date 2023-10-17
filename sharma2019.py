# %%
import matplotlib.pyplot as plt
import bottleneck
import numpy as np
import pandas as pd
import scanpy as sc

# %%
sc.settings.verbosity = 3
sc.logging.print_header()

# %%

samples = ["sharma2019"]

adatas = [
    sc.read_10x_mtx(
        f"/home/mys_721tx/WorkSpace/ciona/tcga/data/{s}/outs/filtered_feature_bc_matrix"
    )
    for s in samples
]


# %%
for i, a in enumerate(adatas):
    a.obs["sample"] = i

adata = sc.concat(adatas)

# %%
adata.write_h5ad("sharma2019_ky21_raw.h5ad")

# %%
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# %%
# annotate the group of mitochondrial genes as 'mt'
adata.var["mt"] = adata.var_names.str.startswith("ENSCIN")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], jitter=0.4)

# %%
adata = adata[adata.obs.n_genes_by_counts < 3000, :]
adata = adata[adata.obs.pct_counts_mt < 20, :]

# %%
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# %%
adata.raw = adata

# %%
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])

# %%
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata)
sc.tl.paga(adata)
sc.pl.paga(
    adata, plot=False
)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata, init_pos="paga")

# %%
adata.write("sharma2019_ky21.h5ad")

# %%
adata = sc.read_h5ad("sharma2019_ky21.h5ad")

# %%
cngs = [
    "KY21:KY21.Chr3.483",
    "KY21:KY21.Chr7.267",
    "KY21:KY21.Chr2.475",
    "KY21:KY21.Chr4.987",
]

opsins = ["KY21:KY21.Chr12.157", "KY21:KY21.Chr11.800", "KY21:KY21.Chr9.932"]

sc.pl.dotplot(adata, ["KY21:KY21.Chr2.490", *cngs, *opsins], groupby="leiden")

# %%
sc.pl.umap(adata, color=["leiden", "KY21:KY21.Chr2.490"])

# %%
sc.pl.umap(
    adata, color=["KY21:KY21.Chr3.1172", "KY21:KY21.Chr1.783", "KY21:KY21.Chr2.793"]
)

# %%
gluts = [
    "KY21:KY21.Chr2.734",
    "KY21:KY21.Chr13.442",
    "KY21:KY21.Chr1.1346",
    "KY21:KY21.Chr12.947",
    "KY21:KY21.Chr4.989",
]

vts = ["KY21:KY21.Chr2.793", "KY21:KY21.Chr1.783", "KY21:KY21.Chr3.1172"]

sc.pl.dotplot(adata, [*gluts, *vts], groupby="leiden")

# %%
adata = sc.read_h5ad("sharma2019_ky21_raw.h5ad")
sc.pp.normalize_total(adata, target_sum=1e6)
