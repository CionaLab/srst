import scanpy as sc
import scvi
import torch

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63

adata = sc.read_h5ad("cao2019_ky21_raw.h5ad")

# annotate the group of mitochondrial genes as 'mt'
adata.var["mt"] = adata.var_names.str.startswith("KY21.MG0")
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True,
)

sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.scrublet(adata, batch_key="sample")

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=8000,
    subset=True,
    batch_key="sample",
)

scvi.model.LinearSCVI.setup_anndata(adata, layer="counts", batch_key="sample")

model = scvi.model.LinearSCVI(
    adata,
    n_latent=10,
)

model.train(
    max_epochs=250,
    check_val_every_n_epoch=10,
)

SCVI_BASIS = "scVI_basis"

adata.varm["scVI_basis"] = model.get_loadings()

SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

SCVI_EXPRESSION_KEY = "scVI_normalized"
adata.layers[SCVI_EXPRESSION_KEY] = model.get_normalized_expression()

sc.pp.neighbors(
    adata,
    use_rep=SCVI_LATENT_KEY,
)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

SCVI_MDE_KEY = "X_scVI_MDE"
adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(
    adata.obsm[SCVI_LATENT_KEY], accelerator="cpu"
)

sc.pp.pca(
    adata,
    layer=SCVI_EXPRESSION_KEY,
    svd_solver="arpack",
)

sc.tl.umap(adata)

adata.write_h5ad("cao2019_ky21.h5ad")

model.save(
    "cao2019_ky21_scvi",
    overwrite=True,
    save_anndata=True,
)
