import numpy as np
import scanpy as sc
import scvi
import torch

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63

PREFIX = "integrated"
GENOME = "ky21"

SCAR_LATENT_KEY = "X_scAR"
SCAR_LAYER = "denoised"
BATCH_KEY = "sample"
COUNTS_LAYER = "counts"
LIBRARY_SIZE = 1e4

adata = sc.read_h5ad(f"{PREFIX}_{GENOME}_denoised.h5ad")

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=8000,
    subset=True,
    batch_key=BATCH_KEY,
)

scvi.model.LinearSCVI.setup_anndata(
    adata,
    layer=SCAR_LAYER,
    batch_key=BATCH_KEY,
)

model = scvi.model.LinearSCVI(
    adata,
    gene_likelihood="nb",
    n_hidden=256,
    n_latent=40,
    n_layers=1,
    dispersion="gene-batch",
)

model.train(
    check_val_every_n_epoch=1,
    max_epochs=800,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="elbo_validation",
    plan_kwargs={"lr": 0.003223950082189226},
)

SCVI_BASIS = "scVI_basis"

adata.varm[SCVI_BASIS] = model.get_loadings()

SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

SCVI_EXPRESSION_KEY = "scVI_normalized"
adata.layers[SCVI_EXPRESSION_KEY] = model.get_normalized_expression(
    library_size=LIBRARY_SIZE,
)

SCVI_LOG1P_KEY = "scVI_log1p"
adata.layers[SCVI_LOG1P_KEY] = np.log1p(
    adata.layers[SCVI_EXPRESSION_KEY],
)

sc.pp.neighbors(
    adata,
    use_rep=SCVI_LATENT_KEY,
)
sc.tl.leiden(
    adata,
    flavor="igraph",
    n_iterations=-1,
)

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

adata.write_h5ad(f"{PREFIX}_{GENOME}.h5ad")

model.save(
    f"{PREFIX}_{GENOME}_scvi",
    overwrite=True,
    save_anndata=True,
)
