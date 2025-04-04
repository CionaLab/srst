import numpy as np
import scanpy as sc
import scvi
import torch

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63

PREFIX = "integrated"
GENOME = "ky21"

BATCH_KEY = "sample"
COUNTS_LAYER = "counts"
LIBRARY_SIZE = 1e4

adata = sc.read_h5ad(f"{PREFIX}_{GENOME}_raw.h5ad")

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

adata.layers[COUNTS_LAYER] = adata.X.copy()
sc.pp.normalize_total(
    adata,
    target_sum=LIBRARY_SIZE,
)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=8000,
    layer=COUNTS_LAYER,
    subset=True,
    batch_key=BATCH_KEY,
)

scvi.model.LinearSCVI.setup_anndata(
    adata,
    layer=COUNTS_LAYER,
    batch_key=BATCH_KEY,
)

# validation loss 3059.297607421875
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
    plan_kwargs={"lr": 0.006092052120980108},
)

model.save(
    f"{PREFIX}_{GENOME}_linear_scVI",
    overwrite=True,
    save_anndata=True,
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

sc.pp.pca(
    adata,
    layer=SCVI_EXPRESSION_KEY,
    svd_solver="arpack",
)

sc.tl.umap(adata)

adata.write_h5ad(f"{PREFIX}_{GENOME}.h5ad")
