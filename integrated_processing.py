import numpy as np
import scanpy as sc
import scvi
import torch

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63
scvi.settings.seed = 0

PREFIX = "integrated"
GENOME = "ky21"

BATCH_KEY = "sample"
COUNTS_LAYER = "counts"
LIBRARY_SIZE = 1e4

adata = sc.read_h5ad(f"{PREFIX}_{GENOME}_raw.h5ad")

adata = adata[adata.obs["singlet"] > 0.5].copy()

# annotate the group of mitochondrial genes as 'mt'
QC_MITO = "mt"

adata.var[QC_MITO] = adata.var_names.str.startswith("KY21.MG0")
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=[QC_MITO],
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
    span=1.0,
)

model_configs = [
    (
        # validation loss 3297.597412109375
        scvi.model.LinearSCVI,
        "nb",
        256,
        40,
        3,
        "gene-batch",
        0.008471190328636252,
    ),
    (
        # validation loss 3227.56982421875
        scvi.model.SCVI,
        "nb",
        256,
        40,
        2,
        "gene-batch",
        0.009379014975295608,
    ),
]

for (
    model_class,
    gene_likelihood,
    n_hidden,
    n_latent,
    n_layers,
    dispersion,
    lr,
) in model_configs:
    model_class.setup_anndata(
        adata,
        layer=COUNTS_LAYER,
        batch_key=BATCH_KEY,
    )

    model = model_class(
        adata,
        gene_likelihood=gene_likelihood,
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_layers=n_layers,
        dispersion=dispersion,
    )

    model.train(
        check_val_every_n_epoch=1,
        max_epochs=800,
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_monitor="elbo_validation",
        plan_kwargs={"lr": lr},
    )

    model.save(
        f"{PREFIX}_{GENOME}_{model.__class__.__name__}",
        overwrite=True,
        save_anndata=True,
    )

models = {
    model.__name__: model.load(f"{PREFIX}_{GENOME}_{model.__name__}", adata=adata)
    for model, *_ in model_configs
}

LINEAR_SCVI_BASIS = "LinearSCVI_basis"

adata.varm[LINEAR_SCVI_BASIS] = models["LinearSCVI"].get_loadings()

LINEAR_SCVI_LATENT_KEY = "X_LinearSCVI"
adata.obsm[LINEAR_SCVI_LATENT_KEY] = models["LinearSCVI"].get_latent_representation()

SCVI_LATENT_KEY = "X_SCVI"
adata.obsm[SCVI_LATENT_KEY] = models["SCVI"].get_latent_representation()

SCVI_EXPRESSION_KEY = "SCVI_normalized"
adata.layers[SCVI_EXPRESSION_KEY] = models["SCVI"].get_normalized_expression(
    library_size=LIBRARY_SIZE,
)

SCVI_LOG1P_KEY = "SCVI_log1p"
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

SIZE_FACTOR = "size_factor"
library_size = adata.layers[COUNTS_LAYER].sum(1)
adata.obs[SIZE_FACTOR] = library_size / np.mean(library_size)

adata.write_h5ad(f"{PREFIX}_{GENOME}.h5ad")
