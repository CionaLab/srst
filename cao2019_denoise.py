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

BATCH_KEY = "sample"

adata_raw = sc.read_h5ad("cao2019_ky21_raw.h5ad")

scvi.external.SCAR.setup_anndata(
    adata,
    batch_key=BATCH_KEY,
)

scvi.external.SCAR.get_ambient_profile(
    adata=adata,
    raw_adata=adata_raw,
    prob=0.9,
)

model = scvi.external.SCAR(
    adata,
    ambient_profile="ambient_profile",
    n_hidden=256,
    n_latent=35,
    n_layers=2,
)

model.train(
    check_val_every_n_epoch=1,
    max_epochs=800,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="elbo_validation",
    plan_kwargs={"lr": 0.001687},
)

SCAR_LATENT_KEY = "X_scAR"
SCAR_LAYER = "denoised"

adata.obsm[SCAR_LATENT_KEY] = model.get_latent_representation()
adata.layers[SCAR_LAYER] = model.get_denoised_counts()

COUNTS_LAYER = "counts"
LIBRARY_SIZE = 1e4

adata.layers[COUNTS_LAYER] = adata.X.copy()
sc.pp.normalize_total(
    adata,
    target_sum=LIBRARY_SIZE,
)
sc.pp.log1p(adata)
adata.raw = adata

adata.write_h5ad("cao2019_ky21_denoised.h5ad")

model.save(
    "cao2019_ky21_scar",
    overwrite=True,
    save_anndata=True,
)
