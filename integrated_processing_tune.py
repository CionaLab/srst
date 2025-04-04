# Install dependencies for Ray from PyPI
# pip install -U "ray[data,train,tune,serve,rllib,default]" "hyperopt"

import scanpy as sc
import scvi
import torch
import ray

from ray import tune
from scvi import autotune

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

model_cls = scvi.model.LinearSCVI
model_cls.setup_anndata(
    adata,
    layer=COUNTS_LAYER,
    batch_key=BATCH_KEY,
)

search_space = {
    "model_params": {
        "n_hidden": tune.choice(
            [
                64,
                128,
                256,
            ]
        ),
        "n_latent": tune.choice(
            [
                10,
                20,
                30,
                40,
            ]
        ),
        "n_layers": tune.choice(
            [
                1,
                2,
                3,
                4,
            ]
        ),
        "dispersion": "gene-batch",
    },
    "train_params": {
        "max_epochs": 100,
        "plan_kwargs": {"lr": tune.loguniform(1e-4, 1e-2)},
    },
}

ray.init(log_to_driver=False)

results = autotune.run_autotune(
    model_cls,
    data=adata,
    mode="min",
    metrics="validation_loss",
    search_space=search_space,
    num_samples=192,
    resources={
        "cpu": 63,
        "gpu": 1,
    },
)

results.result_grid.get_dataframe().to_csv(
    f"{PREFIX}_{GENOME}_processing_linear_scVI_hps.csv",
)

print(
    results.result_grid.get_best_result(
        "validation_loss",
        mode="min",
    )
)

ray.shutdown()
