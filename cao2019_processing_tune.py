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

SCAR_LATENT_KEY = "X_scAR"
SCAR_LAYER = "denoised"
BATCH_KEY = "sample"
COUNTS_LAYER = "counts"
LIBRARY_SIZE = 1e4

adata = sc.read_h5ad("cao2019_ky21_denoised.h5ad")

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=8000,
    subset=True,
    batch_key=BATCH_KEY,
)

model_cls = scvi.model.LinearSCVI
model_cls.setup_anndata(
    adata,
    layer=SCAR_LAYER,
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

results.result_grid.get_dataframe().to_csv("cao2019_processing_hps.csv")

print(
    results.result_grid.get_best_result(
        "validation_loss",
        mode="min",
    )
)
