import pandas as pd
import scanpy as sc
import numpy as np
import scvi
import torch

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63
scvi.settings.seed = 0
scvi.settings.batch_size = 1024

PREFIX = "integrated"
GENOME = "ky21"

STAGES_SC = [
    "midG",
    "earN",
    "latN",
]

d_patterns = {
    k: pd.read_csv(
        f"{PREFIX}_{GENOME}_npisc_{k}_pattern.csv",
        index_col=0,
    )
    for k in STAGES_SC
}

adata = sc.read_h5ad(f"{PREFIX}_{GENOME}.h5ad")

model = scvi.model.LinearSCVI.load(
    f"{PREFIX}_{GENOME}_{scvi.model.LinearSCVI.__name__}"
)

adatas = {s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_{s}.h5ad") for s in STAGES_SC}

adatas_sbc = {
    s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_np_{s}.h5ad") for s in STAGES_SC
}

df_ky_sp = pd.read_csv("npisc/ky2021_swissprot_map.csv")

for k in STAGES_SC:

    de_change = model.differential_expression(
        adatas[k],
        groupby="leiden",
        weights="uniform",
        filter_outlier_cells=True,
        batch_correction=True,
        batch_size=scvi.settings.batch_size,
    )

    de_change["log10_pscore"] = np.log10(de_change["proba_m2"])
    de_change = de_change.join(
        adata.var,
        how="inner",
    ).merge(
        df_ky_sp,
        left_index=True,
        right_on="KY2021",
        how="left",
    )
    de_change["has_in_situ"] = de_change["KY2021"].apply(
        lambda x: x in d_patterns[k].columns
    )
    de_change.to_csv(
        f"{PREFIX}_{GENOME}_npisc_{k}_de.csv",
        index=False,
    )

for k in STAGES_SC:

    de_change = model.differential_expression(
        adatas_sbc[k],
        groupby="leiden",
        weights="uniform",
        filter_outlier_cells=True,
        batch_correction=True,
        batch_size=scvi.settings.batch_size,
    )

    de_change["log10_pscore"] = np.log10(de_change["proba_m2"])
    de_change = de_change.join(
        adata.var,
        how="inner",
    ).merge(
        df_ky_sp,
        left_index=True,
        right_on="KY2021",
        how="left",
    )
    de_change["has_in_situ"] = de_change["KY2021"].apply(
        lambda x: x in d_patterns[k].columns
    )
    de_change.to_csv(
        f"{PREFIX}_{GENOME}_npisc_np_{k}_de.csv",
        index=False,
    )
