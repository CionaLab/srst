# %% [markdown]
# scVI integration of Cao 2019, Sharma 2019 and Winkley 2021. Batch = source.

# %% Setup
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import scanpy as sc
import scvi
import torch
from ray import tune
from scvi import autotune
from sklearn.metrics import adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

torch.set_float32_matmul_precision("high")
scvi.settings.seed = 0
scvi.settings.batch_size = 1024
scvi.settings.dl_num_workers = 0

PREFIX = "integrated"
GENOME = "ky21"

BATCH_KEY = "source"
STAGE_KEY = "stage"
COUNTS_LAYER = "counts"
LIBRARY_SIZE = 1e4
N_HVG = 5000
MIN_GENES = 100
N_MADS = 5
N_MADS_MT = None
MAX_PCT_MT = 80
DOUBLET_KEY = "sample"
N_KL_WARMUP = 50
N_POST_EXPR = 25
REF_SOURCE = None
NUM_SAMPLES = 32
MAX_EPOCHS_TUNE = 200
TRIALS_PER_GPU = 4
CPUS_PER_TRIAL = 8

TOP_K = 5
K_NN = 30
MIN_CELLS_STAGE = 50
MAX_CELLS_METRIC = 20000
SUBSET_SOURCES = ["sharma2019"]
LABEL_KEY = "cao_tissue_type"
MAX_CELLS_SCIB = 30000
TRANSFER_KEY = f"{LABEL_KEY}_knn"
K_TRANSFER = 15
SHARMA_SOURCE = "sharma2019"
CNS_LABEL = "nervous system"
WINKLEY_LABEL_KEY = "winkley_celltype"
SYSVI_CYCLE_WEIGHTS = [2, 5, 10]
SYSVI_PRIOR_COMPONENTS = 5

SCVI_LATENT_KEY = "X_scVI"
SCVI_EXPRESSION_KEY = "scVI_normalized"
SCVI_LOG1P_KEY = "scVI_log1p"
SIZE_FACTOR = "size_factor"

# %% Load
adata = sc.read_h5ad(f"{PREFIX}_{GENOME}_raw.h5ad")
for k in (BATCH_KEY, STAGE_KEY):
    assert k in adata.obs, f"missing adata.obs['{k}']"

# %% QC
QC_MITO = "mt"
adata.var[QC_MITO] = adata.var_names.str.startswith("KY21.MG0")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=[QC_MITO], percent_top=None, log1p=False, inplace=True
)


def mad_outlier(x, n_mads, side="both"):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    lo, hi = x < med - n_mads * mad, x > med + n_mads * mad
    return {"both": lo | hi, "upper": hi}[side]


obs = adata.obs
outlier = np.zeros(adata.n_obs, dtype=bool)
for src, idx in obs.groupby(BATCH_KEY, observed=True).indices.items():
    o = obs.iloc[idx]
    outlier[idx] = mad_outlier(
        np.log1p(o["total_counts"].to_numpy()), N_MADS
    ) | mad_outlier(np.log1p(o["n_genes_by_counts"].to_numpy()), N_MADS)
    if N_MADS_MT is not None:
        outlier[idx] |= mad_outlier(
            o[f"pct_counts_{QC_MITO}"].to_numpy(), N_MADS_MT, "upper"
        )
adata.obs["qc_outlier"] = outlier
print(pd.crosstab(adata.obs[BATCH_KEY], adata.obs["qc_outlier"]))
sc.pl.violin(
    adata,
    ["total_counts", "n_genes_by_counts", f"pct_counts_{QC_MITO}"],
    groupby=BATCH_KEY,
    log=True,
    multi_panel=True,
    save=f"_{PREFIX}_{GENOME}_qc.png",
)

adata = adata[
    ~adata.obs["qc_outlier"]
    & (adata.obs["n_genes_by_counts"] >= MIN_GENES)
    & (adata.obs[f"pct_counts_{QC_MITO}"] < MAX_PCT_MT)
].copy()
sc.pp.filter_genes(adata, min_cells=3)

# %% Doublets
sc.pp.scrublet(adata, batch_key=DOUBLET_KEY)
print(pd.crosstab(adata.obs[BATCH_KEY], adata.obs["predicted_doublet"]))
adata = adata[~adata.obs["predicted_doublet"]].copy()

# %% Gene coverage per source
for src, idx in adata.obs.groupby(BATCH_KEY, observed=True).indices.items():
    n_zero = int((np.asarray(adata.X[idx].sum(0)).ravel() == 0).sum())
    print(f"{src}: {n_zero} / {adata.n_vars} genes with zero counts")

# %% Normalisation / HVGs
adata.layers[COUNTS_LAYER] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=LIBRARY_SIZE)
sc.pp.log1p(adata)
adata.raw = adata

library_size = np.asarray(adata.layers[COUNTS_LAYER].sum(1)).ravel()
adata.obs[SIZE_FACTOR] = library_size / library_size.mean()

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=N_HVG,
    layer=COUNTS_LAYER,
    subset=True,
    batch_key=BATCH_KEY,
)

# %% Stage coverage per source
print(pd.crosstab(adata.obs[BATCH_KEY], adata.obs[STAGE_KEY]))

# %% Search space
model_cls = scvi.model.SCVI
model_cls.setup_anndata(adata, layer=COUNTS_LAYER, batch_key=BATCH_KEY)

search_space = {
    "model_params": {
        "n_hidden": tune.choice([128, 256]),
        "n_latent": tune.choice([10, 20, 30]),
        "n_layers": tune.choice([1, 2, 3]),
        "dropout_rate": tune.choice([0.05, 0.1]),
        "dispersion": "gene-batch",
        "gene_likelihood": "nb",
    },
    "train_params": {
        "max_epochs": MAX_EPOCHS_TUNE,
        "plan_kwargs": {
            "lr": tune.loguniform(1e-4, 1e-2),
            "n_epochs_kl_warmup": N_KL_WARMUP,
        },
        "batch_size": scvi.settings.batch_size,
    },
}

# %% Run autotune
ray.init(log_to_driver=False)
results = autotune.run_autotune(
    model_cls,
    data=adata,
    mode="min",
    metrics="elbo_validation",
    search_space=search_space,
    num_samples=NUM_SAMPLES,
    resources={"cpu": CPUS_PER_TRIAL, "gpu": 1 / TRIALS_PER_GPU},
    scheduler="asha",
    scheduler_kwargs={
        "max_t": MAX_EPOCHS_TUNE,
        "grace_period": N_KL_WARMUP + 10,
        "reduction_factor": 2,
    },
    ignore_reinit_error=True,
)
results.result_grid.get_dataframe().to_csv(
    f"{PREFIX}_{GENOME}_processing_{model_cls.__name__}_hps.csv"
)
top = sorted(
    (r for r in results.result_grid if r.metrics and "elbo_validation" in r.metrics),
    key=lambda r: r.metrics["elbo_validation"],
)[:TOP_K]
ray.shutdown()

# %% Best config by ELBO
print("Top configs by elbo_validation:")
for i, r in enumerate(top):
    lr_i = r.config["train_params"]["plan_kwargs"]["lr"]
    print(
        f"  [{i}] elbo={r.metrics['elbo_validation']:.4f}  lr={lr_i:.2e}  {r.config['model_params']}"
    )

# %% Task metric
obs_all = adata.obs[[BATCH_KEY, STAGE_KEY]].astype(str)
missing = set(SUBSET_SOURCES) - set(obs_all[BATCH_KEY])
assert not missing, f"SUBSET_SOURCES not found in adata.obs['{BATCH_KEY}']: {missing}"

n_cells = obs_all.groupby([BATCH_KEY, STAGE_KEY]).size()
present = n_cells[n_cells >= MIN_CELLS_STAGE].reset_index()[[BATCH_KEY, STAGE_KEY]]
align_pool = present[~present[BATCH_KEY].isin(SUBSET_SOURCES)]
shared_stages = (
    align_pool.groupby(STAGE_KEY)[BATCH_KEY]
    .nunique()
    .loc[lambda x: x >= 2]
    .index.tolist()
)
print("Shared stages used for alignment:", shared_stages)
assert (
    shared_stages
), "no stage is shared by two whole-embryo sources; check stage labels"


def knn(z, k):
    k = min(k, len(z) - 1)
    return (
        NearestNeighbors(n_neighbors=k + 1)
        .fit(z)
        .kneighbors(z, return_distance=False)[:, 1:]
    )


def alignment_score(z):
    """Cross-source kNN mixing at shared stages, relative to perfect mixing."""
    scores = []
    for st in shared_stages:
        m = (obs_all[STAGE_KEY] == st) & ~obs_all[BATCH_KEY].isin(SUBSET_SOURCES)
        idx = np.flatnonzero(m.to_numpy())
        src = obs_all[BATCH_KEY].to_numpy()[idx]
        nn = knn(z[idx], K_NN)
        other = (src[nn] != src[:, None]).mean(1)
        share = (
            pd.Series(src).map(pd.Series(src).value_counts(normalize=True)).to_numpy()
        )
        per_cell = np.minimum(other / (1 - share), 1.0)
        scores += pd.Series(per_cell).groupby(src).mean().tolist()
    return float(np.mean(scores))


def stage_score(z, rng):
    """kNN balanced accuracy of stage within each source."""
    scores = []
    for src, grp in present.groupby(BATCH_KEY):
        if len(grp) < 2:
            continue
        m = (obs_all[BATCH_KEY] == src) & obs_all[STAGE_KEY].isin(grp[STAGE_KEY])
        idx = np.flatnonzero(m.to_numpy())
        if len(idx) > MAX_CELLS_METRIC:
            idx = rng.choice(idx, MAX_CELLS_METRIC, replace=False)
        stg = obs_all[STAGE_KEY].to_numpy()[idx]
        nn = knn(z[idx], K_NN)
        pred = pd.DataFrame(stg[nn]).mode(axis=1)[0].to_numpy()
        scores.append(balanced_accuracy_score(stg, pred))
    return float(np.mean(scores))


def transfer_labels(z):
    """kNN vote of Cao tissue labels onto every cell; labelled cells vote without themselves."""
    ref = np.flatnonzero(adata.obs[LABEL_KEY].notna().to_numpy())
    ref_lab = adata.obs[LABEL_KEY].astype(str).to_numpy()[ref]
    nn = (
        NearestNeighbors(n_neighbors=K_TRANSFER + 1)
        .fit(z[ref])
        .kneighbors(z, return_distance=False)
    )
    is_ref = np.zeros(len(z), dtype=bool)
    is_ref[ref] = True
    nn = np.where(is_ref[:, None], nn[:, 1:], nn[:, :-1])
    votes = pd.DataFrame(ref_lab[nn])
    pred = votes.mode(axis=1)[0].to_numpy()
    conf = (votes.to_numpy() == pred[:, None]).mean(1)
    return pred, conf


def transfer_checks(pred):
    """Sharma cells called nervous system, and agreement with Winkley's own annotation."""
    out = {}
    sharma = (adata.obs[BATCH_KEY] == SHARMA_SOURCE).to_numpy()
    if sharma.any():
        out["sharma_to_cns"] = float((pred[sharma] == CNS_LABEL).mean())
    if WINKLEY_LABEL_KEY in adata.obs:
        w = adata.obs[WINKLEY_LABEL_KEY].notna().to_numpy()
        if w.any():
            out["winkley_ami"] = adjusted_mutual_info_score(
                adata.obs[WINKLEY_LABEL_KEY].astype(str).to_numpy()[w], pred[w]
            )
    return out


# %% Re-train top configs and rank
rows, models = [], {}
for i, r in enumerate(top):
    params = r.config["model_params"]
    lr_i = r.config["train_params"]["plan_kwargs"]["lr"]
    m = model_cls(adata, **params)
    m.train(
        check_val_every_n_epoch=1,
        max_epochs=800,
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_monitor="elbo_validation",
        plan_kwargs={"lr": lr_i, "n_epochs_kl_warmup": N_KL_WARMUP},
        batch_size=scvi.settings.batch_size,
    )
    z = m.get_latent_representation()
    key = f"X_scVI_top{i}"
    adata.obsm[key] = z
    a, b = alignment_score(z), stage_score(z, np.random.default_rng(0))
    checks = transfer_checks(transfer_labels(z)[0]) if LABEL_KEY is not None else {}
    rows.append(
        {
            "embedding": key,
            "model": "scVI",
            "rank_elbo": i,
            **params,
            "lr": lr_i,
            "epochs": len(m.history["elbo_validation"]),
            "elbo_validation": float(m.history["elbo_validation"].iloc[:, 0].min()),
            "alignment": a,
            "stage_preservation": b,
            **checks,
        }
    )
    models[key] = m
    print(rows[-1])

# %% sysVI candidates
sys_adata = ad.AnnData(
    X=adata.X.copy(),
    obs=adata.obs[[BATCH_KEY]].copy(),
    var=pd.DataFrame(index=adata.var_names),
)
scvi.external.SysVI.setup_anndata(sys_adata, batch_key=BATCH_KEY)
arch = {
    k: top[0].config["model_params"][k]
    for k in ("n_hidden", "n_latent", "n_layers", "dropout_rate")
}

for w in SYSVI_CYCLE_WEIGHTS:
    m = scvi.external.SysVI(
        sys_adata, prior="vamp", n_prior_components=SYSVI_PRIOR_COMPONENTS, **arch
    )
    m.train(
        check_val_every_n_epoch=1,
        max_epochs=800,
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_monitor="validation_loss",
        plan_kwargs={"z_distance_cycle_weight": w},
        batch_size=scvi.settings.batch_size,
    )
    z = m.get_latent_representation()
    key = f"X_sysVI_cyc{w}"
    adata.obsm[key] = z
    a, b = alignment_score(z), stage_score(z, np.random.default_rng(0))
    checks = transfer_checks(transfer_labels(z)[0]) if LABEL_KEY is not None else {}
    rows.append(
        {
            "embedding": key,
            "model": "sysVI",
            **arch,
            "cycle_weight": w,
            "epochs": len(m.history["validation_loss"]),
            "alignment": a,
            "stage_preservation": b,
            **checks,
        }
    )
    models[key] = m
    print(rows[-1])

ranking = pd.DataFrame(rows)

# %% Tissue conservation on Cao labels
if LABEL_KEY is not None:
    from scib_metrics.benchmark import Benchmarker, BioConservation

    idx = np.flatnonzero(adata.obs[LABEL_KEY].notna().to_numpy())
    if len(idx) > MAX_CELLS_SCIB:
        idx = np.random.default_rng(0).choice(idx, MAX_CELLS_SCIB, replace=False)
    lab = adata[np.sort(idx)].copy()
    lab.obs[LABEL_KEY] = lab.obs[LABEL_KEY].cat.remove_unused_categories()
    keys = list(models)
    bm = Benchmarker(
        lab,
        batch_key=BATCH_KEY,
        label_key=LABEL_KEY,
        embedding_obsm_keys=keys,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=None,
        n_jobs=min(16, os.cpu_count()),
    )
    bm.benchmark()
    scib = bm.get_results(min_max_scale=False)
    scib.to_csv(f"{PREFIX}_{GENOME}_scib.csv")
    print(scib)
    ranking["tissue_conservation"] = (
        scib.loc[ranking["embedding"], "Bio conservation"].astype(float).to_numpy()
    )

components = [
    c
    for c in ("alignment", "stage_preservation", "tissue_conservation")
    if c in ranking
]
ranking["task_score"] = ranking[components].mean(axis=1)
ranking = ranking.sort_values("task_score", ascending=False)
ranking.to_csv(
    f"{PREFIX}_{GENOME}_processing_{model_cls.__name__}_ranking.csv", index=False
)
print(ranking.to_string(index=False))

# %% Final model
best_key = ranking.iloc[0]["embedding"]
model = models[best_key]
best_scvi_key = ranking.loc[ranking["model"] == "scVI", "embedding"].iloc[0]
expr_model = models[best_scvi_key]
print(f"Selected {best_key} by task score; expression from {best_scvi_key}:")
print(ranking.iloc[0].to_string())
adata.uns["scVI_selected_config"] = {
    k: (v.item() if hasattr(v, "item") else v)
    for k, v in ranking.iloc[0].dropna().to_dict().items()
}
adata.uns["scVI_expression_model"] = best_scvi_key
model.save(
    f"{PREFIX}_{GENOME}_{model.__class__.__name__}",
    overwrite=True,
    save_anndata=True,
)

if expr_model is not model:
    expr_model.save(
        f"{PREFIX}_{GENOME}_SCVI_expression", overwrite=True, save_anndata=True
    )

hist = model.history
train_key, val_key = (
    ("train_loss_epoch", "validation_loss")
    if isinstance(model, scvi.external.SysVI)
    else ("elbo_train", "elbo_validation")
)
plt.plot(hist[train_key], label="train")
plt.plot(hist[val_key], label="validation")
plt.xlabel("epoch")
plt.ylabel(val_key)
plt.legend()
plt.savefig(
    f"{PREFIX}_{GENOME}_training.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# %% Latent space and batch-corrected expression
adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

ref = REF_SOURCE or adata.obs[BATCH_KEY].value_counts().idxmax()
print(f"Expression projected into source: {ref}")
adata.uns["scVI_transform_batch"] = str(ref)
adata.layers[SCVI_EXPRESSION_KEY] = expr_model.get_normalized_expression(
    library_size=LIBRARY_SIZE,
    transform_batch=ref,
    n_samples=N_POST_EXPR,
    return_mean=True,
).to_numpy(dtype=np.float32)
adata.layers[SCVI_LOG1P_KEY] = np.log1p(adata.layers[SCVI_EXPRESSION_KEY])

# %% Cao label transfer
if LABEL_KEY is not None:
    pred, conf = transfer_labels(adata.obsm[SCVI_LATENT_KEY])
    adata.obs[TRANSFER_KEY] = pd.Categorical(pred)
    adata.obs[f"{TRANSFER_KEY}_conf"] = conf
    print(
        pd.crosstab(
            adata.obs[BATCH_KEY], adata.obs[TRANSFER_KEY], normalize="index"
        ).round(3)
    )
    lab = adata.obs[LABEL_KEY].notna()
    print(
        f"Cao self-agreement: {(adata.obs.loc[lab, LABEL_KEY].astype(str) == adata.obs.loc[lab, TRANSFER_KEY].astype(str)).mean():.3f}"
    )
    if WINKLEY_LABEL_KEY in adata.obs:
        w = adata.obs[WINKLEY_LABEL_KEY].notna()
        print(
            pd.crosstab(
                adata.obs.loc[w, WINKLEY_LABEL_KEY], adata.obs.loc[w, TRANSFER_KEY]
            )
        )

# %% Neighbours, Leiden, UMAP
sc.pp.neighbors(
    adata,
    use_rep=SCVI_LATENT_KEY,
)
sc.tl.leiden(
    adata,
    flavor="igraph",
    n_iterations=-1,
    directed=False,
    key_added="leiden",
)
sc.tl.umap(adata)

sc.pp.pca(
    adata,
    layer=SCVI_LOG1P_KEY,
    svd_solver="arpack",
)

# %% Integration check
sc.pl.umap(
    adata,
    color=[BATCH_KEY, STAGE_KEY, "leiden"]
    + ([LABEL_KEY, TRANSFER_KEY] if LABEL_KEY else []),
    ncols=3,
    wspace=0.4,
    save=f"_{PREFIX}_{GENOME}_integration.png",
)

# %% Save
adata.write_h5ad(f"{PREFIX}_{GENOME}.h5ad")
