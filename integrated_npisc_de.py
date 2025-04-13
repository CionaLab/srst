import re

import pandas as pd
import scanpy as sc
import numpy as np
import scvi
import torch

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63
scvi.settings.seed = 0

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

model = scvi.model.LinearSCVI.load(f"{PREFIX}_{GENOME}_linear_scVI")

adatas = {s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_{s}.h5ad") for s in STAGES_SC}

adatas_sbc = {
    s: sc.read_h5ad(f"{PREFIX}_{GENOME}_npisc_np_{s}.h5ad") for s in STAGES_SC
}

# Generate the BLAST map from the following:
# \time blastp -db swissprot -taxids 9606 -query ky2021p.fasta -parse_deflines \
# -outfmt "7 qacc sacc pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovs" \
# -out ky2021_swissprot.txt -num_threads 64 -mt_mode 1

df_ky_sp = pd.read_csv(
    "ky2021_swissprot.txt",
    sep="\t",
    names=[
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "coverage",
    ],
    comment="#",
)
df_ky_sp["qseqid"] = df_ky_sp["qseqid"].apply(lambda x: re.sub(r"\.v.+", "", x))

df_ky_sp = df_ky_sp.loc[df_ky_sp.groupby("qseqid")["evalue"].idxmin()]

(
    pd.merge(
        df_ky_sp,
        pd.read_csv("uniprot_data.csv"),
        left_on="sseqid",
        right_on="uniprot",
    )
    .groupby("qseqid")
    .apply(lambda x: x.nsmallest(1, "evalue"))
    .reset_index(drop=True)[["qseqid", "fullname"]]
).to_csv("ky2021_swissprot_map.csv", index=False)

df_ky_sp = pd.read_csv("ky2021_swissprot_map.csv")

for k in STAGES_SC:

    de_change = model.differential_expression(
        adatas[k],
        groupby="leiden",
        weights="uniform",
        filter_outlier_cells=True,
        batch_correction=True,
    )

    de_change["log10_pscore"] = np.log10(de_change["proba_not_de"])
    de_change = de_change.join(
        adata.var,
        how="inner",
    ).merge(
        df_ky_sp,
        left_index=True,
        right_on="qseqid",
        how="left",
    )
    de_change["has_in_situ"] = de_change["qseqid"].apply(
        lambda x: x in d_patterns[k].columns
    )
    de_change["is_tf"] = de_change["qseqid"].apply(
        lambda x: x in df_ky_sp["qseqid"].values
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
    )

    de_change["log10_pscore"] = np.log10(de_change["proba_not_de"])
    de_change = de_change.join(
        adata.var,
        how="inner",
    ).merge(
        df_ky_sp,
        left_index=True,
        right_on="qseqid",
        how="left",
    )
    de_change["has_in_situ"] = de_change["qseqid"].apply(
        lambda x: x in d_patterns[k].columns
    )
    de_change["is_tf"] = de_change["qseqid"].apply(
        lambda x: x in df_ky_sp["qseqid"].values
    )
    de_change.to_csv(
        f"{PREFIX}_{GENOME}_npisc_np_{k}_de.csv",
        index=False,
    )
