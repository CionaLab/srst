import pandas as pd
import scanpy as sc
import scvi
import torch

torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63
scvi.settings.seed = 0

PREFIX = "integrated"
GENOME = "ky21"
COUNTS_LAYER = "counts"

# cellranger mkref --genome=ky21_cellranger --fasta=HT.RefwMG0.fasta --genes=HT.KY21Gene.2.wMG0.sort.gtf
# STAR --runThreadN 64 --runMode genomeGenerate --genomeDir ky21_star --genomeFastaFiles HT.RefwMG0.fasta --sjdbGTFfile HT.KY21Gene.2.wMG0.sort.gtf --sjdbOverhang 149

# seq -f "SRR905%04g" 987 1007 | parallel --jobs 8 'cellranger count --id={} --transcriptome=ky21_cellranger --fastqs cao2019/{} --sample {} --create-bam true'
# seq -f "SRR811169%01g" 1 4 | parallel --jobs 8 'cellranger count --id={} --transcriptome=ky21_cellranger --fastqs sharma2019/{} --sample {} --create-bam true'
# seq -f "SRR1297104%01g" 2 5 | parallel --jobs 1 'STAR --runThreadN 64 --genomeDir ky21_star --readFilesIn winkley2021/raw_data/{}_R2.fastq.gz winkley2021/raw_data/{}_R1.fastq.gz --soloOutFileNames {}/ features.tsv barcodes.tsv matrix.mtx --soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 12 --soloUMIstart 13 --soloUMIlen 9 --soloCBwhitelist None --soloBarcodeReadLength 0 --clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTGAATGGG --readFilesCommand zcat'

samples = [
    (
        "SRR9050987",
        "midG",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050988",
        "midG",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050989",
        "earN",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050990",
        "earN",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050991",
        "latN",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050992",
        "latN",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050993",
        "iniT",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050994",
        "iniT",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050995",
        "earT",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050996",
        "earT",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050997",
        "midT",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050998",
        "midT",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9050999",
        "latTI",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051000",
        "latTI",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051001",
        "latTI",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051002",
        "latTII",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051003",
        "latTII",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051004",
        "latTII",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051005",
        "larva",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051006",
        "larva",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR9051007",
        "larva",
        "cao2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR8111691",
        "larva",
        "sharma2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR8111692",
        "larva",
        "sharma2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR8111693",
        "larva",
        "sharma2019",
        "outs/filtered_feature_bc_matrix",
    ),
    (
        "SRR8111694",
        "larva",
        "sharma2019",
        "outs/filtered_feature_bc_matrix",
    ),
    # seq -f "SRR1297104%01g" 2 5| parallel 'gzip {}/Gene/{filtered,raw}/{barcodes.tsv,features.tsv,matrix.mtx}'
    (
        "SRR12971042",
        "midG",
        "winkley2021",
        "Gene/filtered",
    ),
    (
        "SRR12971043",
        "midG",
        "winkley2021",
        "Gene/filtered",
    ),
    (
        "SRR12971044",
        "midG",
        "winkley2021",
        "Gene/filtered",
    ),
    (
        "SRR12971045",
        "midG",
        "winkley2021",
        "Gene/filtered",
    ),
]

for s, *_, p in samples:
    adata = sc.read_10x_mtx(
        f"/mnt/storage/Projects/SRR/{s}/{p}",
        var_names="gene_ids",
    )

    adata.layers[COUNTS_LAYER] = adata.X.copy()

    scvi.model.SCVI.setup_anndata(
        adata,
        layer=COUNTS_LAYER,
    )

    model = scvi.model.SCVI(
        adata,
        n_layers=2,
        n_latent=30,
        gene_likelihood="nb",
    )

    model.train(
        check_val_every_n_epoch=1,
        max_epochs=800,
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_monitor="elbo_validation",
    )

    solo = scvi.external.SOLO.from_scvi_model(model)
    solo.train(
        check_val_every_n_epoch=1,
        max_epochs=800,
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_monitor="elbo_validation",
    )

    df1 = solo.predict()

    adata.obs[["singlet", "doublet"]] = df1[["singlet", "doublet"]]

    adata.write_h5ad(f"/mnt/storage/Projects/SRR/{s}/{GENOME}_solo.h5ad")

adatas = [
    sc.read_h5ad(f"/mnt/storage/Projects/SRR/{s}/{GENOME}_solo.h5ad")
    for s, *_ in samples
]

for a, (_, j, k, *_) in zip(adatas, samples):
    a.obs["stage"] = j
    a.obs["source"] = k

adata = sc.concat(
    adatas,
    join="outer",
    fill_value=0,
    label="sample",
)
adata.obs_names_make_unique()
gene_names = pd.read_csv("npisc/ky2021_gene_names.tsv", sep="\t", index_col=0)
adata.var["gene_name"] = adata.var.index.map(gene_names["gene_name"])
adata.var_names_make_unique()
adata.obs["sample"] = adata.obs["sample"].astype("category")

adata.write_h5ad(f"{PREFIX}_{GENOME}_raw.h5ad")
