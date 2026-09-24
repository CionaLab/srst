# %% [markdown]
# Build the raw count AnnData for Cao 2019, Sharma 2019 and Winkley 2021 on KY21.

# %% Commands
# cellranger mkref --genome=ky21_cellranger --fasta=HT.RefwMG0.fasta --genes=HT.KY21Gene.2.wMG0.sort.gtf
# STAR --runThreadN 64 --runMode genomeGenerate --genomeDir ky21_star --genomeFastaFiles HT.RefwMG0.fasta --sjdbGTFfile HT.KY21Gene.2.wMG0.sort.gtf --sjdbOverhang 149
#
# seq -f "SRR905%04g" 985 1007 | parallel --jobs 8 'cellranger count --id={} --transcriptome=ky21_cellranger --fastqs cao2019/{} --sample {} --create-bam true'
# parallel --colsep ' ' 'cellranger count --id=sharma2019_{1} --transcriptome=ky21_cellranger --fastqs sharma2019/{2},sharma2019/{3},sharma2019/{4},sharma2019/{5} --sample {2},{3},{4},{5} --localcores 64 --create-bam true' ::: "larva SRR8111691 SRR8111692 SRR8111693 SRR8111694"
# parallel --jobs 1 --colsep ' ' 'mkdir -p winkley2021_{1} && STAR --runThreadN 64 --genomeDir ky21_star --readFilesIn winkley2021/{2}/{2}_2.fastq.gz,winkley2021/{3}/{3}_2.fastq.gz,winkley2021/{4}/{4}_2.fastq.gz,winkley2021/{5}/{5}_2.fastq.gz winkley2021/{2}/{2}_1.fastq.gz,winkley2021/{3}/{3}_1.fastq.gz,winkley2021/{4}/{4}_1.fastq.gz,winkley2021/{5}/{5}_1.fastq.gz --readFilesCommand zcat --outFileNamePrefix winkley2021_{1}/ --outSAMtype None --soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 12 --soloUMIstart 13 --soloUMIlen 9 --soloCBwhitelist None --soloBarcodeReadLength 0 --soloFeatures Gene GeneFull --soloCellFilter EmptyDrops_CR --soloOutFileNames ./ features.tsv barcodes.tsv matrix.mtx --clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTGAATGGG' ::: "c64 SRR12971026 SRR12971027 SRR12971028 SRR12971029" "iniG SRR12971034 SRR12971035 SRR12971036 SRR12971037" "midG SRR12971042 SRR12971043 SRR12971044 SRR12971045"
# parallel -j 64 gzip -f {1}/{2}/{3}/{4} ::: winkley2021_c64 winkley2021_iniG winkley2021_midG ::: Gene GeneFull ::: filtered raw ::: barcodes.tsv features.tsv matrix.mtx

# %% Setup
import itertools

import pandas as pd
import scanpy as sc

PREFIX = "integrated"
GENOME = "ky21"
ROOT = "/mnt/storage/Projects/SRR"
GENE_NAMES = "npisc/ky2021_gene_names.tsv"
STAGE_ORDER = [
    "c64",
    "iniG",
    "midG",
    "earN",
    "latN",
    "iniT",
    "earT",
    "midT",
    "latTI",
    "latTII",
    "larva",
]
MAX_BARCODE_JACCARD = 0.1
CAO_META = "npisc/cao2019_meta.tsv"
CAO_LABELS = ["tissue_type", "clusters", "batch"]
WINKLEY_META = "npisc/winkley2021_meta.tsv"
WINKLEY_LABELS = ["CellType", "Cluster", "group", "adult"]
MIN_FRAC_MATCHED = 0.5
CAO_STAGE = {
    "C110": "iniG",
    "midG": "midG",
    "earlyN": "earN",
    "lateN": "latN",
    "ITB": "iniT",
    "ETB": "earT",
    "MTB": "midT",
    "LTB1": "latTI",
    "LTB2": "latTII",
    "lv": "larva",
}

CR = "outs/filtered_feature_bc_matrix"
SS = "GeneFull/filtered"
samples = [
    ("SRR9050985", "iniG", "cao2019", CR),
    ("SRR9050986", "iniG", "cao2019", CR),
    ("SRR9050987", "midG", "cao2019", CR),
    ("SRR9050988", "midG", "cao2019", CR),
    ("SRR9050989", "earN", "cao2019", CR),
    ("SRR9050990", "earN", "cao2019", CR),
    ("SRR9050991", "earT", "cao2019", CR),
    ("SRR9050992", "earT", "cao2019", CR),
    ("SRR9050993", "latN", "cao2019", CR),
    ("SRR9050994", "latN", "cao2019", CR),
    ("SRR9050995", "iniT", "cao2019", CR),
    ("SRR9050996", "iniT", "cao2019", CR),
    ("SRR9050997", "midT", "cao2019", CR),
    ("SRR9050998", "midT", "cao2019", CR),
    ("SRR9050999", "latTI", "cao2019", CR),
    ("SRR9051000", "latTI", "cao2019", CR),
    ("SRR9051001", "latTI", "cao2019", CR),
    ("SRR9051002", "latTII", "cao2019", CR),
    ("SRR9051003", "latTII", "cao2019", CR),
    ("SRR9051004", "latTII", "cao2019", CR),
    ("SRR9051005", "larva", "cao2019", CR),
    ("SRR9051006", "larva", "cao2019", CR),
    ("SRR9051007", "larva", "cao2019", CR),
    ("sharma2019_larva", "larva", "sharma2019", CR),
    ("winkley2021_c64", "c64", "winkley2021", SS),
    ("winkley2021_iniG", "iniG", "winkley2021", SS),
    ("winkley2021_midG", "midG", "winkley2021", SS),
]
meta = pd.DataFrame(samples, columns=["sample", "stage", "source", "path"]).set_index(
    "sample"
)

# %% Read
adatas = {
    s: sc.read_10x_mtx(f"{ROOT}/{s}/{p}", var_names="gene_ids")
    for s, p in meta["path"].items()
}

var0 = adatas[meta.index[0]].var_names
for s, a in adatas.items():
    assert a.var_names.equals(var0), f"{s}: gene set differs from {meta.index[0]}"

# %% Runs of one library
# Lanes of one library share cell barcodes; independent libraries share almost none.
rows = []
for (src, stg), grp in meta.groupby(["source", "stage"]):
    for a, b in itertools.combinations(grp.index, 2):
        x = set(adatas[a].obs_names.str.replace(r"-\d+$", "", regex=True))
        y = set(adatas[b].obs_names.str.replace(r"-\d+$", "", regex=True))
        rows.append((src, stg, a, b, len(x), len(y), len(x & y) / len(x | y)))
overlap = pd.DataFrame(
    rows, columns=["source", "stage", "run_a", "run_b", "n_a", "n_b", "jaccard"]
)
print(overlap.to_string(index=False))
shared = overlap[overlap["jaccard"] > MAX_BARCODE_JACCARD]
assert shared.empty, (
    "runs share barcodes and are lanes of one library; rerun counting with their "
    f"FASTQs together:\n{shared.to_string(index=False)}"
)

# %% Concatenate
adata = sc.concat(adatas, join="inner", label="sample", index_unique="-")
adata.obs = adata.obs.join(meta[["stage", "source"]], on="sample")
adata.obs["sample"] = adata.obs["sample"].astype("category")
adata.obs["source"] = adata.obs["source"].astype("category")
adata.obs["stage"] = pd.Categorical(
    adata.obs["stage"], categories=STAGE_ORDER, ordered=True
)

gene_names = pd.read_csv(GENE_NAMES, sep="\t", index_col=0)
adata.var["gene_name"] = adata.var_names.map(gene_names["gene_name"])

print(pd.crosstab(adata.obs["source"], adata.obs["stage"]))
print(adata.obs.groupby("sample", observed=True).size().describe())


# %% Label transfer
def transfer_labels(adata, source, ref, labels, tag):
    """Match published libraries to runs by barcode overlap, then attach per-cell labels."""
    is_src = (adata.obs["source"] == source).to_numpy()
    ours = pd.DataFrame(
        {
            "sample": adata.obs["sample"].astype(str).to_numpy()[is_src],
            "barcode": adata.obs_names[is_src].str.split("-").str[0],
        },
        index=adata.obs_names[is_src],
    )

    n_published = len(ref)
    hits = ours.merge(ref[["library", "barcode"]], on="barcode")
    overlap = pd.crosstab(hits["library"], hits["sample"])
    frac = overlap.div(ref["library"].value_counts().reindex(overlap.index), axis=0)
    frac = frac.reindex(ref["library"].unique(), fill_value=0)
    match = pd.DataFrame(
        {
            "sample": frac.idxmax(axis=1),
            "frac_matched": frac.max(axis=1).round(3),
            "n_cells": ref["library"].value_counts(),
        }
    )
    matched = match["frac_matched"] > MIN_FRAC_MATCHED
    match["sample"] = match["sample"].where(matched)
    match["run_stage"] = match["sample"].map(meta["stage"])
    print(match.sort_index().to_string())
    print(
        f"{tag}: published libraries without a run here:", list(match.index[~matched])
    )
    runs = meta.index[meta["source"] == source]
    print(
        f"{tag}: runs without published cells:",
        sorted(set(runs) - set(match.loc[matched, "sample"])),
    )

    ref = ref.assign(sample=ref["library"].map(match.loc[matched, "sample"])).dropna(
        subset=["sample"]
    )
    ref["sample"] = ref["sample"].astype(str)
    key = lambda s, b: s + "|" + b
    lab = ref.set_index(key(ref["sample"], ref["barcode"]))[labels]
    assert lab.index.is_unique, f"{tag}: two published cells map to the same cell here"
    lab.columns = f"{tag}_" + lab.columns.str.lower().str.replace(
        r"\W+", "_", regex=True
    ).str.strip("_")
    lab = lab.reindex(key(ours["sample"], ours["barcode"]))
    lab.index = ours.index
    adata.obs = adata.obs.join(lab)
    for c in lab.columns:
        adata.obs[c] = adata.obs[c].astype("category")

    found = lab.notna().any(axis=1)
    print(f"{tag}: cells here with a published label: {found.mean():.1%}")
    print(f"{tag}: published cells found here: {found.sum() / n_published:.1%}")
    return adata, match


# %% Cao 2019 annotation
cao = pd.read_csv(CAO_META, sep="\t", index_col=0, dtype={c: str for c in CAO_LABELS})
adata, cao_match = transfer_labels(adata, "cao2019", cao, CAO_LABELS, "cao")
m = cao_match[cao_match["frac_matched"] > MIN_FRAC_MATCHED]
assert m["sample"].is_unique, "two Cao libraries match the same run"
expected = m.index.str.split(".").str[0].map(CAO_STAGE)
assert (
    expected == m["run_stage"].to_numpy()
).all(), "a Cao library matched a run of a different stage"

# %% Winkley 2021 annotation
winkley = pd.read_csv(WINKLEY_META, sep="\t", index_col=0, dtype=str)
parts = winkley.index.to_series().str.extract(r"^(?P<group>.+)_(?P<barcode>[ACGTN]+)$")
assert parts["barcode"].notna().all(), "unparsed Winkley cell names"
winkley["group"] = parts["group"]
winkley["barcode"] = parts["barcode"]
winkley["library"] = winkley["group"] + ":" + winkley["adult"]
adata, winkley_match = transfer_labels(
    adata, "winkley2021", winkley, WINKLEY_LABELS, "winkley"
)

# %% Save
adata.write_h5ad(f"{PREFIX}_{GENOME}_raw.h5ad")
