# Code for "Spatially resolving single-cell transcriptomes in the embryo of the marine invertebrate *Ciona*"
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21298925.svg)](https://doi.org/10.5281/zenodo.21298925)

This repository contains code for the study
"Spatially resolving single-cell transcriptomes in the embryo of the marine
invertebrate *Ciona*", including integrated preprocessing and NPISC (Neural
Plate in situ Integration with Single Cell) workflows.

## Software

Analysis is performed in Python 3.12.

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Setup

Initialize and fetch the `npisc` submodule:

```bash
git submodule update --init --recursive npisc
```

## Data Sources

This analysis integrates publicly available datasets from multiple studies,
including:

* [Cao et al. 2019](https://doi.org/10.1038/s41586-019-1385-y)
* [Sharma et al. 2019](https://doi.org/10.1016/j.ydbio.2018.09.023)
* [Winkley et al. 2021](https://doi.org/10.1186/s12915-021-01122-0)

## Download SRRs (SRA-tools)

Use SRA-tools (`prefetch` and `fasterq-dump`) to download and convert the SRRs.
The commands below organize files into source-specific directories used by
`integrated_pre.py`.

```bash
mkdir -p cao2019 sharma2019 winkley2021

# Cao et al. 2019: SRR9050985-SRR9051007
for srr in $(seq -f "SRR905%04g" 985 1007); do
  mkdir -p "cao2019/${srr}"
  prefetch "$srr"
  fasterq-dump "$srr" --split-files --threads 16 --outdir "cao2019/${srr}"
  gzip "cao2019/${srr}/${srr}_1.fastq" "cao2019/${srr}/${srr}_2.fastq" 2>/dev/null || true
done

# Sharma et al. 2019: SRR8111691-SRR8111694
for srr in $(seq -f "SRR811169%01g" 1 4); do
  mkdir -p "sharma2019/${srr}"
  prefetch "$srr"
  fasterq-dump "$srr" --split-files --threads 16 --outdir "sharma2019/${srr}"
  gzip "sharma2019/${srr}/${srr}_1.fastq" "sharma2019/${srr}/${srr}_2.fastq" 2>/dev/null || true
done

# Winkley et al. 2021 SRRs used here
for srr in $( { seq -f "SRR1297102%01g" 6 9; seq -f "SRR1297103%01g" 4 7; seq -f "SRR1297104%01g" 2 5; } ); do
  mkdir -p "winkley2021/${srr}"
  prefetch "$srr"
  fasterq-dump "$srr" --split-files --threads 16 --outdir "winkley2021/${srr}"
  gzip "winkley2021/${srr}/${srr}_1.fastq" "winkley2021/${srr}/${srr}_2.fastq"
done
```

## Genome Index And Alignment

Generate references for Cell Ranger and STAR:

```bash
cellranger mkref \
  --genome=ky21_cellranger \
  --fasta=HT.RefwMG0.fasta \
  --genes=HT.KY21Gene.2.wMG0.sort.gtf

STAR --runThreadN 64 \
  --runMode genomeGenerate \
  --genomeDir ky21_star \
  --genomeFastaFiles HT.RefwMG0.fasta \
  --sjdbGTFfile HT.KY21Gene.2.wMG0.sort.gtf \
  --sjdbOverhang 149
```

Run quantification/alignment:

```bash
# Cao et al. 2019 (Cell Ranger)
seq -f "SRR905%04g" 985 1007 | parallel --jobs 8 \
  'cellranger count --id={} --transcriptome=ky21_cellranger --fastqs cao2019/{} --sample {} --create-bam true'

# Sharma et al. 2019 (Cell Ranger)
seq -f "SRR811169%01g" 1 4 | parallel --jobs 8 \
  'cellranger count --id={} --transcriptome=ky21_cellranger --fastqs sharma2019/{} --sample {} --create-bam true'

# Winkley et al. 2021 (STARsolo)
{ seq -f "SRR1297102%01g" 6 9; seq -f "SRR1297103%01g" 4 7; seq -f "SRR1297104%01g" 2 5; } \
| parallel --jobs 1 \
  'STAR --runThreadN 64 --genomeDir ky21_star --readFilesIn winkley2021/{}/{}_2.fastq.gz winkley2021/{}/{}_1.fastq.gz --soloOutFileNames {}/ features.tsv barcodes.tsv matrix.mtx --soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 12 --soloUMIstart 13 --soloUMIlen 9 --soloCBwhitelist None --soloBarcodeReadLength 0 --clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTGAATGGG --readFilesCommand zcat'

# Optional: gzip STARsolo matrix outputs expected by read_10x_mtx
{ seq -f "SRR1297102%01g" 6 9; seq -f "SRR1297103%01g" 4 7; seq -f "SRR1297104%01g" 2 5; } \
| parallel 'gzip {}/Gene/{filtered,raw}/{barcodes.tsv,features.tsv,matrix.mtx}'
```

## Pipeline

Run the analysis in this order:

1. Generate the integrated raw AnnData object:

```bash
python integrated_pre.py
```

2. Tune LDVAE hyperparameters:

```bash
python integrated_processing_tune.py
```

3. Update the tuned hyperparameters in `integrated_processing.py`, then run
  processing:

```bash
python integrated_processing.py
```

4. Run full NPISC analysis interactively (required to update some results
  during execution):

```bash
python integrated_npisc.py
```

5. Run differential expression analysis:

```bash
python integrated_npisc_de.py
```

## Repository Layout

* `npisc/`: NPISC-specific analysis scripts and outputs
* Top-level `*.py` scripts: integration, preprocessing, benchmarking, and
  downstream analysis pipelines
* `*.h5ad`: intermediate and final AnnData objects

## Authors

* Yishen Miao
* Matthew Kourakis
* Abhinav Bharat
* Isabela Machado
* Matthew Tang
* William C. Smith

## License

* [GNU General Public License v3.0](http://www.gnu.org/licenses/gpl-3.0.html)
