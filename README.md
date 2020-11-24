# Ciona robusta Single Cell RNASeq Analysis Project

This repository hosts files related to the *Ciona robusta* single cell RNASeq
analysis project. The data comes from [Cao
2019](https://doi.org/10.1038/s41586-019-1385-y).

## Usage

### Getting started

1. Install R 4.0.0
2. Install `tidyverse`, `furrr`, and `MAST`.

### Download single cell data

1. Download the following files to `data` directory from the
    [Broad institute single cell portal](https://portals.broadinstitute.org/single_cell/study/SCP454/comprehensive-single-cell-transcriptome-lineages-of-a-proto-vertebrate):

    | File name                           | SHA256                                                             |
    |-------------------------------------|--------------------------------------------------------------------|
    | expression_matrix_10stage.tsv.gz    | `522692396722709d65795df68572224c604521f9219a510daec0aa53b990133c` |
    | ciona10stage.cluster.upload.new.txt | `aefa4ccb2ce436f78b3f788ae051d441834ca9b18a64f82e49d9fafb41e0e97b` |

2. Decompress the file in place if necessary. Remove the extra header line in
    `ciona10stage.cluster.upload.new.txt`.

### Calculating differential expression

1. Run `diff_expr.R`. The script requires a lot amount of RAM and CPU time.
    Adjust `options(future.globals.maxSize)` if memory is running out. It
    writes the result in `diff_expr.tsv`.
