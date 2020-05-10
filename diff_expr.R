library(tidyverse)
library(furrr)
library(MAST)

as_matrix <- function(x) {
    if (!tibble::is_tibble(x)) stop("x must be a tibble")
    y <- as.matrix.data.frame(x[, -1])
    rownames(y) <- x[[1]]
    y
}


# translate stages in the paper to stages in the data
stages <- list(
    "iniG" = "C110",
    "midG" = "midG",
    "earN" = "earlyN",
    "latN" = "lateN",
    "iniTI" = "ITB",
    "earTI" = "ETB",
    "midTII" = "MTB",
    "latTI" = "LTB1",
    "latTII" = "LTB2",
    "larva" = "lv"
)

exprs <- read_tsv("data/expression_matrix_10stage.tsv")

meta_cell <- read_tsv("data/ciona10stage.cluster.upload.new.txt") %>%
    rename_all(tolower) %>%
    rename_all(~ sub(" ", "_", .)) %>%
    separate(name, c("stage", "barcode"), "_") %>%
    separate(stage, c("stage", "replica"), "\\.")

mat_exprs <- FromMatrix(as_matrix(exprs), meta_cell)

mat_subsets <- split(mat_exprs, "tissue_type") %>%
    future_map(~ split(.x, "stage"))

comparisons <- future_map2(
    head(names(stages), -1),
    tail(names(stages), -1),
    ~ c(.x, .y)
)

options(future.globals.maxSize =  16 * 1024 * 1024 * 1024)

mat_subsets <- future_map(
    mat_subsets,
    function(subset) future_map(
        comparisons,
        ~ future_map(.x, ~ subset[stages[[.x]]])
    ) %>%
        future_map(~ future_pmap(.x, cbind)) %>%
        unlist()
)
