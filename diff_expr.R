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
rename_all(~sub(" ", "_", .)) %>%
separate(name, c("stage", "barcode"), "_") %>%
separate(stage, c("stage", "replica"), "\\.") %>%
mutate(
    stage = factor(
        stage,
        levels = c(
            "C110",
            "midG",
            "earlyN",
            "lateN",
            "ITB",
            "ETB",
            "MTB",
            "LTB1",
            "LTB2",
            "lv"
        )
    )
)

mat_exprs <- FromMatrix(as_matrix(exprs), meta_cell)

options(future.globals.maxSize =  16 * 1024 * 1024 * 1024)

mat_subsets <- split(mat_exprs, "tissue_type") %>%
future_map(~split(.x, "stage"))

comparisons <- future_map2(
    head(names(stages), -1),
    tail(names(stages), -1),
    ~c(.x, .y)
)

mat_subsets <- future_map(
    mat_subsets,
    function(subset) future_map(
        comparisons,
        ~future_map(.x, ~subset[stages[[.x]]])
    ) %>%
    future_map(~future_pmap(.x, cbind)) %>%
    unlist()
)

# reset memory limit to avoid running out of memory.
options(future.globals.maxSize = 500 * 1024 * 1024)

zlm_output <- map(mat_subsets, ~map(.x, ~zlm(~stage, .x)))

sumr <- map(zlm_output, ~map(.x, ~summary(.x, doLRT = TRUE)))

fc <- map(
    sumr,
    ~map(
        .x,
        ~.x$datatable[contrast != "(Intercept)"]
    ) %>%
    map(
        ~merge(
            .x[component == "H", .(primerid, `Pr(>Chisq)`)],
            .x[component == "logFC", .(primerid, coef, ci.hi, ci.lo)],
            by = "primerid"
        )
    ) %>%
    map(
        ~.x[, fdr := p.adjust(`Pr(>Chisq)`, "fdr")]
    )
)

fc_filtered <- map(
    fc,
    ~map(
        .x,
        ~as_tibble(.x) %>%
        filter(fdr < 0.05 & abs(coef) > log2(1.5)) %>%
        rename(`Pr(>Chisq)` = "chisq") %>%
        arrange(fdr, abs(coef))
    )
)

tissue_type <- names(fc_filtered) %>%
map(~sub("\\W+", "_", .x)) %>%
unlist()

map(
    fc_filtered,
    ~map2(
        .x,
        map(
            comparisons,
            ~paste(.x, collapse = "-")
        ),
        ~add_column(.x, comp = .y)
    ) %>%
    bind_rows()
) %>%
map2(
    tissue_type,
    ~add_column(.x, tissue_type = .y)
) %>%
bind_rows() %>%
write_tsv("diff_expr.tsv")
