library(monocle3)
library(tidyverse)
library(egg)

as_matrix <- function(x) {
    if (!tibble::is_tibble(x)) stop("x must be a tibble")
    y <- as.matrix.data.frame(x[, -1])
    rownames(y) <- x[[1]]
    y
}

exprs <- read_tsv("data/expression_matrix_10stage.tsv") %>%
as_matrix()

meta_cell <- read_tsv("data/ciona10stage.cluster.upload.new.txt") %>%
rename_all(tolower) %>%
rename_all(~sub(" ", "_", .)) %>%
separate(name, c("stage", "barcode"), "_", remove = FALSE) %>%
separate(stage, c("stage", "replica"), "\\.") %>%
mutate(
    stage = factor(stage) %>% recode_factor(
        `C110` = "iniG",
        `midG` = "midG",
        `earlyN` = "earN",
        `lateN` = "latN",
        `ITB` = "iniTI",
        `ETB` = "earTI",
        `MTB` = "midTII",
        `LTB1` = "latTI",
        `LTB2` = "latTII",
        `lv` = "larva"
    ),
    tissue_type = factor(tissue_type) %>% recode(
        `muscle & heart` = "muscle_heart",
        `nervous system` = "nervous_system"
    )
)

meta_gene <- read_csv("data/gene_markers.csv") %>%
select(gene, gene_short_name = anno1) %>%
distinct() %>%
left_join(
    tibble(name = rownames(exprs)),
    .,
    by = c("name" = "gene")
) %>%
mutate(
    gene_short_name = coalesce(gene_short_name, name),
    gene_short_name = sub("KH2012:", "", gene_short_name)
)

cds <- new_cell_data_set(
    exprs,
    cell_metadata = meta_cell %>% column_to_rownames("name"),
    gene_metadata = meta_gene %>% column_to_rownames("name")
) %>%
preprocess_cds(., num_dim = 100) %>%
align_cds(., alignment_group = "stage") %>%
reduce_dimension(
    .,
    umap.n_neighbors = 100
) %>%
cluster_cells(.) %>%
learn_graph(., use_partition = FALSE) %>%
order_cells(
    .,
    root_cells = meta_cell %>%
    filter(stage %in% c("iniG")) %>%
    pull(name)
)

list(
    list(
        cds,
        color_cells_by = "stage",
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds,
        color_cells_by = "pseudotime",
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE
    ),
    list(
        cds,
        color_cells_by = "tissue_type",
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(., filename = "cds_1.2020-11-27.png", height = 21)

pr_test_res <- graph_test(cds, neighbor_graph = "principal_graph", cores = 64)
pr_deg_ids <- row.names(subset(pr_test_res, q_value < 0.05))

gene_module_df <- find_gene_modules(
    cds[pr_deg_ids, ],
    resolution = c(10 ^ seq(-6, -1))
)

cell_group_df <- tibble::tibble(
    cell = row.names(colData(cds)),
    cell_group = partitions(cds)[colnames(cds)]
)

agg_mat <- aggregate_gene_expression(
    cds,
    gene_module_df,
    cell_group_df
)
row.names(agg_mat) <- stringr::str_c("Module ", row.names(agg_mat))
colnames(agg_mat) <- stringr::str_c("Partition ", colnames(agg_mat))

pheatmap::pheatmap(
    agg_mat,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    scale = "column",
    clustering_method = "ward.D2",
    fontsize = 6,
    filename = "cds_1_module_heatmap_2020-11-27.png"
)

plot_cells(
    cds,
    genes = gene_module_df %>% filter(module %in% c(17, 16, 30)),
    group_cells_by = "partition",
    color_cells_by = "partition",
    show_trajectory_graph = FALSE
) %>%
ggsave(
    .,
    filename = "cds_1_modules.2020-11-27.png",
    width = 21
)

marker_test_res <- top_markers(
    cds,
    group_cells_by = "partition",
    reference_cells = 1000,
    cores = 64
)

top_specific_markers <- marker_test_res %>%
filter(fraction_expressing >= 0.10) %>%
group_by(cell_group) %>%
top_n(5, pseudo_R2)

top_specific_marker_ids <- unique(top_specific_markers %>% pull(gene_id))

plot_genes_by_group(
    cds,
    top_specific_marker_ids,
    group_cells_by = "partition",
    ordering_type = "cluster_row_col",
    max.size = 5
) %>%
ggsave(
    .,
    filename = "cds_1_top_5.2020-11-27.png",
    width = 14,
    height = 28
)
