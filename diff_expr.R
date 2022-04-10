library(monocle3)
library(tidyverse)
library(egg)

as_matrix <- function(x) {
    if (!tibble::is_tibble(x)) stop("x must be a tibble")
    y <- as.matrix.data.frame(x[, -1])
    rownames(y) <- x[[1]]
    y
}

plot_filename <- function(
    ...,
    date = format(Sys.time(), "%Y-%m-%d"),
    format = "png"
) {
    paste(
        c(..., date, format),
        sep = ".",
        collapse = "."
    )
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

cds_subset <- cds[, meta_cell %>% pull(stage) == "larva"] %>%
preprocess_cds(., num_dim = 100) %>%
align_cds(., alignment_group = "replica") %>%
reduce_dimension(
    .,
    umap.n_neighbors = 100
) %>%
cluster_cells(.) %>%
learn_graph(., use_partition = FALSE)

list(
    list(
        cds_subset,
        group_cells_by = "cluster",
        label_cell_groups = TRUE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subset,
        color_cells_by = "tissue_type",
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subset,
        color_cells_by = "replica",
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_larva"),
    height = 21
)

list(
    list(
        cds_subset,
        genes = c("KH2012:KH.C14.377"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subset,
        genes = c("KH2012:KH.L96.86"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subset,
        genes = c("KH2012:KH.C10.165"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subset,
        genes = c("KH2012:KH.C10.454"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_larva", "genes"),
    height = 14,
    width = 14
)

list(
    list(
        cds_subset,
        genes = c("KH2012:KH.C14.377"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subset,
        genes = c("KH2012:KH.L132.17"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_larva", "hox"),
    height = 10,
    width = 7
)

pr_test_res <- graph_test(cds_subset, neighbor_graph = "principal_graph", cores = 64)
pr_deg_ids <- row.names(subset(pr_test_res, q_value < 0.05))

gene_module_df <- find_gene_modules(
    cds_subset[pr_deg_ids, ],
    resolution = c(10 ^ seq(-6, -1))
)

cell_group_df <- tibble::tibble(
    cell = row.names(colData(cds_subset)),
    cell_group = partitions(cds_subset)[colnames(cds_subset)]
)

agg_mat <- aggregate_gene_expression(
    cds,
    gene_module_df,
    cell_group_df
)
row.names(agg_mat) <- stringr::str_c("Module ", row.names(agg_mat))
colnames(agg_mat) <- stringr::str_c("Partition ", colnames(agg_mat))

marker_test_res <- top_markers(
    cds_subset,
    group_cells_by = "cluster",
    reference_cells = 1000,
    cores = 64
)

top_specific_markers <- marker_test_res %>%
filter(fraction_expressing >= 0.10) %>%
group_by(cell_group) %>%
top_n(10, pseudo_R2)

top_specific_marker_ids <- unique(top_specific_markers %>% pull(gene_id))

plot_genes_by_group(
    cds_subset,
    top_specific_marker_ids,
    ordering_type = "maximal_on_diag"
) %>%
ggsave(
    .,
    filename = plot_filename("cds_larva", "top_gene"),
    width = 14,
    height = 49
)

pheatmap::pheatmap(
    agg_mat,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    scale = "column",
    clustering_method = "ward.D2",
    fontsize = 6,
    filename = plot_filename("cds_1_module_heatmap")
)

cds_subcl <- choose_cells(cds_subset)

cds_subcl <- cds_subcl %>%
preprocess_cds(., num_dim = 100) %>%
align_cds(., alignment_group = "replica") %>%
reduce_dimension(.) %>%
cluster_cells(.) %>%
learn_graph(., use_partition = FALSE)

list(
    list(
        cds_subcl,
        group_cells_by = "cluster",
        label_cell_groups = TRUE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        color_cells_by = "tissue_type",
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        color_cells_by = "replica",
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_neural_sub"),
    height = 14
)

list(
    list(
        cds_subcl,
        genes = c("KH2012:KH.C14.377"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.L96.86"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.C10.165"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.C10.454"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_neural_sub", "genes"),
    height = 10,
    width = 10
)

list(
    list(
        cds_subcl,
        genes = c("KH2012:KH.C14.377"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.L132.17"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_neural_sub", "hox"),
    height = 10,
    width = 7
)

list(
    list(
        cds_subcl,
        genes = c("KH2012:KH.S761.6"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.L22.28"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.S1155.1"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_neural_sub", "gaba"),
    height = 14
)

list(
    list(
        cds_subcl,
        genes = c("KH2012:KH.C3.324"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.C1.1125"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_neural_sub", "glut"),
    height = 10
)

list(
    list(
        cds_subcl,
        genes = c("KH2012:KH.L171.13"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.C11.495"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.C12.337"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    ),
    list(
        cds_subcl,
        genes = c("KH2012:KH.C1.467"),
        label_cell_groups = FALSE,
        label_leaves = FALSE,
        label_root = FALSE,
        label_branch_points = FALSE,
        show_trajectory_graph = FALSE
    )
) %>%
map(~ exec(plot_cells, !!!.)) %>%
exec(ggarrange, !!!.) %>%
ggsave(
    .,
    filename = plot_filename("cds_neural_sub", "opsin"),
    height = 10,
    width = 10
)

marker_test_res <- top_markers(
    cds_subcl,
    reference_cells = 1000,
    cores = 64
)

top_specific_markers <- marker_test_res %>%
filter(fraction_expressing >= 0.10) %>%
group_by(cell_group) %>%
top_n(40, pseudo_R2)

top_specific_marker_ids <- unique(top_specific_markers %>% pull(gene_id))

plot_genes_by_group(
    cds_subcl,
    top_specific_marker_ids,
    max.size = 5
) %>%
ggsave(
    .,
    filename = plot_filename("cds_neural_sub", "top_40"),
    width = 14,
    height = 28
)

subset_pr_test_res <- graph_test(
    cds_subcl,
    neighbor_graph = "principal_graph",
    cores = 64
)

pr_deg_ids <- row.names(subset(subset_pr_test_res, q_value < 0.05))


plot_cells(
    cds,
    genes = gene_module_df %>% filter(module %in% c(17, 16, 30)),
    group_cells_by = "partition",
    color_cells_by = "partition",
    show_trajectory_graph = FALSE
) %>%
ggsave(
    .,
    filename = plot_filename("cds_1_modules"),
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
    filename = plot_filename("cds_1_modules", "top_5"),
    width = 14,
    height = 28
)
