library(tidyverse)

dt <- read_tsv("diff_expr.full.tsv") %>%
mutate(
    comp = factor(
        comp,
        levels = c(
            "iniG-midG",
            "midG-earN",
            "earN-latN",
            "latN-iniTI",
            "iniTI-earTI",
            "earTI-midTII",
            "midTII-latTI",
            "latTI-latTII",
            "latTII-larva"
        )
    )
) %>%
drop_na() %>%
mutate(
    p_threshold = ifelse(
        fdr < 0.05,
        "FDR < 0.05",
        "FDR ≥ 0.05"
    ),
    fc_threshold = ifelse(
        abs(coef) > 1.5,
        "|logFC| > 1.5",
        "|logFC| ≤ 1.5"
    ),
    threshold = interaction(p_threshold, fc_threshold, sep = ", ")
)

ggplot(dt) +
geom_point(aes(x = coef, y = -log10(fdr), color = threshold)) +
facet_grid(comp ~ tissue_type) +
xlab("log2 Fold Change") +
ylab("-log10 FDR") +
theme(
    axis.text.x = element_text(angle = 90, hjust = 1),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin()
) +
guides(col = guide_legend(nrow = 2, byrow = TRUE))

ggsave("volcano.png", height = 8.5,  width = 7)
