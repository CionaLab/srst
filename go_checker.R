library(tidyverse)
library(gprofiler2)

# nolint start
#if not uploaded run:
# gmt_token <- upload_GMT_file("data/cirobu_kh2012_go.gmt")
# nolint end

gmt_token <- "gp__es2F_ytvD_o34"

dt <- read_tsv("diff_expr.tsv") %>%
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
mutate(downreg = coef  < 0) %>%
unite("group", c(downreg, comp, tissue_type), sep = "\\+") %>%
select(group, primerid) %>%
group_by(group)

results <- split(dt$primerid, dt$group) %>%
map(~gost(., organism = gmt_token, evcodes = TRUE))

results %>%
map(~ .$result) %>%
enframe() %>%
drop_na() %>%
unnest(value) %>%
select(name, p_value, term_id, term_name, intersection) %>%
separate(name, c("downreg", "comp", "tissue_type"), sep = "\\+") %>%
write_tsv("go_results.tsv")
