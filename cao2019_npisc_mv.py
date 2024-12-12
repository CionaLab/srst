import re

import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt


def make_marker(
    adata: sc.AnnData,
    d_pattern: pd.DataFrame,
    df_de: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify and sort marker genes based on differential expression analysis.

    :param adata: Annotated data matrix.
    :type adata: sc.AnnData
    :param d_pattern: DataFrame containing gene patterns.
    :type d_pattern: pd.DataFrame
    :param df_de: DataFrame containing differential expression results.
    :type df_de: pd.DataFrame

    :return: DataFrame with sorted marker genes.
    :rtype: pd.DataFrame
    """

    m = adata.var_names.intersection(d_pattern.columns)

    m_sorted = (
        df_de[
            df_de["qseqid"].isin(m) & df_de["is_de_fdr_0.05"] & (df_de["lfc_mean"] > 1)
        ]
        .groupby("qseqid")
        .size()
        .sort_values(ascending=False)
        .index
    )

    m_sorted = adata.var.loc[m_sorted]["gene_name"].reset_index()
    return m_sorted


PATTERN_STAGES = r"(early|mid|late) (gastrula|neurula)"
PATTERN_CELLS = r"[Aa]\d+\.\d+$"

STAGES_IN_SITU = [
    ("mid gastrula", "mid_gastrula.geojson", "midG"),
    ("late gastrula", "late_gastrula.geojson", "latG"),
    ("early neurula", "early_neurula.geojson", "earN"),
    ("mid neurula", "mid_neurula.geojson", "midN"),
    ("late neurula", "late_neurula.geojson", "latN"),
]

df = pd.read_csv("npisc/pass_02.tsv", sep="\t")
df_map = pd.read_csv("npisc/kh2012_ky2021_map.tsv", sep="\t")
df_map["query"] = "KH2012:" + df_map["query"]
df["Gene"] = df["Gene"].apply(lambda x: x.split(" ")[0])
df = pd.merge(df, df_map, how="left", left_on="Gene", right_on="query")
df = df.drop(columns=["Gene", "query"])
df = df.rename(columns={"subject": "Gene"})

df["Stage"] = df["Stage"].apply(lambda x: (" ".join(re.findall(PATTERN_STAGES, x)[0])))
df = df[df["Territory_eq"].str.contains(PATTERN_CELLS)]
df = df[["Stage", "Gene", "Territory_eq"]].drop_duplicates()

dfs = dict(tuple(df.groupby("Stage")))

df_counts = {
    stage: df.groupby("Territory_eq")["Gene"]
    .nunique()
    .reset_index()
    .rename(columns={"Gene": "n"})
    for stage, df in dfs.items()
}

d_patterns = {
    stage: pd.pivot_table(
        df,
        values="Gene",
        index="Territory_eq",
        columns="Gene",
        aggfunc="size",
        fill_value=0,
    ).astype(bool)
    for stage, df in dfs.items()
}

SCVI_LATENT_KEY = "X_scVI"
SCVI_BASIS = "scVI_basis"
SCVI_MDE_KEY = "X_scVI_MDE"
SCVI_EXPRESSION_KEY = "scVI_normalized"
SCVI_LOG1P_KEY = "scVI_log1p"

STAGES_MAPPING = [
    ("mid gastrula", "midG"),
    ("early neurula", "earN"),
    ("late neurula", "latN"),
]

adata = sc.read_h5ad("cao2019_ky21.h5ad")

adatas = {s: sc.read_h5ad(f"cao2019_npisc_ky21_{s}.h5ad") for _, s in STAGES_MAPPING}

adatas_sbc = {
    s: sc.read_h5ad(f"cao2019_npisc_ky21_np_{s}.h5ad") for _, s in STAGES_MAPPING
}

df_des = {
    k: pd.read_csv(f"cao2019_npisc_ky21_{k}_de.csv", index_col=0)
    for _, k in STAGES_MAPPING
}

for k1, k2 in STAGES_MAPPING:

    markers_sorted = make_marker(
        adatas[k2],
        d_patterns[k1],
        df_des[k2],
    )

    markers_sorted.to_csv(
        f"cao2019_npisc_ky21_{k2}_markers.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6.5, 9), dpi=300)
    vp = sc.pl.stacked_violin(
        adatas[k2],
        markers_sorted["qseqid"],
        groupby="leiden",
        layer=SCVI_LOG1P_KEY,
        ax=ax,
        return_fig=True,
    )
    d_ax = vp.get_axes()
    d_ax["mainplot_ax"].set_xticklabels(
        markers_sorted["gene_name"],
    )

    fig.tight_layout()
    fig.savefig(f"cao2019_npisc_ky21_{k2}_markers.png")

df_des_np = {
    k: pd.read_csv(f"cao2019_npisc_ky21_np_{k}_de.csv", index_col=0)
    for _, k in STAGES_MAPPING
}

for k1, k2 in STAGES_MAPPING:

    markers_sorted = make_marker(
        adatas_sbc[k2],
        d_patterns[k1],
        df_des_np[k2],
    )

    markers_sorted.to_csv(
        f"cao2019_npisc_ky21_np_{k2}_markers.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6.5, 9), dpi=300)
    vp = sc.pl.stacked_violin(
        adatas_sbc[k2],
        markers_sorted["qseqid"],
        groupby="leiden",
        layer=SCVI_LOG1P_KEY,
        ax=ax,
        return_fig=True,
    )
    d_ax = vp.get_axes()
    d_ax["mainplot_ax"].set_xticklabels(
        markers_sorted["gene_name"],
    )

    fig.tight_layout()
    fig.savefig(f"cao2019_npisc_ky21_np_{k2}_markers.png")
