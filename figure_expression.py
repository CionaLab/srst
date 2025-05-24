# %%
import re

import pandas as pd
import scanpy as sc
import networkx as nx
import seaborn as sns
import scvi
import torch

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

from npisc.build_matrix import (
    adjacent,
    split_adata,
    map_cells,
    plot_np,
    make_stage_name,
    make_digraph,
    CrossStage,
)


torch.set_float32_matmul_precision("high")
scvi.settings.dl_num_workers = 63
scvi.settings.seed = 0


PREFIX = "integrated"
GENOME = "ky21"

PATTERN_STAGES = r"(early|mid|late) (gastrula|neurula)"
PATTERN_CELLS = r"[Aa]\d+\.\d+$"

STAGES_IN_SITU = [
    ("mid gastrula", "mid_gastrula.geojson", "midG"),
    ("late gastrula", "late_gastrula.geojson", "latG"),
    ("early neurula", "early_neurula.geojson", "earN"),
    ("mid neurula", "mid_neurula.geojson", "midN"),
    ("late neurula", "late_neurula.geojson", "latN"),
]

COUNTS_LAYER = "counts"
SIZE_FACTOR = "size_factor"
BATCH_KEY = "sample"
SCVI_LATENT_KEY = "X_scVI"
SCVI_BASIS = "scVI_basis"
SCVI_EXPRESSION_KEY = "scVI_normalized"
SCVI_LOG1P_KEY = "scVI_log1p"

d_patterns = {
    k: pd.read_csv(
        f"{PREFIX}_{GENOME}_npisc_{k}_pattern.csv",
        index_col=0,
    )
    for _, _, k in STAGES_IN_SITU
}

gdfs = {k: gpd.read_file(f"npisc/{file}") for _, file, k in STAGES_IN_SITU}

for k in gdfs:
    gdfs[k]["tmp_name"] = gdfs[k]["name"].str.replace(
        "*",
        "",
        regex=False,
    )

# %%
df_exp = pd.read_csv("expression.csv")


GENES = [
    "NKX2.6",
    "LMX1r",
    "LMX1",
    "PAX6",
]

for gene in GENES:
    filtered_df = df_exp[df_exp["gene"].isin([gene, "OTX"])]
    for k, d in filtered_df.groupby("stage"):
        d_tmp = gdfs[k].merge(
            d,
            how="left",
            left_on="tmp_name",
            right_on="blastomere",
        )

        # Mark coexpression status
        def coexp(row):
            if row["gene"] == gene and (
                d_tmp[
                    (d_tmp["tmp_name"] == row["tmp_name"]) & (d_tmp["gene"] == "OTX")
                ].shape[0]
                > 0
            ):
                return "coexpress"
            elif row["gene"] == gene:
                return "gene"
            elif row["gene"] == "OTX":
                return "otx"
            else:
                return "missing"

        d_tmp["coexpression"] = d_tmp.apply(coexp, axis=1)

        # For each blastomere, keep only one row: prefer coexpress > gene > otx > missing
        d_tmp = d_tmp.sort_values(
            by=["coexpression"],
            key=lambda x: x.map({"coexpress": 3, "gene": 2, "otx": 1, "missing": 0}),
            ascending=False,
        )

        color_map = {
            "coexpress": "purple",
            "gene": "red",
            "otx": "lightblue",
            "missing": "grey",
        }

        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
        d_tmp.plot(
            ax=ax,
            color=d_tmp["coexpression"].map(color_map),
            edgecolor="grey",
            linewidth=0.8,
            alpha=0.6,
        )

        # Custom legend
        legend_patches = [
            mpatches.Patch(color="purple", label="Coexpress"),
            mpatches.Patch(color="red", label=gene),
            mpatches.Patch(color="lightblue", label="OTX"),
            mpatches.Patch(color="grey", label="Missing"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8, frameon=True)

        # Annotate names
        d_tmp.apply(
            lambda x: ax.annotate(
                text=f"{x['name']}",
                xy=x.geometry.centroid.coords[0],
                ha="center",
                color="black",
                fontsize=6,
                path_effects=[
                    pe.withStroke(linewidth=2, foreground="white"),
                ],
            ),
            axis=1,
        )
        ax.axis("off")
        ax.set_title(f"{k} - {gene}")
        fig.tight_layout()
        fig.savefig(f"in_situ_cartoon_{k}_{gene}.png")

# %%
