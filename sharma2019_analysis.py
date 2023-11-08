# %%
import scanpy as sc

# %%
adata = sc.read_h5ad("sharma2019_ky21.h5ad")

# %%
cngs = [
    "KY21:KY21.Chr3.483",
    "KY21:KY21.Chr7.267",
    "KY21:KY21.Chr2.475",
    "KY21:KY21.Chr4.987",
]

opsins = ["KY21:KY21.Chr12.157", "KY21:KY21.Chr11.800", "KY21:KY21.Chr9.932"]

sc.pl.dotplot(adata, ["KY21:KY21.Chr2.490", *cngs, *opsins], groupby="leiden")

# %%
sc.pl.umap(adata, color=["leiden", "KY21:KY21.Chr2.490"])

# %%
sc.pl.umap(
    adata, color=["KY21:KY21.Chr3.1172", "KY21:KY21.Chr1.783", "KY21:KY21.Chr2.793"]
)

# %%
gluts = [
    "KY21:KY21.Chr2.734",
    "KY21:KY21.Chr13.442",
    "KY21:KY21.Chr1.1346",
    "KY21:KY21.Chr12.947",
    "KY21:KY21.Chr4.989",
]

vts = ["KY21:KY21.Chr2.793", "KY21:KY21.Chr1.783", "KY21:KY21.Chr3.1172"]

sc.pl.dotplot(adata, [*gluts, *vts], groupby="leiden")

# %%
adata = sc.read_h5ad("sharma2019_ky21_raw.h5ad")
sc.pp.normalize_total(adata, target_sum=1e6)

# %%
genes = [
    f"KY21:{gene}"
    for gene in [
        "KY21.Chr2.490",
        # "KY21.Chr2.793",
        # "KY21.Chr1.783",
        # "KY21.Chr3.1172",
        # "KY21.Chr2.1082"
        "KY21.Chr1.838",
        "KY21.Chr1.563",
        # "KY21.Chr4.768",
        # "KY21.Chr2.793",
        # "KY21.Chr3.483",
        # "KY21.Chr11.539",
    ]
]

# %%

sc.pl.dotplot(adata, genes, groupby="leiden", vmin=0, vmax=4)

# %%
