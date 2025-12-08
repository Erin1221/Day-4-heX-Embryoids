import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr

# -------------------------
# 0. Basic configuration
# -------------------------
IN_H5AD = r"c:/Users/erinc/Desktop/D4_merged.h5ad"
OUT_DIR = r"c:/Users/erinc/Desktop/cellrank_results"
FIG_DIR = OUT_DIR  # save all figures in the same directory

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sc.settings.figdir = FIG_DIR
sc.settings.verbosity = 2
cr.settings.verbosity = 2

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    # -------------------------
    # 1. Load AnnData
    # -------------------------
    adata = sc.read_h5ad(IN_H5AD)
    print(">> Loaded AnnData:\n", adata)

    # Ensure seurat_clusters is categorical (CellRank requires categorical clusters)
    if "seurat_clusters" in adata.obs.columns:
        adata.obs["seurat_clusters"] = adata.obs["seurat_clusters"].astype("category")
    else:
        raise ValueError("The key 'seurat_clusters' was not found in adata.obs.")

    # -------------------------
    # 2. Ensure KNN graph exists (connectivities)
    # -------------------------
    if "connectivities" not in adata.obsp.keys():
        print(">> No connectivities found in adata.obsp, recomputing neighbors...")
        # Use existing PCA to compute neighbors
        sc.pp.neighbors(
            adata,
            use_rep="X_pca",
            n_neighbors=30
        )
    print(">> obsp keys after neighbors:", adata.obsp.keys())

    # -------------------------
    # 3. Build ConnectivityKernel + GPCCA estimator
    # -------------------------
    print(">> Building ConnectivityKernel...")
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    print(">> ConnectivityKernel built:", ck)

    g = cr.estimators.GPCCA(ck)
    print(g)

    # -------------------------
    # 4. Compute macrostates and terminal states
    # -------------------------
    # n_states can be adjusted; 7 matches your previous analysis
    print(">> Fitting GPCCA (macrostates)...")
    g.fit(cluster_key="seurat_clusters", n_states=7)

    print(">> Predicting terminal states...")
    g.predict_terminal_states()

    # Plot all macrostates
    g.plot_macrostates(
        which="all",
        basis="umap",
        color="seurat_clusters",
        same_plot=False,
        save=os.path.join(FIG_DIR, "macrostates_all.png")
    )

    # Plot terminal macrostates
    g.plot_macrostates(
        which="terminal",
        basis="umap",
        color="seurat_clusters",
        same_plot=False,
        save=os.path.join(FIG_DIR, "macrostates_terminal.png")
    )

    print(">> Terminal states categories:", g.terminal_states.cat.categories)

    # -------------------------
    # 5. Compute fate probabilities
    # -------------------------
    # Important for Windows: solver='direct' avoids PETSc problems
    print(">> Computing fate probabilities (connectivity kernel only)...")
    g.compute_fate_probabilities(
        solver="direct",
        use_petsc=False,
        show_progress_bar=False
    )

    # Plot lineage probabilities on the UMAP
    g.plot_fate_probabilities(
        same_plot=False,
        basis="umap",
        save=os.path.join(FIG_DIR, "fate_probabilities.png")
    )

    # -------------------------
    # 6. Export fate probabilities to CSV
    # -------------------------
    # CellRank stores lineage probabilities in adata.obsm['lineages_fwd']
    if "lineages_fwd" not in adata.obsm:
        raise KeyError("Missing 'lineages_fwd' in adata.obsm; fate probabilities not successfully computed.")

    lineages_mat = adata.obsm["lineages_fwd"]  # shape: (n_cells, n_lineages)

    # Use terminal_states categories as lineage names
    lineage_names = list(g.terminal_states.cat.categories)
    if lineages_mat.shape[1] != len(lineage_names):
        print("!! Warning: Number of columns in lineages_fwd does not match the number of terminal states.")
        lineage_names = [f"lineage_{i}" for i in range(lineages_mat.shape[1])]

    fate_df = pd.DataFrame(
        lineages_mat,
        index=adata.obs_names,
        columns=lineage_names
    )
    fate_csv = os.path.join(OUT_DIR, "fate_probabilities_connectivity.csv")
    fate_df.to_csv(fate_csv)
    print(">> Saved fate probabilities to:", fate_csv)

    # -------------------------
    # 7. (Optional) Compute lineage drivers
    # -------------------------
    print(">> Computing lineage drivers (optional)...")
    drivers_df = g.compute_lineage_drivers(
        cluster_key="seurat_clusters",
        method="fisher"
    )
    drivers_csv = os.path.join(OUT_DIR, "lineage_drivers_connectivity.csv")
    drivers_df.to_csv(drivers_csv)
    print(">> Saved lineage drivers to:", drivers_csv)

    # -------------------------
    # 8. Save updated AnnData with CellRank results
    # -------------------------
    out_h5ad = os.path.join(OUT_DIR, "D4_merged_cellrank_connectivity.h5ad")
    adata.write(out_h5ad)
    print(">> Wrote updated AnnData to:", out_h5ad)


if __name__ == "__main__":
    main()
