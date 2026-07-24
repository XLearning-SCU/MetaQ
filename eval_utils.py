import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from alive_progress import alive_bar
from sklearn.metrics import balanced_accuracy_score


def pearson_pytorch(A, B):
    if isinstance(A, sparse.csr_matrix) or isinstance(A, sparse.csc_matrix):
        A = A.toarray()
    if isinstance(B, sparse.csr_matrix) or isinstance(B, sparse.csc_matrix):
        B = B.toarray()
    A = torch.from_numpy(A).half()
    B = torch.from_numpy(B).half()
    if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()

    A_mean = A - A.mean(dim=1, keepdim=True)
    B_mean = B - B.mean(dim=1, keepdim=True)
    A_std = A_mean.norm(dim=1)
    B_std = B_mean.norm(dim=1)
    cov_matrix = torch.mm(A_mean, B_mean.t())
    correlation_matrix = cov_matrix / torch.outer(A_std, B_std)

    return correlation_matrix.cpu().numpy()


def pairwise_correlation(data, save_name):
    cell_num = data.shape[0]
    save_path = "./save/" + save_name + "_" + str(cell_num) + "cells_correlation.npy"
    try:
        corr = np.load(save_path)
        print("Correlation Loaded from:", save_path)
    except:
        print("Computing Pairwise Correlation...")
        corr = np.zeros((cell_num, cell_num), dtype=np.float16)
        chunk_size = 5000
        chunks_num = cell_num // chunk_size + 1 if cell_num % chunk_size != 0 else 0
        total_steps = chunks_num * chunks_num
        with alive_bar(total_steps, enrich_print=False) as bar:
            for i in range(0, cell_num, chunk_size):
                row_start, row_end = i, min(i + chunk_size, cell_num)
                for j in range(0, cell_num, chunk_size):
                    col_start, col_end = j, min(j + chunk_size, cell_num)
                    corr[row_start:row_end, col_start:col_end] = pearson_pytorch(
                        data[row_start:row_end], data[col_start:col_end]
                    )
                    bar()
        np.save(save_path, corr)
        print("Correlation Saved to:", save_path)
    return corr


def compactness(corr, metacell_assign):
    cell_num = corr.shape[0]
    metacell_num = len(np.unique(metacell_assign))
    assignment_ids = np.unique(metacell_assign)
    compactness = []
    for i in assignment_ids:
        idx = np.where(metacell_assign == i)[0]
        if len(idx) == 0:
            continue
        compactness.append(
            np.mean(corr[idx][:, idx]) * len(idx) / cell_num * metacell_num
        )
    return compactness


def separation(corr, metacell_assign):
    cell_num = corr.shape[0]
    metacell_num = len(np.unique(metacell_assign))
    assignment_ids = np.unique(metacell_assign)
    separation = []
    for i in assignment_ids:
        idx = np.where(metacell_assign == i)[0]
        complementary_idx = np.where(metacell_assign != i)[0]
        if len(idx) == 0:
            continue
        separation.append(
            np.mean(1 - corr[idx][:, complementary_idx].max(axis=1))
            * len(idx)
            / cell_num
            * metacell_num
        )
    return separation


def plot_compactness_separation(
    raw_count, assignment_adata, save_name, multi_omics=False
):
    raw_adata = sc.AnnData(raw_count)
    sc.pp.normalize_total(raw_adata, target_sum=1e4)
    sc.pp.log1p(raw_adata)

    if not multi_omics:
        corr_save_name = save_name.split("_")[0]
    else:
        corr_save_name = save_name.split("_")[0] + "_" + save_name.split("_")[-1]
    corr = pairwise_correlation(raw_adata.X, corr_save_name)
    assignments = assignment_adata.obs["metacell"]

    compactness_scores = compactness(corr, assignments)
    separation_scores = separation(corr, assignments)

    np.savetxt("./save/" + save_name + "_compactness.txt", np.array(compactness_scores))
    np.savetxt("./save/" + save_name + "_separation.txt", np.array(separation_scores))

    print("* Average Compactness Score:", np.mean(compactness_scores))
    print("* Average Separation Score:", np.mean(separation_scores))

    data = pd.DataFrame(columns=["Metric", "Score"])
    for score in compactness_scores:
        data = pd.concat(
            [data, pd.DataFrame({"Metric": ["Compactness"], "Score": [score]})]
        )
    for score in separation_scores:
        data = pd.concat(
            [data, pd.DataFrame({"Metric": ["Separation"], "Score": [score]})]
        )

    plt.figure(figsize=(4, 7), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    sns.boxplot(
        data=data,
        y="Score",
        x="Metric",
        saturation=0.55,
        fliersize=0.5,
        linewidth=0.5,
        width=0.87,
    )

    plt.title("Metacell Metric")
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_metric.png", transparent=False)
    plt.close()


def plot_metacell_umap(adata, save_name, meta_size=10, cell_size=0.5):
    umap = (
        pd.DataFrame(adata.obsm["X_umap"])
        .set_index(adata.obs_names)
        .join(adata.obs["metacell"])
    )
    umap["metacell"] = umap["metacell"].astype("category")
    mcs = umap.groupby("metacell").mean().reset_index()

    plt.figure(figsize=(7, 7), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    sns.scatterplot(x=0, y=1, hue="metacell", data=umap, s=cell_size, legend=None)
    sns.scatterplot(
        x=0,
        y=1,
        s=meta_size,
        hue="metacell",
        data=mcs,
        edgecolor="black",
        linewidth=1,
        legend=None,
    )

    plt.xlabel(f"UMAP1")
    plt.ylabel(f"UMAP2")
    plt.title("Metacell Assignment")
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_umap.png", transparent=False)
    plt.close()


def plot_metacell_size(adata, save_name, bins=50):
    label_df = adata.obs["metacell"].reset_index()
    count = label_df.groupby("metacell").count().iloc[:, 0]

    plt.figure(figsize=(7, 3), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    sns.histplot(data=count, stat="density", bins=bins, kde=True)

    plt.xlabel("Number of Cells per Metacell")
    plt.title("Metacell Size")
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_size.png", transparent=False)
    plt.close()


def plot_celltype_purity(adata, annotations, save_name):
    annotations = pd.Series(annotations, index=adata.obs_names, dtype=object)
    type_counts = annotations.value_counts(sort=False)
    celltypes = type_counts.index.to_numpy()
    class_size = type_counts.to_numpy()

    assignments = adata.obs["metacell"]
    assignment_ids = np.unique(assignments)

    majority_type_pred = annotations.copy()
    for i in assignment_ids:
        idx = assignments == i
        majority_type = annotations[idx].value_counts().idxmax()
        majority_type_pred[idx] = majority_type

    label_codes, _ = pd.factorize(
        pd.concat([annotations, majority_type_pred], ignore_index=True)
    )
    annotations_int = label_codes[: len(annotations)]
    majority_type_pred_int = label_codes[len(annotations) :]
    balanced_acc = balanced_accuracy_score(annotations_int, majority_type_pred_int)
    print("* Balanced Cell Type Purity:", balanced_acc)

    sorted_indices = np.argsort(class_size)[::-1]
    sorted_celltypes = celltypes[sorted_indices]

    rows = []
    for type in celltypes:
        type_idx = annotations == type
        preds = majority_type_pred[type_idx]
        for type_ in celltypes:
            total_number = (preds == type_).sum()
            frac = total_number / type_idx.sum()
            rows.append(
                {
                    "Cell Type": type,
                    "Prediction by Majority": type_,
                    "Fraction": frac,
                }
            )
    data = pd.DataFrame(rows)
    data["Cell Type"] = pd.CategoricalIndex(
        data["Cell Type"], categories=sorted_celltypes
    )
    data.sort_values("Cell Type")
    data["Prediction by Majority"] = pd.CategoricalIndex(
        data["Prediction by Majority"],
        categories=sorted_celltypes,
    )
    data.sort_values("Prediction by Majority")
    data = data.pivot(
        index="Cell Type", columns="Prediction by Majority", values="Fraction"
    )
    data = data.fillna(0)

    plt.figure(figsize=(7.3, 7), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    ax = sns.heatmap(
        data=data,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette("Reds", as_cmap=True),
        cbar=False,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

    plt.title("Balanced Cell Type Purity: {:.2f}%".format(balanced_acc * 100))
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_purity_heatmap.png", transparent=False)
    plt.close()

    purities = []
    rows = []
    for i in assignment_ids:
        idx = assignments == i
        majority_type = annotations[idx].value_counts().idxmax()
        purity = np.sum(annotations[idx] == majority_type) / len(annotations[idx])
        purities.append(purity)
        rows.append(
            {
                "Cell Type": majority_type,
                "Metacell Purity": purity,
            }
        )
    data = pd.DataFrame(rows)
    avg_purity = data["Metacell Purity"].mean()
    print("* Average Cell Type Purity:", avg_purity)
    np.savetxt("./save/" + save_name + "_purity.txt", np.array(purities))

    plt.figure(figsize=(7, 4), dpi=300)

    ax = sns.boxplot(
        data=data,
        x="Cell Type",
        y="Metacell Purity",
        saturation=0.55,
        fliersize=0.5,
        linewidth=0.5,
        width=0.87,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

    plt.title("Metacell Purity")
    plt.tight_layout()

    plt.savefig("./figures/" + save_name + "_purity_box.png", transparent=False)
    plt.close()
