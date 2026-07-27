import warnings

warnings.filterwarnings("ignore")

import os
import torch
import random
import argparse
import matplotlib
import numpy as np
import scanpy as sc
import seaborn as sns
from model import MetaQ
from matplotlib import rcParams
from alive_progress import alive_bar
from engine import train_one_epoch, warm_one_epoch, inference
from data_utils import load_data, compute_metacell
from eval_utils import (
    plot_metacell_umap,
    plot_metacell_size,
    plot_celltype_purity,
    plot_compactness_separation,
)


def main(args):
    device = torch.device(args.device)

    adata_list, dataloader_train, dataloader_eval, input_dims = load_data(args)
    omics_num = len(adata_list)
    streaming = getattr(dataloader_train, "backed", False)
    skip_pairwise_metrics = getattr(args, "skip_pairwise_metrics", False)
    skip_visualization = getattr(args, "skip_visualization", False)
    if streaming and not skip_pairwise_metrics:
        raise ValueError(
            "Zarr streaming requires --skip_pairwise_metrics because the "
            "pairwise correlation matrix is not out-of-core"
        )

    print("Target metacell number:", args.metacell_num)

    net = MetaQ(
        input_dims=input_dims,
        data_types=args.data_type,
        entry_num=args.metacell_num,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    print("======= Training Start =======")

    with alive_bar(args.train_epoch, enrich_print=False) as bar:
        loss_rec_his = loss_vq_his = 1e7
        stable_epochs = 0
        if args.codebook_init == "Random":
            warm_epochs = 0
        else:
            # For Kmeans and Geometric initialization
            warm_epochs = min(50, int(args.train_epoch * 0.2))
        for epoch in range(args.train_epoch):
            bar()
            if epoch < warm_epochs:
                warm_one_epoch(
                    model=net,
                    data_types=args.data_type,
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
            elif epoch == warm_epochs:
                embeds, ids, _, _, _ = inference(
                    model=net,
                    data_types=args.data_type,
                    data_loader=dataloader_eval,
                    device=device,
                )
                net.quantizer.init_codebook(embeds, method=args.codebook_init)
                if omics_num == 1:
                    net.copy_decoder_q()
            else:
                loss_rec, loss_vq = train_one_epoch(
                    model=net,
                    data_types=args.data_type,
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
                converge = (abs(loss_vq_his - loss_vq) <= 1e-5) and (
                    abs(loss_rec_his - loss_rec) <= 1e-5
                )
                if converge:
                    stable_epochs += 1
                    if stable_epochs >= args.converge_threshold:
                        print("Early Stopping.")
                        break
                else:
                    stable_epochs = 0
                    loss_rec_his = loss_rec
                    loss_vq_his = loss_vq

    print("======= Training Done =======")

    print("")

    print("======= Inference Start =======")
    embeds, ids, delta_confs, rec_q_percent, loss_codebook = inference(
        model=net, data_types=args.data_type, data_loader=dataloader_eval, device=device
    )

    print("* Quantized Reconstruction Percent:", rec_q_percent)
    print("* Metacell Delta Assignment Confidence:", np.mean(delta_confs))
    print("* Codebook Loss:", loss_codebook)
    print("")

    if not os.path.exists("./save/"):
        os.makedirs("./save/")

    for i in range(omics_num):
        metacell_path = (
            "./save/"
            + args.save_name
            + "_"
            + args.data_type[i]
            + "_"
            + str(args.metacell_num)
            + "metacell.h5ad"
        )
        adata = adata_list[i]
        metacell_adata = compute_metacell(adata, ids, args)
        metacell_adata.write_h5ad(metacell_path)
        print(args.data_type[i] + " metacell data saved at:", metacell_path)

    assignment_path = (
        "./save/" + args.save_name + "_" + str(args.metacell_num) + "metacell_ids.h5ad"
    )
    adata = sc.AnnData(embeds, dtype=np.float32)
    adata.obs["metacell"] = ids
    if args.type_key in adata_list[0].obs_keys():
        if streaming:
            adata.obs[args.type_key] = np.asarray(
                adata_list[0].get_obs_column(args.type_key)
            )
        else:
            adata.obs[args.type_key] = adata_list[0].obs[args.type_key]
    if not skip_visualization:
        sc.set_figure_params(figsize=(7, 7), dpi=300)
        sc.pp.neighbors(adata, use_rep="X", metric="cosine")
        sc.tl.umap(adata)
        if args.type_key in adata.obs_keys():
            sc.pl.umap(
                adata,
                color=[args.type_key],
                save="_" + args.save_name + "_embedding.png",
                palette=sns.color_palette(
                    "husl", np.unique(adata.obs[args.type_key].values).size
                ),
                show=False,
            )
        rcParams.update(matplotlib.rcParamsDefault)
    adata.write_h5ad(assignment_path)
    print("Metacell assignment saved at:", assignment_path)
    print("")

    if not skip_visualization:
        fig_save_name = args.save_name + "_" + str(args.metacell_num) + "metacell"
        plot_metacell_umap(adata, fig_save_name)
        plot_metacell_size(adata, fig_save_name)
        if args.type_key in adata.obs_keys():
            plot_celltype_purity(adata, adata.obs[args.type_key], fig_save_name)
        if not skip_pairwise_metrics:
            for i in range(omics_num):
                plot_compactness_separation(
                    dataloader_train.dataset.raw_list[i].numpy(),
                    adata,
                    fig_save_name + "_" + args.data_type[i],
                    omics_num > 1,
                )

    print("======= Inference Done =======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data configs (User defined)
    parser.add_argument("--data_path", nargs="+", type=str)
    parser.add_argument(
        "--data_type", nargs="+", type=str, choices=["RNA", "ADT", "ATAC"]
    )
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--metacell_num", type=int)
    parser.add_argument("--type_key", type=str, default="celltype")

    # Training configs (Recommend to use the default)
    parser.add_argument(
        "--codebook_init",
        type=str,
        choices=["Random", "Kmeans", "Geometric"],
        default="Random",
    )
    parser.add_argument("--train_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_pairwise_metrics", action="store_true")
    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--converge_threshold", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    assert len(args.data_path) == len(
        args.data_type
    ), "Number of data path and data type mismatch"

    # Randomization
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.random_seed)

    main(args)
