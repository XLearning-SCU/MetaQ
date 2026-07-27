# MetaQ

[![DOI](https://zenodo.org/badge/897895114.svg)](https://doi.org/10.5281/zenodo.14271479)

This is the implementation of the paper "MetaQ: fast, scalable and accurate metacell inference via deep single-cell quantization". To reduce the file size, we attach the Human Fetal Atlas subset of 25,000 cells with submission. The other datasets used in this work have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1-utA_n5UXQirN9SK5eXAptIYZ6swMCs2?usp=sharing).

## Dependency

Our MetaQ algorithm is implemented in Python, with the following package dependencies.

- torch>=2.1
- numpy=1.26.0
- pandas<3
- alive-progress=3.1.5
- scanpy=1.9.6
- anndata>=0.10
- zarr<3
- numcodecs
- scipy=1.11.3
- scikit-learn=1.1.3
- faiss-gpu=1.7.4 (needed if using Kmeans initialization)
- geosketch=1.2 (needed if using Geometric initialization)

For building the Python environment, we recommend using [Anaconda](https://www.anaconda.com/) to manage packages. After the installation, the conda environment could be built and activated by running

```bash
conda create -n MetaQ python=3.11.6
conda activate MetaQ
```

After creating a conda environment, the above packages could be installed by running

```bash
conda install (package-name)[=package-version]
```

or 

```bash
pip install (package-name)[==package-version]
```

We recommend installing Pytorch following the instructions on [Pytorch's official website](https://pytorch.org/) for different operating systems. With normal network conditions, installation only takes a few minutes.

All our experiments are conducted on an Nvidia RTX 3090 GPU on the Ubuntu 20.04 OS with CUDA 12.2. The code can generally be successfully run as long as the Conda environment is consistent.

### Quick Installation

We have also released MetaQ as a Python package on [PyPI](https://pypi.org/project/MetaQ-sc/). To install MetaQ, run

```bash
pip install MetaQ-sc
```

Note that to correctly install pytorch to enable GPU acceleration, we recommend first prepare the conda environment following the above instructions, and then install MetaQ.

## Usage

### Input Data

MetaQ accepts [AnnData](https://anndata.readthedocs.io/en/latest/index.html) in the `h5ad` or Zarr format as input. To use MetaQ for metacell inference, run

```bash
python MetaQ.py --data_path $data_file_path --data_type $data_type --metacell_num $target_metacell_number --save_name $save_name
```

- `--data_path [str(s)]` corresponds to the input data path for metacell inference, which could be both uni- and multi-omics (by setting `--data_path $data_file_path1 $data_file_path2 ...`).
- `--data_type [str(s)]` refers to the type of the input data, choosing from `["RNA", "ADT", "ATAC"]`. Note the multiple data types need to be provided consistently with the multi-omics input data.
- `--metacell_num [int]` denotes the target number of metacells.
- `--save_name [str]` sets the file name prefix when saving the results.

### For Pip Installation

If you have installed MetaQ via pip, you could alternatively run the following script to perform metacell inference:

```python
from MetaQ_sc import run_metaq

run_metaq(
    data_path=[
        "./data/HumanFetalAtlas_25K.h5ad",
    ],  # the path to the input h5ad data
    data_type=[
        "RNA",
    ],  # the type of the input data
    metacell_num=500,  # the target number of metacells
    save_name="HumanAtlas",  # the file name prefix when saving the results
)

```

### Optional Configs

In addition to the above arguments, which must be provided to perform metacell inference, the following configs could also be adjusted, but we recommend simply leaving them as the default:

- `--type_key [str]` (Default="celltype") the key of cell type annotations in the input data. Note that MetaQ does not require and would not utilize the annotation of the single cell data. If provided, after metacell inference, MetaQ would annotate metacells by the predominant type within each metacell, and compute the metacell purity across different cell types.
- `--codebook_init ["Random", "Kmeans", "Geometric"]` (Default="Random") the codebook initialization strategy, where "Geometric" denotes initializing codebook entries using the [Geometric sketching](https://github.com/brianhie/geosketch) algorithm.
- `--train_epoch [int]` (Default=300) the training epochs.
- `--batch_size [int]` (Default=512) the size of mini-batch.
- `--num_workers [int]` (Default=0) the number of DataLoader worker processes. Increase it when running from a script protected by an `if __name__ == "__main__":` guard.
- `--skip_pairwise_metrics` skips the quadratic-memory compactness and separation evaluation. Use this for large datasets.
- `--skip_visualization` skips the full-cell neighbor graph, UMAP, and plotting. Use this for large datasets.
- `--chunk_size [int]` (Default=5000) the number of consecutive cells processed from a Zarr input at a time.
- `--cache_path [str]` stores the reusable raw-HVG training cache and disk-backed inference arrays for a Zarr input.
- `--converge_threshold [int]` (Default=10) early stop training when the losses are stable for several consecutive epochs.
- `--random_seed [int]` (Default=1) the random seed.

For an atlas-scale Zarr input, use:

```bash
python MetaQ.py \
  --data_path "./data/atlas.zarr" \
  --data_type "RNA" \
  --metacell_num 500 \
  --save_name "Atlas" \
  --cache_path "./cache/Atlas" \
  --chunk_size 5000 \
  --num_workers 0 \
  --skip_pairwise_metrics \
  --skip_visualization
```

The Zarr path keeps the original MetaQ preprocessing and training logic, while
streaming raw counts into a reusable HVG cache. Every cell participates in each
training epoch. Metacells are still computed as the mean of full-gene
library-size-normalized and log-transformed expression, matching the in-memory
path. RAM usage is bounded primarily by the chunk, batch, and final
metacell-by-gene matrix; the reusable cache uses disk space proportional to
cells multiplied by selected features.

### Output

After training, MetaQ would produce the following output files:

- `./save/[save_name]_[data_type]_[metacell_num]metacell.h5ad` the inferred metacells.
- `./save/[save_name]_[metacell_num]metacell_ids.h5ad` the orginal cell embeddings along with the metacell assignments.
- `./save/[save_name]_[metacell_num]metacell_purity.txt` the purity of each metacell.
- `./save/[save_name]_[metacell_num]metacell_[data_type]_compactness.txt` the compactness of each metacell.
- `./save/[save_name]_[metacell_num]metacell_[data_type]_separation.txt` the separation of each metacell.

as well as visualization figures, including:

- `./figures/[save_name]_[metacell_num]_purity_box.png` the metacell purity boxplot.
- `./figures/[save_name]_[metacell_num]_purity_heatmap.png` the metacell purity heatmap.
- `./figures/[save_name]_[metacell_num]metacell_[data_type]_metric.png` the metacell compactness and separation boxplot.
- `./figures/[save_name]_[metacell_num]_size.png` the metacell size distribution.
- `./figures/[save_name]_[metacell_num]_umap.png` the metacell assignments projected onto the 2D umap.
- `./figures/[save_name]_embedding.png` the umap of original cell embeddings.

## Examples

### Human Fetal Atlas

To apply MetaQ on the human fetal atlas dataset, run

```bash
python MetaQ.py --data_path "./data/HumanFetalAtlas_25K.h5ad" --data_type "RNA" --metacell_num 500 --save_name "HumanAtlas"
```

After training, the inferred metacell data would be saved at `./save/HumanAtlas_RNA_500metacell.h5ad`, the original cell embedding along with the metacell assignments would be saved at `./save/HumanAtlas_500metacell_ids.h5ad`. Additionally, the purity, compactness, separation, metacell size, and assignment visualization figures would be saved at `./figures/`. On our machine, running the above command takes about 4 minutes, with the following metacell metrics outputs:

```bash
* Quantized Reconstruction Percent: 0.6569222211837769
* Metacell Delta Assignment Confidence: 0.04337984
* Codebook Loss: 0.3145217578125

* Balanced Cell Type Purity: 0.8821164511014821
* Average Cell Type Purity: 0.9236600659920444

* Average Compactness Score: 0.25925527587890623124
* Average Separation Score: 0.618920166015625
```

Then, we could use the inferred metacell to train a classifier and see its classification performance on the original single-cell data, by running

```bash
python ./downstream_tasks/classification.py --original_path "./data/HumanFetalAtlas_25K.h5ad" --metacell_path "./save/HumanAtlas_RNA_500metacell.h5ad"
```

On our machine, running the above command would output the classification performance:

```bash
* Classification Accuracy: 0.92736
* Balanced Accuracy: 0.8770141876867136
```

### Multi-Omics Human Bone Marrow

To apply MetaQ on the RNA+ADT multi-omics human bone marrow dataset, run

```bash
python MetaQ.py --data_path "./data/CITE_RNA.h5ad" "./data/CITE_ADT.h5ad" --data_type "RNA" "ADT" --metacell_num 613 --save_name "CITE"
```

On our machine, running the above command takes about 7 minutes, with the following metacell metrics outputs:

```bash
* Quantized Reconstruction Percent: 0.8198934197425842
* Metacell Delta Assignment Confidence: 0.010354019
* Codebook Loss: 0.09053823583500424

* Balanced Cell Type Purity: 0.8496414738911595
* Average Cell Type Purity: 0.9291949726120836
```

followed by the compactness and separation scores of RNA and ADT metacells, respectively.

Then, we could perform WNN integration (need to first install the python package `muon=0.1.5`) to integrate the paired multi-omics metacells by running

```bash
python ./downstream_tasks/wnn_integration.py --rna_metacell_path "./save/CITE_RNA_613metacell.h5ad" --adt_metacell_path "./save/CITE_ADT_613metacell.h5ad"
```

The visualization of WNN integration results would be saved to `./figures/X_umap_wnn.png`.

### Multi-Omics Mouse Kidney

To apply MetaQ on the ATAC uni-omics data from the mouse kidney dataset, run

```bash
python MetaQ.py --data_path "./data/MouseKidney_ATAC.h5ad" --data_type "ATAC" --metacell_num 291 --save_name "MouseKidneyATAC"
```

On our machine, running the above command takes about 9 minutes, with the following metacell metrics outputs:

```bash
* Quantized Reconstruction Percent: 0.9826998114585876
* Metacell Delta Assignment Confidence: 0.042781264
* Codebook Loss: 0.12140726575407

* Balanced Cell Type Purity: 0.6816895145716168
* Average Cell Type Purity: 0.7677749314606965

* Average Compactness Score: 0.08769313570407151
* Average Separation Score: 0.7993338844801919
```

To apply MetaQ on the RNA+ATAC multi-omics mouse kidney dataset, run

```bash
python MetaQ.py --data_path "./data/MouseKidney_RNA.h5ad" "./data/MouseKidney_ATAC.h5ad" --data_type "RNA" "ATAC" --metacell_num 291 --save_name "MouseKidney"
```

On our machine, running the above command takes about 10 minutes, with the following metacell metrics outputs:

```bash
* Quantized Reconstruction Percent: 0.9737035036087036
* Metacell Delta Assignment Confidence: 0.041269705
* Codebook Loss: 0.15390669031695292

* Balanced Cell Type Purity: 0.9283162979852139
* Average Cell Type Purity: 0.9312628650606035
```

followed by the compactness and separation scores of RNA and ATAC metacells, respectively.

We could access the metacell quality by computing the correlation between chromatin accessibility and gene expression values, with Signac-identified peak-to-gene correspondences, by running

```bash
python ./downstream_tasks/rna_atac_correlation.py --metacell_id_path "./save/MouseKidney_291metacell_ids.h5ad" --rna_metacell_path "./save/MouseKidney_RNA_291metacell.h5ad" --atac_signac_path "./data/MouseKidney_ATAC_Signac.h5ad"
```

The RNA-ATAC correlation illustration will be saved to `./figures/RNA_ATAC_correlation.png`.

### Human Pancreas

To apply MetaQ on the human pancreas dataset, run

```bash
python MetaQ.py --data_path "./data/HumanPancreas.h5ad" --data_type "RNA" --metacell_num 590 --save_name "HumanPancreas"
```

On our machine, running the above command takes about 2 minutes, with the following metacell metrics outputs:

```bash
* Quantized Reconstruction Percent: 0.9180241227149963
* Metacell Delta Assignment Confidence: 0.03500142
* Codebook Loss: 0.11781787284623231

* Balanced Cell Type Purity: 0.9029837141850394
* Average Cell Type Purity: 0.9787889382704344

* Average Compactness Score: 0.5885199314772804
* Average Separation Score: 0.352471484467584
```

Then, we could apply the Harmony integration method (need to first install the Python package `harmonypy=0.0.6`) on the inferred metacells by running

```bash
python ./downstream_tasks/harmony_integration.py --original_path "data/HumanPancreas.h5ad" --metacell_path "./save/HumanPancreas_RNA_590metacell.h5ad" --metacell_id_path "./save/HumanPancreas_590metacell_ids.h5ad"
```

The integrated metacell would be saved to the original path `./save/HumanPancreas_RNA_590metacell.h5ad` by adding `.obsm["X_pca_harmony"]` to the adata. The umap visualizations of integration results would be saved to `./figures/umap_harmony_type.png` and `./figures/umap_harmony_batch.png`. The code would also compute the cLIST and iLISI scores, with the plot saved to `./figures/lisi_metric.png`.

Furthermore, we could perform clustering on the integrated metacell and map the clustering assignments back to the original cells for evaluation by running

```bash
python ./downstream_tasks/clustering.py --original_path "data/HumanPancreas.h5ad" --metacell_path "./save/HumanPancreas_RNA_590metacell.h5ad" --metacell_id_path "./save/HumanPancreas_590metacell_ids.h5ad"
```

The expected clustering metrics output is:

```bash
* AMI: 0.8854958417656463
* ARI: 0.9311810637109637
* Homogeneity: 0.8780871736172928
```

### Human PBMC Perturbation

The original dataset could be downloaded from the [Kaggle Competition](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data). We infer metacells within cells of the same type, donor, and perturbation. These metacells are then concatenated across different groups to compute differential expression (DE) values.

For simplicity, we provide the inferred metacells on the negative control (Dimethyl Sulfoxide) across all six cell types. As an example, to apply DE analysis on the inferred metacells concerning cell types, run

```bash
python ./downstream_tasks/de_analysis.py --metacell_path "./data/Perturbation_Control.h5ad" --group_key "celltype"
```

The DE value and rank illustrations would be saved to `./figures/de_heatmap_celltype_value.png` and `./figures/de_heatmap_celltype_rank.png`, respectively.

### Human Thyroid Cancer

To apply MetaQ to the human thyroid cancer dataset, run

```bash
python MetaQ.py --data_path "./data/HumanThyroidCancer.h5ad" --data_type "RNA" --metacell_num 462 --save_name "HumanThyroid"
```

On our machine, running the above command takes about 5 minutes, with the following metacell metrics outputs:

```bash
* Quantized Reconstruction Percent: 0.8225199580192566
* Metacell Delta Assignment Confidence: 0.040232606
* Codebook Loss: 0.1452780069797641

* Balanced Cell Type Purity: 0.8823801540855807
* Average Cell Type Purity: 0.9296899189611451

* Average Compactness Score: 0.3887489242066606
* Average Separation Score: 0.4737923321508765
```

To investigate the robustness of MetaQ, we could manually subsample rare cell types and run MetaQ on the subsampled data. We could also try other codebook initialization strategies, such as [Geometric sketching](https://github.com/brianhie/geosketch), by running

```bash
python MetaQ.py --data_path "./data/HumanThyroidCancer.h5ad" --data_type "RNA" --metacell_num 462 --save_name "HumanThyroidGeometric" --codebook_init "Geometric"
```

On our machine, running the above command takes about 5 minutes, with the following metacell metrics outputs:

```bash
* Quantized Reconstruction Percent: 0.822187602519989
* Metacell Delta Assignment Confidence: 0.031774633
* Codebook Loss: 0.14311686708689536

* Balanced Cell Type Purity: 0.8722521142586172
* Average Cell Type Purity: 0.9284227889767889

* Average Compactness Score: 0.38860391138590933
* Average Separation Score: 0.4733415874228979
```

As can be seen, MetaQ achieves similar metacell compactness, separation, and purity with different codebook initialization strategies. Moreover, we could try different target metacell numbers by running

```bash
python MetaQ.py --data_path "./data/HumanThyroidCancer.h5ad" --data_type "RNA" --metacell_num 231 --save_name "HumanThyroid"
```

To measure the consistency between two metacell assignments with different reduction rates, we could compute their homogeneity score by running

```bash
python downstream_tasks/homogeneity_score.py --metacell_id_A_path "./save/HumanThyroid_462metacell_ids.h5ad" --metacell_id_B_path "./save/HumanThyroid_231metacell_ids.h5ad"
```

The expected homogeneity score output is:

```bash
* Homogeneity Score: 0.5262503386631685
```
