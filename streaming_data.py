import math
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import zarr
from numcodecs import Blosc
from scipy import sparse
from sklearn.utils import sparsefuncs

try:
    from anndata.io import read_elem, sparse_dataset
except ImportError:
    from anndata.experimental import read_elem, sparse_dataset


CACHE_VERSION = 1


def is_zarr_path(path):
    path = Path(path)
    return path.suffix == ".zarr" and path.is_dir()


class ZarrDataSource:
    """Row-oriented access to an AnnData Zarr store without loading X."""

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.root = zarr.open_group(self.path, mode="r")
        if "X" not in self.root:
            raise ValueError(f"AnnData Zarr store has no X matrix: {self.path}")

        x_node = self.root["X"]
        if hasattr(x_node, "shape"):
            self.x = x_node
            self.shape = tuple(x_node.shape)
        else:
            self.x = sparse_dataset(x_node)
            self.shape = tuple(x_node.attrs["shape"])

        self.n_obs, self.n_vars = self.shape
        self._obs_columns = list(self.root["obs"].attrs.get("column-order", []))
        self._obs_cache = {}

    def read_rows(self, start, end):
        block = self.x[start:end]
        if sparse.issparse(block):
            return block.tocsr()
        return np.asarray(block)

    def obs_keys(self):
        return self._obs_columns

    def get_obs_column(self, key):
        if key not in self._obs_columns:
            raise KeyError(key)
        if key in self._obs_cache:
            return self._obs_cache[key]

        node = self.root["obs"][key]
        encoding_type = node.attrs.get("encoding-type", "")
        if encoding_type == "categorical":
            categories = np.asarray(read_elem(node["categories"]))
            codes = np.asarray(node["codes"][:])
            values = pd.Series(
                pd.Categorical.from_codes(codes, categories=categories), name=key
            )
        else:
            values = pd.Series(np.asarray(read_elem(node)), name=key)
        self._obs_cache[key] = values
        return values


def _row_normalize(counts, target_sum=1e4, dtype=np.float32):
    row_sums = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.float64)
    inverse_size_factors = np.zeros_like(row_sums)
    nonzero = row_sums != 0
    inverse_size_factors[nonzero] = target_sum / row_sums[nonzero]

    if sparse.issparse(counts):
        normalized = counts.astype(dtype, copy=True).tocsr()
        sparsefuncs.inplace_row_scale(
            normalized, inverse_size_factors.astype(dtype, copy=False)
        )
    else:
        normalized = np.asarray(counts, dtype=dtype).copy()
        normalized *= inverse_size_factors[:, None]
    return row_sums / target_sum, normalized


def _sum_and_sum_squares(matrix, n_vars):
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        sums = np.bincount(
            matrix.indices, weights=matrix.data, minlength=n_vars
        ).astype(np.float64, copy=False)
        sum_squares = np.bincount(
            matrix.indices, weights=np.square(matrix.data), minlength=n_vars
        ).astype(np.float64, copy=False)
        return sums, sum_squares

    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix.sum(axis=0), np.square(matrix).sum(axis=0)


def _mean_and_variance(sums, sum_squares, n_obs):
    means = sums / n_obs
    if n_obs <= 1:
        return means, np.zeros_like(means)
    variances = (sum_squares - np.square(sums) / n_obs) / (n_obs - 1)
    variances[variances < 0] = 0
    return means, variances


def _highly_variable_mask(
    means,
    variances,
    n_top_genes=None,
    min_disp=0.5,
    max_disp=np.inf,
    min_mean=0.0125,
    max_mean=3,
    n_bins=20,
):
    """Match Scanpy's dispersion-based ``flavor='seurat'`` selection."""

    safe_means = means.copy()
    safe_means[safe_means == 0] = 1e-12
    dispersions = variances / safe_means
    dispersions[dispersions == 0] = np.nan
    dispersions = np.log(dispersions)
    log_means = np.log1p(safe_means)

    frame = pd.DataFrame({"means": log_means, "dispersions": dispersions})
    frame["mean_bin"] = pd.cut(frame["means"], bins=n_bins)
    grouped = frame.groupby("mean_bin", observed=False)["dispersions"]
    dispersion_mean = grouped.mean()
    dispersion_std = grouped.std(ddof=1)

    one_gene_per_bin = dispersion_std.isnull()
    dispersion_std = dispersion_std.copy()
    dispersion_mean = dispersion_mean.copy()
    dispersion_std.loc[one_gene_per_bin] = dispersion_mean.loc[one_gene_per_bin]
    dispersion_mean.loc[one_gene_per_bin] = 0

    bin_values = frame["mean_bin"].to_numpy()
    dispersion_norm = (
        frame["dispersions"].to_numpy()
        - dispersion_mean.loc[bin_values].to_numpy()
    ) / dispersion_std.loc[bin_values].to_numpy()

    if n_top_genes is not None:
        finite_values = dispersion_norm[~np.isnan(dispersion_norm)].copy()
        n_top_genes = min(int(n_top_genes), means.size, finite_values.size)
        if n_top_genes == 0:
            return np.zeros(means.size, dtype=bool)
        finite_values[::-1].sort()
        cutoff = finite_values[n_top_genes - 1]
        return np.nan_to_num(dispersion_norm) >= cutoff

    dispersion_norm = dispersion_norm.copy()
    dispersion_norm[np.isnan(dispersion_norm)] = 0
    return np.logical_and.reduce(
        (
            log_means > min_mean,
            log_means < max_mean,
            dispersion_norm > min_disp,
            dispersion_norm < max_disp,
        )
    )


def _cache_matches(group, source, data_type):
    return (
        group.attrs.get("metaq_cache_version") == CACHE_VERSION
        and group.attrs.get("complete", False)
        and group.attrs.get("source_path") == source.path
        and tuple(group.attrs.get("source_shape", ())) == source.shape
        and group.attrs.get("data_type") == data_type
    )


def build_training_cache(source, data_type, cache_path, chunk_size):
    """Create one on-disk raw-HVG matrix plus preprocessing statistics."""

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(cache_path), mode="a")
    if _cache_matches(group, source, data_type):
        print("Training cache loaded from:", str(cache_path))
        return str(cache_path)

    if len(group) and group.attrs.get("metaq_cache_version") != CACHE_VERSION:
        raise ValueError(
            f"Refusing to overwrite non-MetaQ Zarr store at {cache_path}. "
            "Choose a different --cache_path."
        )
    for key in list(group.keys()):
        del group[key]

    group.attrs.update(
        {
            "metaq_cache_version": CACHE_VERSION,
            "complete": False,
            "source_path": source.path,
            "source_shape": source.shape,
            "data_type": data_type,
        }
    )
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    size_factors = group.create_dataset(
        "size_factor",
        shape=(source.n_obs,),
        chunks=(min(chunk_size, source.n_obs),),
        dtype="f4",
        compressor=compressor,
        overwrite=True,
    )

    print("Computing streaming normalization and HVG statistics:", source.path)
    gene_sums = np.zeros(source.n_vars, dtype=np.float64)
    gene_sum_squares = np.zeros(source.n_vars, dtype=np.float64)
    for start in range(0, source.n_obs, chunk_size):
        end = min(start + chunk_size, source.n_obs)
        counts = source.read_rows(start, end)
        sf, normalized = _row_normalize(counts)
        size_factors[start:end] = sf.astype(np.float32)
        if sparse.issparse(normalized):
            np.log1p(normalized.data, out=normalized.data)
            np.expm1(normalized.data, out=normalized.data)
        else:
            np.log1p(normalized, out=normalized)
            np.expm1(normalized, out=normalized)
        sums, sum_squares = _sum_and_sum_squares(normalized, source.n_vars)
        gene_sums += sums
        gene_sum_squares += sum_squares

    means, variances = _mean_and_variance(
        gene_sums, gene_sum_squares, source.n_obs
    )
    if data_type == "RNA":
        n_top_genes = 3000 if source.n_vars < 5000 else None
        hvg_mask = _highly_variable_mask(
            means, variances, n_top_genes=n_top_genes
        )
    elif data_type == "ATAC":
        hvg_mask = _highly_variable_mask(
            means, variances, n_top_genes=30000
        )
    elif data_type == "ADT":
        hvg_mask = np.ones(source.n_vars, dtype=bool)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    hvg_indices = np.flatnonzero(hvg_mask).astype(np.int64)
    if hvg_indices.size == 0:
        raise ValueError("No highly variable features were selected")
    group.create_dataset(
        "hvg_indices",
        data=hvg_indices,
        compressor=compressor,
        overwrite=True,
    )

    storage_chunk_rows = max(
        1,
        min(
            chunk_size,
            source.n_obs,
            (64 * 1024 * 1024) // (hvg_indices.size * np.dtype("f4").itemsize),
        ),
    )
    raw_hvg_store = group.create_dataset(
        "raw_hvg",
        shape=(source.n_obs, hvg_indices.size),
        chunks=(storage_chunk_rows, hvg_indices.size),
        dtype="f4",
        compressor=compressor,
        overwrite=True,
    )

    print("Writing streaming raw-HVG training cache:", str(cache_path))
    scaled_sums = np.zeros(hvg_indices.size, dtype=np.float64)
    scaled_sum_squares = np.zeros(hvg_indices.size, dtype=np.float64)
    for start in range(0, source.n_obs, chunk_size):
        end = min(start + chunk_size, source.n_obs)
        counts = source.read_rows(start, end)
        raw_hvg = counts[:, hvg_indices]
        if sparse.issparse(raw_hvg):
            raw_hvg = raw_hvg.toarray()
        raw_hvg = np.asarray(raw_hvg, dtype=np.float32)
        raw_hvg_store[start:end] = raw_hvg

        sf = np.asarray(size_factors[start:end], dtype=np.float32)
        inverse_sf = np.zeros_like(sf)
        nonzero = sf != 0
        inverse_sf[nonzero] = 1 / sf[nonzero]
        normalized_log = raw_hvg.copy()
        normalized_log *= inverse_sf[:, None]
        np.log1p(normalized_log, out=normalized_log)
        scaled_sums += normalized_log.sum(axis=0, dtype=np.float64)
        scaled_sum_squares += np.square(normalized_log).sum(
            axis=0, dtype=np.float64
        )

    scale_mean, scale_variance = _mean_and_variance(
        scaled_sums, scaled_sum_squares, source.n_obs
    )
    scale_std = np.sqrt(scale_variance)
    scale_std[scale_std == 0] = 1
    group.create_dataset(
        "scale_mean",
        data=scale_mean.astype(np.float32),
        compressor=compressor,
        overwrite=True,
    )
    group.create_dataset(
        "scale_std",
        data=scale_std.astype(np.float32),
        compressor=compressor,
        overwrite=True,
    )
    group.attrs["complete"] = True
    print(
        str(cache_path),
        "built with shape",
        [source.n_obs, int(hvg_indices.size)],
    )
    return str(cache_path)


class ZarrMetaQDataset:
    def __init__(self, cache_paths, batch_size, chunk_size, random_seed):
        self.cache_paths = list(cache_paths)
        self.batch_size = int(batch_size)
        self.chunk_size = int(chunk_size)
        self.random_seed = int(random_seed)
        if self.batch_size < 2:
            raise ValueError("Zarr streaming requires batch_size >= 2")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")

        groups = [zarr.open_group(path, mode="r") for path in self.cache_paths]
        cell_counts = [group["raw_hvg"].shape[0] for group in groups]
        if len(set(cell_counts)) != 1:
            raise ValueError("Zarr modalities contain different numbers of cells")
        self.cell_num = cell_counts[0]
        if self.cell_num < 2:
            raise ValueError("MetaQ training requires at least two cells")
        self.input_dims = [group["raw_hvg"].shape[1] for group in groups]

        self.effective_batch_size = self.batch_size
        if (
            self.cell_num > self.batch_size
            and self.cell_num % self.effective_batch_size == 1
        ):
            for candidate in range(self.batch_size - 1, 1, -1):
                if self.cell_num % candidate != 1:
                    self.effective_batch_size = candidate
                    break
            else:
                candidate = self.batch_size + 1
                while self.cell_num % candidate == 1:
                    candidate += 1
                self.effective_batch_size = candidate

    def __len__(self):
        return self.cell_num

    @property
    def batch_count(self):
        return math.ceil(self.cell_num / self.effective_batch_size)

    def iter_batches(self, shuffle, epoch):
        groups = [zarr.open_group(path, mode="r") for path in self.cache_paths]
        raw_stores = [group["raw_hvg"] for group in groups]
        sf_stores = [group["size_factor"] for group in groups]
        means = [
            np.asarray(group["scale_mean"][:], dtype=np.float32)
            for group in groups
        ]
        stds = [
            np.asarray(group["scale_std"][:], dtype=np.float32)
            for group in groups
        ]

        starts = np.arange(0, self.cell_num, self.chunk_size)
        rng = np.random.default_rng(self.random_seed + epoch)
        if shuffle:
            rng.shuffle(starts)

        raw_buffers = [None for _ in groups]
        sf_buffers = [None for _ in groups]
        buffered = 0

        for start in starts:
            end = min(int(start) + self.chunk_size, self.cell_num)
            raw_chunks = [
                np.asarray(store[int(start) : end], dtype=np.float32)
                for store in raw_stores
            ]
            sf_chunks = [
                np.asarray(store[int(start) : end], dtype=np.float32)
                for store in sf_stores
            ]
            if shuffle:
                order = rng.permutation(end - int(start))
                raw_chunks = [values[order] for values in raw_chunks]
                sf_chunks = [values[order] for values in sf_chunks]

            if buffered:
                raw_chunks = [
                    np.concatenate([buffer, values], axis=0)
                    for buffer, values in zip(raw_buffers, raw_chunks)
                ]
                sf_chunks = [
                    np.concatenate([buffer, values], axis=0)
                    for buffer, values in zip(sf_buffers, sf_chunks)
                ]

            available = raw_chunks[0].shape[0]
            full_end = (available // self.effective_batch_size) * (
                self.effective_batch_size
            )
            for batch_start in range(0, full_end, self.effective_batch_size):
                batch_end = batch_start + self.effective_batch_size
                yield self._make_batch(
                    raw_chunks,
                    sf_chunks,
                    means,
                    stds,
                    batch_start,
                    batch_end,
                )

            buffered = available - full_end
            if buffered:
                raw_buffers = [values[full_end:] for values in raw_chunks]
                sf_buffers = [values[full_end:] for values in sf_chunks]
            else:
                raw_buffers = [None for _ in groups]
                sf_buffers = [None for _ in groups]

        if buffered:
            yield self._make_batch(
                raw_buffers, sf_buffers, means, stds, 0, buffered
            )

    @staticmethod
    def _make_batch(raw_values, sf_values, means, stds, start, end):
        x_list = []
        sf_list = []
        raw_list = []
        for raw, sf, mean, std in zip(
            raw_values, sf_values, means, stds
        ):
            raw_batch = np.ascontiguousarray(raw[start:end], dtype=np.float32)
            sf_batch = np.ascontiguousarray(
                sf[start:end].reshape(-1, 1), dtype=np.float32
            )
            inverse_sf = np.zeros(sf_batch.shape[0], dtype=np.float32)
            nonzero = sf_batch[:, 0] != 0
            inverse_sf[nonzero] = 1 / sf_batch[nonzero, 0]

            x_batch = raw_batch.copy()
            x_batch *= inverse_sf[:, None]
            np.log1p(x_batch, out=x_batch)
            x_batch -= mean
            x_batch /= std
            x_batch[x_batch > 10] = 10

            x_list.append(torch.from_numpy(x_batch))
            sf_list.append(torch.from_numpy(sf_batch))
            raw_list.append(torch.from_numpy(raw_batch))
        return {"x": x_list, "sf": sf_list, "raw": raw_list}


class ZarrBatchLoader:
    backed = True

    def __init__(self, dataset, shuffle, result_dir):
        self.dataset = dataset
        self.shuffle = shuffle
        self.result_dir = str(result_dir)
        self._epoch = 0

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        return self.dataset.iter_batches(self.shuffle, epoch)

    def __len__(self):
        return self.dataset.batch_count

    def next_inference_prefix(self):
        return os.path.join(self.result_dir, "inference")


def load_zarr_data(args):
    if getattr(args, "num_workers", 0) != 0:
        raise ValueError("Zarr streaming currently requires --num_workers 0")

    print("=======Loading and Preprocessing Zarr Data=======")
    sources = [ZarrDataSource(path) for path in args.data_path]
    cell_counts = [source.n_obs for source in sources]
    if len(set(cell_counts)) != 1:
        raise ValueError("Zarr modalities contain different numbers of cells")

    if args.metacell_num > 1000 and args.batch_size <= 512:
        args.batch_size = 4096

    cache_root = Path(args.cache_path or f"./cache/{args.save_name}")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = []
    for index, (source, data_type) in enumerate(
        zip(sources, args.data_type)
    ):
        cache_paths.append(
            build_training_cache(
                source,
                data_type,
                cache_root / f"modality_{index}_{data_type}.zarr",
                args.chunk_size,
            )
        )

    dataset = ZarrMetaQDataset(
        cache_paths=cache_paths,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        random_seed=args.random_seed,
    )
    train_loader = ZarrBatchLoader(
        dataset, shuffle=True, result_dir=cache_root
    )
    eval_loader = ZarrBatchLoader(
        dataset, shuffle=False, result_dir=cache_root
    )
    return sources, train_loader, eval_loader, dataset.input_dims


def compute_metacell_zarr(source, meta_ids, args):
    meta_ids = np.asarray(meta_ids).astype(np.int64, copy=False)
    metacell_num = int(meta_ids.max()) + 1
    metacell_sizes = np.bincount(meta_ids, minlength=metacell_num)
    nonempty = metacell_sizes != 0
    expression_sums = np.zeros(
        (metacell_num, source.n_vars), dtype=np.float64
    )

    for start in range(0, source.n_obs, args.chunk_size):
        end = min(start + args.chunk_size, source.n_obs)
        counts = source.read_rows(start, end)
        _, normalized = _row_normalize(counts, dtype=np.float32)
        if sparse.issparse(normalized):
            np.log1p(normalized.data, out=normalized.data)
        else:
            np.log1p(normalized, out=normalized)

        chunk_ids = meta_ids[start:end]
        membership = sparse.csr_matrix(
            (
                np.ones(end - start, dtype=np.float32),
                (chunk_ids, np.arange(end - start)),
            ),
            shape=(metacell_num, end - start),
        )
        chunk_sums = membership @ normalized
        if sparse.issparse(chunk_sums):
            chunk_sums = chunk_sums.toarray()
        expression_sums += np.asarray(chunk_sums)

    data_meta = (
        expression_sums[nonempty]
        / metacell_sizes[nonempty, None]
    ).astype(np.float32)
    metacell_adata = ad.AnnData(data_meta)

    if args.type_key in source.obs_keys():
        annotations = source.get_obs_column(args.type_key)
        categorical = pd.Categorical(annotations)
        codes = categorical.codes
        valid = codes >= 0
        type_counts = np.zeros(
            (metacell_num, len(categorical.categories)), dtype=np.int64
        )
        np.add.at(
            type_counts,
            (meta_ids[valid], codes[valid]),
            1,
        )
        majority_codes = type_counts.argmax(axis=1)
        majority_types = np.asarray(categorical.categories)[majority_codes]
        metacell_adata.obs[args.type_key] = majority_types[nonempty]

    return metacell_adata
