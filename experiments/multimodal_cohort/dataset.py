from multiprocessing import allow_connection_pickling
import os
import copy
import time
import numpy as np
import pandas as pd
import torch
from itertools import chain, combinations
from sklearn.model_selection import ShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

from multimodal_cohort.fetchers import *
from multimodal_cohort.utils import discretizer, extract_and_order_by

class MultimodalDataset(torch.utils.data.Dataset):
    """ Multimodal dataset
    """

    def __init__(self, idx_path, metadata_path=None, indices=None,
                 transform=None, on_the_fly_transform=None,
                 on_the_fly_inter_transform=None, overwrite=False):

        self.idx_per_mod = dict(np.load(idx_path, allow_pickle=True))
        self.modalities = list(self.idx_per_mod)
        self.metadata = (pd.read_table(metadata_path) if metadata_path
                         else None)
        n_samples = [len(self.idx_per_mod[key]) for key in self.modalities]
        if not all([n_samples[i] == n_samples[(i+1) % len(n_samples)]
                    for i in range(len(n_samples))]):
            raise ValueError("All modalities do not have the same number of"
                             "samples.")
        if self.metadata is not None and n_samples[0] != len(self.metadata):
            raise ValueError("The data and metadata do not have the same"
                             "number of samples.")
        if transform is not None and type(transform) is dict:
            if not all([k in self.modalities for k in transform.keys()]):
                raise ValueError("The transform should be either a function,"
                                 "or a dict with modalities as keys and"
                                 "function as values.")
        if (on_the_fly_transform is not None and
           type(on_the_fly_transform) is dict):
            if not all([k in self.modalities for k in on_the_fly_transform]):
                raise ValueError("The transform should be either a function,"
                                 "or a dict with modalities as keys and"
                                 "function as values.")
        self.n_samples = n_samples[0]
        self.indices = indices
        self.modality_subsets = list(chain.from_iterable(
            combinations(self.modalities, n) for n in range(
                1, len(self.modalities)+1)))
        self.idx_per_modality_subset = self.compute_idx_per_modality_subset()

        # if residualizers is not None:
        datasetdir = "/".join(idx_path.split("/")[:-1])
        column_names = {}
        for mod in self.modalities:
            columns = np.load(os.path.join(
                datasetdir, "{}_names.npy".format(mod)),
                                allow_pickle=True)
            columns = [col.replace("&", "_").replace("-", "_") for col in columns]
            column_names[mod] = columns

        data_path = idx_path.replace("idx", "data").replace(".npz", ".npy")
        data_path = data_path.replace("_train", "").replace("_test", "")
        if transform is not None and transform:
            transformed_data = {}
            data_path = data_path.replace(".npy", "_transformed.npy")
            for mod in self.modalities:
                mod_path = data_path.replace("multiblock", mod)
                if overwrite or not os.path.exists(mod_path):
                    orig_mod_path = mod_path.replace("_transformed", "")
                    data = np.load(orig_mod_path, mmap_mode="r")
                    if type(transform) == dict:
                        if mod in transform.keys():
                            columns = column_names[mod]
                            mod_metadata = pd.read_table(os.path.join(datasetdir, "{}_metadata.tsv".format(mod)))
                            df = pd.concat([mod_metadata,
                                            pd.DataFrame(data, columns=columns)],
                                            axis=1)
                            transformed_data[mod] = transform[mod](df)[columns].values
                        else:
                            transformed_data[mod] = data
                    else:
                        transformed_data[mod] = transform(data)
                    np.save(mod_path, transformed_data[mod])

        self.data = {}
        for mod in self.modalities:
            mod_path = data_path.replace("multiblock", mod)
            self.data[mod] = np.load(mod_path, mmap_mode="r+")
        self.on_the_fly_transform = on_the_fly_transform
        self.on_the_fly_inter_transform = on_the_fly_inter_transform

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.n_samples

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        idx_per_mod = {
            mod: self.idx_per_mod[mod][idx] for mod in self.modalities}
        ret = {
            mod: self.data[mod][int(idx_per_mod[mod])]
            for mod in self.modalities if idx_per_mod[mod] is not None}
        if self.metadata is not None:
            ret["metadata"] = self.metadata.iloc[idx].to_dict()
        if self.on_the_fly_transform is not None:
            transform = self.on_the_fly_transform
            for mod in self.modalities:
                if mod in ret.keys():
                    data = ret[mod]
                    data = torch.tensor(data)
                    if type(transform) == dict and mod in transform.keys():
                        ret[mod] = transform[mod](data)
                    elif type(transform) != dict:
                        ret[mod] = transform(data)
        if self.on_the_fly_inter_transform is not None:
            ret = self.on_the_fly_inter_transform(ret)
        label = 0
        if "asd" in ret["metadata"]:
            label = ret["metadata"]["asd"] - 1
        metadata = ret["metadata"]
        del ret["metadata"]
        return ret, label, metadata

    def compute_idx_per_modality_subset(self):
        all_idx = list(range(len(self)))
        idx_per_modality_subset = [[] for _ in self.modality_subsets]
        for idx in all_idx:
            modalities = []
            for mod in self.modalities:
                true_idx = idx
                if self.indices is not None:
                    true_idx = self.indices[idx]
                if self.idx_per_mod[mod][true_idx] is not None:
                    modalities.append(mod)
            for sub_idx, subset in enumerate(self.modality_subsets):
                if (all([mod in subset for mod in modalities]) and
                    all([mod in modalities for mod in subset])):
                    idx_per_modality_subset[sub_idx].append(idx)
                    break
        return idx_per_modality_subset

    def get_modality_proportions(self):
        return [len(sub_idx) / len(self) for sub_idx in self.idx_per_modality_subset]

    
class DataManager(object):
    """ Data manager that builds the datasets
    """
    available_datasets = ["hbn", "euaims"]
    fetchers = {
        "hbn": make_hbn_fetchers,
        "euaims": make_euaims_fetchers,
    }
    available_modalities = {
        "hbn": ["clinical", "rois"],
        "euaims": ["clinical", "rois"],
    }
    defaults = {
        "hbn": hbn_defaults,
        "euaims": euaims_defaults,
    }

    def __init__(self, dataset, datasetdir, modalities, transform=None,
                 on_the_fly_transform=None, on_the_fly_inter_transform=None,
                 test_size="defaults", validation=None, val_size=0.2,
                 stratify="defaults", discretize="defaults", seed="defaults",
                 overwrite=False, **fetcher_kwargs):
        if dataset not in self.available_datasets:
            raise ValueError("{} dataset is not available".format(dataset))
        if not all([mod in self.available_modalities[dataset]
                    for mod in modalities]):
            wrong_mods = [mod for mod in modalities
                          if mod not in self.available_modalities[dataset]]
            raise ValueError(
                "{} is not an available modality for {} dataset".format(
                    wrong_mods[0], dataset))

        if test_size == "defaults":
            test_size = self.defaults[dataset]["multiblock"]["test_size"]
        if not (test_size is None or (test_size >= 0 and test_size < 1)):
            raise ValueError("The test size must be in [0, 1) or None")
        if stratify == "defaults":
            stratify = self.defaults[dataset]["multiblock"]["stratify"]
        if discretize == "defaults":
            discretize = self.defaults[dataset]["multiblock"]["discretize"]
        if seed == "defaults":
            seed = self.defaults[dataset]["multiblock"]["seed"]
        if seed != int(seed):
            raise ValueError("The seed must be an integer")

        self.dataset = dataset
        self.modalities = modalities
        self.test_size = test_size

        if not os.path.isdir(datasetdir):
            os.makedirs(datasetdir)

        fetchers = self.fetchers[dataset](datasetdir)
        self.fetcher = fetchers["multiblock"](
            blocks=modalities, seed=seed, stratify=stratify,
            discretize=discretize, test_size=test_size, overwrite=overwrite,
            **fetcher_kwargs)

        idx_path = self.fetcher.train_input_path
        metadata_path = self.fetcher.train_metadata_path

        if validation is not None:
            assert (type(validation) is int) and (validation > 0)
            idx_per_mod = np.load(idx_path, mmap_mode="r", allow_pickle=True)
            metadata = pd.read_table(metadata_path)
            modalities = list(idx_per_mod)
            full_indices = []
            not_full_indices = []
            for idx in range(len(idx_per_mod[modalities[0]])):
                if True in [indices[idx] is None for indices in idx_per_mod.values()]:
                    not_full_indices.append(idx)
                else:
                    full_indices.append(idx)
            self.train_dataset = {}
            splitter = ShuffleSplit(
                validation, test_size=val_size, random_state=seed)
            y = None
            if stratify is not None:
                assert type(stratify) is list
                splitter = MultilabelStratifiedShuffleSplit(
                    validation, test_size=val_size, random_state=seed)
                y = metadata[stratify].iloc[full_indices].copy()
                for name in stratify:
                    if name in discretize:
                        y[name] = discretizer(y[name].values)
            splitted = splitter.split(full_indices, y)
            for fold, (train_idx, valid_idx) in enumerate(splitted):
                train_idx = np.array(train_idx.tolist() + not_full_indices)
                self.train_dataset[fold] = {}
                self.train_dataset[fold]["train"] = MultimodalDataset(
                    idx_path, metadata_path,
                    train_idx, transform, on_the_fly_transform,
                    on_the_fly_inter_transform, overwrite)
                self.train_dataset[fold]["valid"] = MultimodalDataset(
                    idx_path, metadata_path,
                    valid_idx, transform, on_the_fly_transform,
                    on_the_fly_inter_transform, overwrite)
                self.train_dataset[fold]["train_idx"] = train_idx
                self.train_dataset[fold]["valid_idx"] = valid_idx
            self.train_dataset["all"] = MultimodalDataset(
                idx_path, metadata_path, None,
                transform, on_the_fly_transform,
                on_the_fly_inter_transform, overwrite)
        else:
            self.train_dataset = MultimodalDataset(
                idx_path, metadata_path, None, transform,
                on_the_fly_transform, on_the_fly_inter_transform,
                overwrite)

        if test_size is None or test_size > 0:
            idx_path = self.fetcher.test_input_path
            metadata_path = self.fetcher.test_metadata_path
            self.test_dataset = MultimodalDataset(
                idx_path, metadata_path, None, transform,
                on_the_fly_transform, on_the_fly_inter_transform,
                overwrite)

    def __getitem__(self, key):
        if key not in ["train", "test"]:
            raise ValueError("The key must be 'train' or 'test'")
        if key == "test" and self.test_size == 0:
            raise ValueError("This dataset does not have test data")
        return self.train_dataset if key == "train" else self.test_dataset

    
class MissingModalitySampler(torch.utils.data.Sampler):
    """
    """
    def __init__(self, dataset, batch_size, indices=None, stratify=None,
                 discretize=None, seed=42):
        super().__init__(dataset)
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.stratify = stratify
        self.discretize = discretize
        self.seed = seed

    def __len__(self):
        size = 0
        for idx, _ in enumerate(self.dataset.modality_subsets):
            size += (len(self.dataset.idx_per_modality_subset[idx]) +
                     self.batch_size - 1) // self.batch_size
        return size

    def __iter__(self):
        idx_per_modality_subset = copy.deepcopy(
            self.dataset.idx_per_modality_subset)
        indices = []
        batch_idx = 0
        incomplete_batch_indices = []
        complete_batch_indices = []
        for idx, _ in enumerate(self.dataset.modality_subsets):
            local_batch_idx = 0
            mod_subset_idx = idx_per_modality_subset[idx]
            n_batchs = (len(mod_subset_idx) +
                         self.batch_size - 1) // self.batch_size
            if self.stratify is not None and n_batchs > 1:
                real_indices = mod_subset_idx.copy()
                if self.indices is not None:
                    real_indices = self.indices[mod_subset_idx].tolist()
                metadata = self.dataset.metadata.iloc[real_indices]
                splitter = MultilabelStratifiedKFold(
                    n_batchs, shuffle=True, random_state=self.seed)
                y = metadata[self.stratify].copy()
                for name in self.stratify:
                    if name in self.discretize:
                        y[name] = discretizer(y[name].values)
                splitted = splitter.split(mod_subset_idx, y)
            while ((len(mod_subset_idx) > 0 and
                    (self.stratify is None or n_batchs == 1)) or
                   (self.stratify is not None and n_batchs > 1 and
                    local_batch_idx < n_batchs)):
                size = min(len(mod_subset_idx),
                           self.batch_size)
                if size < self.batch_size:
                    incomplete_batch_indices.append(batch_idx)
                else:
                    complete_batch_indices.append(batch_idx)
                if self.stratify is None or n_batchs == 1:
                    new_indices = np.random.choice(
                        mod_subset_idx,
                        size=size, replace=False)
                    for i in new_indices:
                        mod_subset_idx.remove(i)
                else:
                    _, new_indices = next(splitted)
                    new_indices = np.array(mod_subset_idx)[new_indices]
                indices.append(new_indices)
                batch_idx += 1
                local_batch_idx += 1
        complete_order = np.random.choice(complete_batch_indices,
                                          size=len(complete_batch_indices),
                                          replace=False)
        incomplete_order = np.random.choice(incomplete_batch_indices,
                                          size=len(incomplete_batch_indices),
                                          replace=False)
        complete_indices = []
        if len(complete_order) > 0:
            complete_indices = np.array(indices, dtype="object")[complete_order].tolist()
        incomplete_indices = []
        if len(incomplete_order) > 0:
            incomplete_indices = np.array(indices, dtype="object")[incomplete_order].tolist()
        indices = complete_indices + incomplete_indices
        return iter(indices)