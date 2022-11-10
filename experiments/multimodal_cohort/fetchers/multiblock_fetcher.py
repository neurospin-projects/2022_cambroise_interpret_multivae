import os
import numpy as np
import pandas as pd
from collections import namedtuple
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit

from multimodal_cohort.utils import discretizer, extract_and_order_by


Item = namedtuple("Item", ["train_input_path", "test_input_path",
                           "train_metadata_path", "test_metadata_path"])


def fetch_multiblock_wrapper(datasetdir, defaults):
    """ Fetcher wrapper for multiblock data

    Parameters
    ----------
    datasetdir: string
        path to the folder in which to save the data
    defaults: dict
        default values for the wrapped function

    Returns
    -------
    fetcher: function
        corresponding fetcher

    """

    def fetch_multiblock(
            blocks=defaults["blocks"], test_size=defaults["test_size"],
            stratify=defaults["stratify"], discretize=defaults["discretize"],
            seed=defaults["seed"], allow_missing_blocks=defaults["allow_missing_blocks"],
            overwrite=False, **kwargs):
        """ Fetches and preprocesses multi block data

        Parameters
        ----------
        blocks: list of strings, see defaults
            blocks of data to fetch, all must be in the key list of FETCHERS
        test_size: float, see defaults
            proportion of the dataset to keep for testing. Preprocessing models
            will only be fitted on the training part and applied to the test
            set. You can specify not to use a testing set by setting it to 0.
            None means it uses the test defined in the openBHB Challenge.
        stratify: string or list of strings, see defaults
            variables to consider when splitting the data, to create balanced
            train / test sets
        discretize: list of strings, see defaults
            variables to discretize for the iterative stratification for
            splitting the data into train / test
        seed: int, default 42
            random seed to split the data into train / test
        remove_outliers: bool, see defaults
            if True removes the subjects considered as outliers
        remove_nans: bool or float, see defaults
            if True remove the subjects that contains missing values (np.nan).
            If it is a float, replaces the missing values by this value
        overwrite: bool, default False
            if True forces the fetchers to write all the data, even if it
            already exists
        kwargs: dict
            additional arguments to be passed to each fetcher indivudally.
            Keys are the name of the fetchers, and values are a dictionnary
            containing arguments and the values for this fetcher

        Returns
        -------
        item: namedtuple
            a named tuple containing 'train_input_path', 'train_metadata_path',
            and 'test_input_path', 'test_metadata_path'
        """

        path = os.path.join(datasetdir, "multiblock_idx_train.npz")
        metadata_path = os.path.join(datasetdir, "metadata_train.tsv")
        path_test, metadata_path_test = None, None
        if test_size is None or test_size > 0:
            path_test = os.path.join(datasetdir, "multiblock_idx_test.npz")
            metadata_path_test = os.path.join(datasetdir, "metadata_test.tsv")

        subject_column_name = "participant_id"
        block_paths = {}
        for block in blocks:
            block_paths[block] = {
                "data": os.path.join(datasetdir, block + "_data.npy"),
                "subjects": os.path.join(datasetdir, block + "_subjects.npy")
            }
        block_paths["metadata"] = os.path.join(datasetdir, "metadata.tsv")
        if not os.path.isfile(path) or overwrite:
            subj_per_block = {}

            subjects_test = None

            for block in blocks:
                new_subj = np.load(block_paths[block]["subjects"], allow_pickle=True)
                subj_per_block[block] = new_subj
            # Remove subjects that arent in all the channels
            common_subjects = sorted(list(
                set.intersection(*map(set, subj_per_block.values()))))
            if allow_missing_blocks:
                all_subjects = set.union(*map(set, subj_per_block.values()))
                other_subjects = sorted(list(all_subjects.difference(common_subjects)))
                # missing_modalities = [[] for _ in other_subjects]
            index = {}
            for block in blocks:
                subjects = subj_per_block[block].tolist()
                new_index = []
                for sub in common_subjects:
                    new_index.append(subjects.index(sub))
                if allow_missing_blocks:
                    for sub in other_subjects:
                        if sub in subjects:
                            new_index.append(subjects.index(sub))
                        else:
                            new_index.append(None)
                            # missing_modalities[sub_idx].append(block)
                index[block] = np.array(new_index)


            metadata = pd.read_table(block_paths["metadata"])
            # all_metadata = extract_and_order_by(metadata, subject_column_name, all_subjects)
            common_metadata = extract_and_order_by(metadata, subject_column_name, common_subjects)
            index_train_subjects = list(range(len(common_subjects)))
            if test_size is not None and test_size > 0:
                splitter = ShuffleSplit(1, test_size=test_size,
                                        random_state=seed)
                y = None
                if stratify is not None:
                    splitter = MultilabelStratifiedShuffleSplit(
                        1, test_size=test_size, random_state=seed)
                    if not type(stratify) is list:
                        stratify = [stratify]
                    y = common_metadata[stratify].copy()
                    for name in stratify:
                        if name in discretize:
                            y[name] = discretizer(y[name].values)
                    # if allow_missing_blocks:
                        # y["missing_modalities"] = [
                        #     np.unique(missing_modalities).tolist().index(miss)
                        #     for miss in missing_modalities]
                index_train_subjects, index_test_subjects = next(
                    splitter.split(common_subjects, y))
            elif test_size is None:
                index_train_subjects = [
                    idx for idx, sub in enumerate(common_subjects)
                    if sub in subjects_train]
                index_test_subjects = [
                    idx for idx, sub in enumerate(common_subjects)
                    if sub in subjects_test]                

            subjects_train = np.array(common_subjects)[index_train_subjects]
            if test_size is None or test_size > 0:
                subjects_test = np.array(common_subjects)[index_test_subjects]
            if allow_missing_blocks:
                subjects_train = np.array(subjects_train.tolist() + other_subjects)
                index_train_subjects = (list(index_train_subjects)
                                        + list(range(len(common_subjects), len(all_subjects))))
            index_train = {}
            index_test = {}
            for block in blocks:
                index_train[block] = index[block][index_train_subjects]
                if test_size is None or test_size > 0:
                    index_test[block] = index[block][index_test_subjects]
            # Loading the metadata
            metadata_train = extract_and_order_by(metadata, subject_column_name, subjects_train)
            if test_size is None or test_size > 0:
                metadata_test = extract_and_order_by(metadata, subject_column_name, subjects_test)

            # Saving
            np.savez(path, **index_train)
            metadata_train.to_csv(metadata_path, index=False, sep="\t")
            if test_size is None or test_size > 0:
                np.savez(path_test, **index_test)
                metadata_test.to_csv(metadata_path_test, index=False, sep="\t")
        return Item(train_input_path=path, test_input_path=path_test,
                    train_metadata_path=metadata_path,
                    test_metadata_path=metadata_path_test)

    return fetch_multiblock
