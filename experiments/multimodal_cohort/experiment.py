from ctypes import resize
import random
import os
import numpy as np
import pandas as pd
import json
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
#from utils.BaseExperiment import BaseExperiment

from modalities.multimodal_cohort import Clinical, Rois

from multimodal_cohort.dataset import MultimodalDataset, DataManager


from multimodal_cohort.networks.VAE import VAE
from multimodal_cohort.networks.networks import Encoder, Decoder

from utils.BaseExperiment import BaseExperiment

class Residualizer:
    def __init__(self, by_continuous, by_categorical):
        self.by_continuous = by_continuous
        self.by_categorical = by_categorical
        self.estimators = None

    def fit(self, df, columns_to_residualize):
        formula = ("{} ~ " + " + ".join(self.by_continuous) + " + " +
                   " + ".join(["C({})".format(cat) for cat in self.by_categorical]))
        self.columns_to_residualize = columns_to_residualize
        self.estimators = []
        for col in columns_to_residualize:
            est = sm.OLS.from_formula(formula.format(col), data=df)
            res = est.fit()
            self.estimators.append(res)

    def transform(self, df):
        if self.estimators is None:
            raise ValueError("You must fit the residualizer before transforming data")
        new_df = df.copy()
        for col_idx, col in enumerate(self.columns_to_residualize):
            new_df[col] -= self.estimators[col_idx].predict(exog=df[self.by_continuous + self.by_categorical])
        return new_df

    def fit_transform(self, df, columns_to_residualize):
        self.fit(df, columns_to_residualize)
        return self.transform(df)
    
    def inverse_transform(self, df):
        if self.estimators is None:
            raise ValueError("You must fit the residualizer before transforming data")
        new_df = df.copy()
        for col_idx, col in enumerate(self.columns_to_residualize):
            new_df[col] += self.estimators[col_idx].predict(exog=df[self.by_continuous + self.by_categorical])
        return new_df
    


class MultimodalExperiment(BaseExperiment):
    def __init__(self, flags, alphabet):
        super().__init__(flags)
        # self.flags = flags
        # self.name = flags.name
        # self.dataset_name = flags.dataset
        self.num_modalities = flags.num_mods
        self.alphabet = alphabet
        self.residualize_by = dict()
        # self.residualize_by["rois"] = dict(
        #     continuous=["age"],
        #     categorical=["sex", "site"]
        # )
        # self.plot_img_size = torch.Size((3, 28, 28))
        # self.font = ImageFont.truetype('FreeSerif.ttf', 38)
        self.flags.num_features = len(alphabet)

        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()

        self.mm_vae = self.set_model()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()
        self.test_samples = self.get_test_samples()
        self.eval_metric = accuracy_score
        self.paths_fid = self.set_paths_fid()
        self.labels = ['ASD']
    
    @classmethod
    def get_experiment(cls, flags_file, alphabet_file, checkpoint_file):
        flags = torch.load(flags_file)
        with open(alphabet_file, "rt") as of:
            alphabet = str("".join(json.load(of)))
        experiment = MultimodalExperiment(flags, alphabet)
        checkpoint = torch.load(checkpoint_file)
        experiment.mm_vae.load_state_dict(checkpoint)
        return experiment, flags

    def set_model(self):
        model = VAE(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device)
        return model

    def set_modalities(self):
        if type(self.flags.style_dim) is int:
            self.flags.style_dim = [self.flags.style_dim] * self.num_modalities
        elif len(self.flags.style_dim) != self.num_modalities:
            self.flags.style_dim = [self.flags.style_dim[0]] * self.num_modalities
        mods = [Clinical, Rois]
        mods = [mods[m](self.flags.input_dim[m], Encoder(self.flags, m),
                        Decoder(self.flags, m), self.flags.class_dim,
                        self.flags.style_dim[m], self.flags.likelihood) for m in range(self.num_modalities)]
        mods_dict = {m.name: m for m in mods}
        return mods_dict

    def set_scalers(self, dataset, residualizers=None):
        scalers = {}
        for mod in self.modalities:
            scaler = StandardScaler()
            all_training_data = []
            all_metadata = []
            for data in dataset:
                if mod in data[0].keys():
                    all_training_data.append(data[0][mod])
                    all_metadata.append(data[2])
            if residualizers is not None and mod in residualizers:
                print("resid before scaling")
                all_training_data = np.asarray(all_training_data)
                df = pd.DataFrame.from_records(all_metadata)
                columns = np.load(os.path.join(self.flags.datasetdir, self.modalities[mod].names_file), allow_pickle=True)
                columns = [col.replace("&", "_").replace("-", "_") for col in columns]
                df = pd.concat([df, pd.DataFrame(all_training_data, columns=columns)], axis=1)
                all_training_data = self.residualizers[mod].transform(df)[columns].values
            scaler.fit(all_training_data)
            scalers[mod] = scaler
        return scalers

    def set_residualizers(self, dataset):
        if "residualize_by" not in vars(self):
            print("Residualize by doesnt exist")
            self.residualizers = dict()
            pass
        residualizers = {}
        for mod in self.modalities:
            if mod in self.residualize_by.keys():
                all_training_data = []
                all_metadata = []
                for idx, data in enumerate(dataset):
                    if mod in data[0].keys():
                        all_training_data.append(data[0][mod])
                        all_metadata.append(data[2])
                all_training_data = np.asarray(all_training_data)
                # if mod in self.scalers.keys():
                #     all_training_data = self.scalers[mod].transform(all_training_data)
                df = pd.DataFrame.from_records(all_metadata)
                columns = np.load(os.path.join(self.flags.datasetdir, self.modalities[mod].names_file), allow_pickle=True)
                columns = [col.replace("&", "_").replace("-", "_") for col in columns]
                df = pd.concat([df, pd.DataFrame(all_training_data, columns=columns)], axis=1)
                residualizers[mod] = Residualizer(by_continuous=self.residualize_by[mod]["continuous"],
                                                  by_categorical=self.residualize_by[mod]["categorical"])
                residualizers[mod].fit(df, columns)
        return residualizers

    def unsqueeze_0(self, x):
        return x.unsqueeze(0)

    def set_dataset(self):
        train = []
        test = []
        scalers = []
        residualizers = []
        validation = None
        n_models = 1
        test_size = 0.2
        if "num_models" in vars(self.flags) and self.flags.num_models > 1:
            validation = self.flags.num_models
            test_size = 0
            n_models = validation
            
        manager = DataManager(self.flags.dataset, self.flags.datasetdir,
                                list(self.modalities), overwrite=True,
                                allow_missing_blocks=self.flags.allow_missing_blocks,
                                validation=validation, test_size=test_size)
        for model_idx in range(n_models):
            train_dataset = manager.train_dataset
            train_idx = None
            test_input_path = manager.fetcher.test_input_path
            test_metadata_path = manager.fetcher.test_metadata_path
            test_idx = None
            if validation is not None:
                train_dataset = train_dataset[model_idx]["train"]
                train_idx = train_dataset[model_idx]["train_idx"]
                test_input_path = manager.fetcher.train_input_path
                test_metadata_path = manager.fetcher.train_metadata_path
                test_idx = train_dataset[model_idx]["test_idx"]
            residualizer = self.set_residualizers(train_dataset)
            residualizers.append(residualizer)
            scalers.append(self.set_scalers(train_dataset, residualizer))
            self.transform = {mod: transforms.Compose([
                self.unsqueeze_0,
                scaler.transform,
                transforms.ToTensor(),
                torch.squeeze]) for mod, scaler in scalers[model_idx].items()}
            transform = {mod: residualizer.transform for mod, residualizer in residualizers[model_idx].items()}
            # transform = None
            train.append(MultimodalDataset(manager.fetcher.train_input_path,
                                    manager.fetcher.train_metadata_path,
                                    train_idx,
                                    on_the_fly_transform=self.transform,
                                    transform=transform))
            test.append(MultimodalDataset(test_input_path, test_metadata_path,
                                    test_idx,
                                    on_the_fly_transform=self.transform,
                                    transform=transform))
        if n_models == 1:
            train = train[0]
            test = test[0]
            residualizers = residualizers[0]
            scalers = scalers[0]
        self.dataset_train = train
        self.dataset_test = test
        self.residualizers = residualizers
        self.scalers = scalers
        print(len(train))
        print(len(test))

    def set_optimizer(self):
        # optimizer definition
        total_params = sum(p.numel() for p in self.mm_vae.parameters())
        params = list(self.mm_vae.parameters());
        print('num parameters: ' + str(total_params))
        optimizer = optim.Adam(params,
                               lr=self.flags.initial_learning_rate,
                               betas=(self.flags.beta_1,
                               self.flags.beta_2))
        self.optimizer = optimizer

    def set_rec_weights(self):
        rec_weights = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            rec_weights[mod.name] = 1.0
        return rec_weights

    def set_style_weights(self):
        weights = {m: self.flags.beta_style for m in self.modalities}
        return weights

    def get_transform_cohort(self):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform

    def get_test_samples(self, num_images=2):
        n_test = len(self.dataset_test)
        samples = []
        for i in range(num_images):
            ix = random.randint(0, n_test-1)
            sample, _, _ = self.dataset_test[ix]
            for key in sample.keys():
                if sample[key] is not None:
                    sample[key] = sample[key].to(self.flags.device)
            samples.append(sample)
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))
