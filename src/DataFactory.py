import kagglehub
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class CrabDataset(Dataset):

    def __init__(self, df):

        self.df = df

        # Replace sex values M, F, I with numerical values
        self.df['Sex'] = self.df['Sex'].replace('M', 0)
        self.df['Sex'] = self.df['Sex'].replace('F', 1)
        self.df['Sex'] = self.df['Sex'].replace('I', 2)

        # Cast values
        self.df = self.df.astype(np.float32)

        # Convert to torch
        self.matrix = torch.tensor(self.df.values)

    def __getitem__(self, idx):
        return self.matrix[idx, 0:-1], self.matrix[idx, -1]

    def __len__(self):
        return len(self.matrix)


class DataFactory:

    def __init__(self, config):

        self.config = config

        # Download
        self.path = kagglehub.dataset_download("sidhus/crab-age-prediction")

        # Open as pandas dataframe
        self.df = pd.read_csv(self.path + '/CrabAgePrediction.csv')

        self.dataset = CrabDataset(self.df)

    def get_loaders(self):

        train_set, val_set, test_set = torch.utils.data.random_split(self.dataset, (self.config['settings']['train_size'],
                                                                                    self.config['settings']['val_size'],
                                                                                    self.config['settings']['test_size']))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config['training_space']['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config['training_space']['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.config['training_space']['batch_size'], shuffle=True)

        return train_loader, val_loader, test_loader



