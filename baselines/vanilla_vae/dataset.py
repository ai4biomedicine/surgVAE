import os
import random
import numpy as np

import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, outcomes, types):
        self.data_tensor = data_tensor
        self.indices = range(len(self))
        self.outcomes = outcomes
        self.types = types
        #log normalization
        self.data_tensor = torch.log(self.data_tensor+1)



    def __getitem__(self, index1):

        data1 = self.data_tensor[index1]
        outcome = self.outcomes[index1]
        type = self.types[index1]

        return data1, outcome, type

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(batch_size, data, outcomes, types):
    data = np.array(data)
    # Check if data type is object
    if data.dtype == np.object_:
        print("Data contains non-numeric values. Attempting to convert...")

        try:
            # Attempt to convert data to float
            data = data.astype(np.float32)
        except ValueError as ve:
            print(f"Failed to convert data to float: {ve}")
    data = torch.from_numpy(data).float()

    outcomes = torch.from_numpy(np.array(outcomes)).float()
    types = torch.from_numpy(np.array(types))
    train_kwargs = {'data_tensor':data, 'outcomes':outcomes, 'types':types}
    dset = CustomTensorDataset


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                                shuffle=True, drop_last=False)

    data_loader = train_loader
    return data_loader
