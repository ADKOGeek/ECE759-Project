#########################################################
# File used to load RadChar dataset into custom Dataset object and then split to train, validation, and test loaders
#########################################################

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import numpy as np

class RadCharDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = ToTensor

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, index):
        label = torch.from_numpy(np.asarray(self.labels[index].tolist())[1:6].astype(np.float32))
        data = torch.from_numpy(np.concatenate((np.real(self.data[index]).astype(np.float32).reshape(1,512), np.imag(self.data[index]).astype(np.float32).reshape(1,512)), axis=0))
        sample = {'data':data,'rad_params':label[1:],'class_label':label[0].to(torch.int32)}

        return sample
    

def load_data(batch_size):
    with h5py.File('./data/RadChar-Tiny.h5', 'r') as f:
        h5_iqs = f['iq']
        h5_labels = f['labels']

        loaded_h5_iqs = h5_iqs[...] # raw IQ data
        loaded_h5_labels = h5_labels[...] # Label data (signal index, type, # pulses, pulse width (seconds), time_delay (seconds), pulse repetition interval (seconds), SNR (dB))

        dataset = RadCharDataset(loaded_h5_iqs, loaded_h5_labels)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader, test_loader



