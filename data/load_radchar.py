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
        data = torch.transpose(torch.from_numpy(self.data[index, :, :]).squeeze(dim=0), 0, 1)
        sample = {'data':data,'rad_params':label[1:],'class_label':label[0].to(torch.int32)}

        return sample
    

def load_data(batch_size):
    with h5py.File('./data/RadChar-Tiny.h5', 'r') as f:
        #load data
        h5_iqs = f['iq']
        h5_labels = f['labels']
        loaded_h5_iqs = h5_iqs[...] # raw IQ data
        loaded_h5_labels = h5_labels[...] # Label data (signal index, type, # pulses, pulse width (seconds), time_delay (seconds), pulse repetition interval (seconds), SNR (dB))

        #extract IQ components
        real_part = np.expand_dims(np.real(loaded_h5_iqs).astype(np.float32), 2)
        imag_part = np.expand_dims(np.imag(loaded_h5_iqs).astype(np.float32), 2)
        raw_data = np.concatenate([real_part, imag_part], 2)

        #calculate mean and variance of dataset for samples
        data_mean = np.mean(raw_data)
        data_std = np.std(raw_data)

        #normalize dataset
        data_normed = (raw_data - data_mean) / data_std

        #put dataset into data loaders
        dataset = RadCharDataset(data_normed, loaded_h5_labels)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=0) #set num_workers to 64 for cloud resource
        val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader



