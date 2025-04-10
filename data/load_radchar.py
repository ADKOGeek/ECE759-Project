#########################################################
# File used to load RadChar dataset into custom Dataset object and then split to train, validation, and test loaders
#########################################################

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import numpy as np

class RadCharDataset(Dataset):
    def __init__(self, data, p_type, param_labels):
        self.data = data
        self.p_type = p_type
        self.param_labels = param_labels
        self.transform = ToTensor

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, index):
        data = torch.transpose(torch.from_numpy(self.data[index, :, :]).squeeze(dim=0), 0, 1)
        param_labels = torch.from_numpy(self.param_labels[index,:])
        p_type = torch.from_numpy(self.p_type[index,:])
        sample = {'data':data,'rad_params':param_labels,'class_label':p_type}

        return sample
    

def load_data(batch_size):
    with h5py.File('./data/RadChar-Tiny.h5', 'r') as f:
        #load data
        h5_iqs = f['iq']
        h5_labels = f['labels']
        loaded_h5_iqs = h5_iqs[...] # raw IQ data
        loaded_h5_labels = np.asarray(h5_labels[...].tolist()) # Label data (signal index, type, # pulses, pulse width (seconds), time_delay (seconds), pulse repetition interval (seconds), SNR (dB))

        #extract IQ components
        real_part = np.expand_dims(np.real(loaded_h5_iqs).astype(np.float32), 2)
        imag_part = np.expand_dims(np.imag(loaded_h5_iqs).astype(np.float32), 2)
        raw_data = np.concatenate([real_part, imag_part], 2)

        #calculate mean and variance of dataset for samples
        data_mean = np.mean(raw_data)
        data_std = np.std(raw_data)

        #normalize dataset
        data_normed = (raw_data - data_mean) / data_std

        #extract param data
        p_type = np.expand_dims(loaded_h5_labels[:,1].astype(np.int32), 1)
        num_pulses = loaded_h5_labels[:,2].astype(np.float32)
        pulse_width = loaded_h5_labels[:,3].astype(np.float32)
        time_delay = loaded_h5_labels[:,4].astype(np.float32)
        pulse_repetition = loaded_h5_labels[:,5].astype(np.float32)

        #normalize params between 0 and 1
        num_pulses = np.expand_dims(((num_pulses - np.min(num_pulses)) / (np.max(num_pulses) - np.min(num_pulses))), 1)
        pulse_width = np.expand_dims(((pulse_width - np.min(pulse_width)) / (np.max(pulse_width) - np.min(pulse_width))), 1)
        time_delay = np.expand_dims(((time_delay - np.min(time_delay)) / (np.max(time_delay) - np.min(time_delay))),1)
        pulse_repetition = np.expand_dims(((pulse_repetition - np.min(pulse_repetition)) / (np.max(pulse_repetition) - np.min(pulse_repetition))),1)

        #put labels back into one array along column axis
        param_labels = np.concatenate([num_pulses, pulse_width, time_delay, pulse_repetition], 1)

        #put dataset into data loaders
        dataset = RadCharDataset(data_normed, p_type, param_labels)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=0) #set num_workers to 64 for cloud resource
        val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader



