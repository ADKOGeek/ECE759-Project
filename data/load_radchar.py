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
        sample = {'data':data ,'label':label}

        return sample
    

def load_data(batch_size):
    with h5py.File('./data/RadChar-Tiny.h5', 'r') as f:
        h5_iqs = f['iq']
        h5_labels = f['labels']

        loaded_h5_iqs = h5_iqs[...] # raw IQ data
        loaded_h5_labels = h5_labels[...] # Label data (signal index, type, # pulses, pulse width (seconds), time_delay (seconds), pulse repetition interval (seconds), SNR (dB))

        dataset = RadCharDataset(loaded_h5_iqs, loaded_h5_labels)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataloader




