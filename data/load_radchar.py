import h5py
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
        sample = {'data':[np.real(self.data[index]), np.imag(self.data[index])] ,'label':self.labels[index]}
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




