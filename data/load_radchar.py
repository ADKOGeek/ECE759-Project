import h5py
from torch.utils.data import DataLoader, Dataset

with h5py.File('./RadChar-Tiny.h5', 'r') as f:
    h5_iqs = f['iq']
    h5_labels = f['labels']

    loaded_h5_iqs = h5_iqs[...] # raw IQ data
    loaded_h5_labels = h5_labels[...] # Label data (signal index, type, # pulses, pulse width (seconds), time_delay (seconds), pulse repetition interval (seconds), SNR (dB))

    one_label = loaded_h5_labels[0]

    print(one_label)
    print(one_label.dtype)
    print("Loaded Data")

    

