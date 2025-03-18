import torch
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.input = nn.Linear(1024, 512) #placeholder architecture with linear layers
        self.output = nn.Linear(512, 7)

    def forward(self, x):
        x = x.reshape(self.batch_size, 1, 1024)
        x = self.input(x)
        x = self.output(x)
        return x
